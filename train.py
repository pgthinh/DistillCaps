import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch
import torch._dynamo
import torch.distributed as dist
import transformers
import yaml
from torch.utils.data import Dataset
from transformers import (HfArgumentParser, Trainer, TrainingArguments,
                          set_seed, trainer_pt_utils, trainer_utils)

from dataset import AudioDataset, collate_fn
from models.loae import CedLlama7BCaptionModel
from utils.utils import get_cpu_mem_info, get_gpu_info, get_tcp_address

torch._dynamo.config.suppress_errors = True


class LoaeTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss

    # def log(self, logs: Dict[str, float]) -> None:
    #     log_str_list = []
    #     for key, value in logs.items():
    #         log_str_list.append("{} {}".format(key, round(value, 12)))
    #     logging.info(
    #         "epoch {}/{}, step {}, {}".format(
    #             round(self.state.epoch, 3),
    #             self.state.num_train_epochs,
    #             self.state.global_step,
    #             ",".join(log_str_list),
    #         )
    #     )

    #     # print cpu and gpu info
    #     if self.state.global_step % 1000 == 0:
    #         gpu_id = int(os.getenv("CUDA_VISIBLE_DEVICES"))
    #         gpu_info = get_gpu_info(gpu_id)
    #         cpu_mem = get_cpu_mem_info()
    #         logging.info(
    #             "step {} gpu {} info: mem({}/{}/{})MB, rate:{}% cpu mem:{}/{:.3f}/{:.3f} GB".format(
    #                 self.state.global_step,
    #                 gpu_id,
    #                 gpu_info[0],
    #                 gpu_info[1],
    #                 gpu_info[2],
    #                 gpu_info[3],
    #                 cpu_mem["count"],
    #                 cpu_mem["virt_mem"],
    #                 cpu_mem["res_mem"],
    #             )
    #         )

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    _eval_dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        self._memory_tracker.start()
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        model = self._wrap_model(self.model, training=False, dataloader=eval_dataloader)
        num_examples = self.num_examples(eval_dataloader)
        model.eval()

        total_loss = 0.0
        total_samples = 0
        teach_loss = 0.0
        student_loss = 0.0
        distil_loss = 0.0
        contrastive_loss = 0.0
        with torch.no_grad():
            for i, batch_data in enumerate(eval_dataloader):
                (loss, outputs) = self.compute_loss(model, batch_data, return_outputs=True)
                if torch.isfinite(loss):
                    batch_size = trainer_pt_utils.find_batch_size(batch_data)
                    total_samples += batch_size
                    total_loss += loss.item() * batch_size
                    teach_loss += outputs[2].item() * batch_size
                    student_loss += outputs[3].item() * batch_size
                    distil_loss += outputs[4].item() * batch_size
                    # contrastive_loss += outputs[5].item() * batch_size

        eval_time = time.time() - start_time
        avg_loss = total_loss / (total_samples if total_samples > 0 else 1)
        avg_teach_loss = teach_loss / (total_samples if total_samples > 0 else 1)
        avg_student_loss = student_loss / (total_samples if total_samples > 0 else 1)
        avg_distil_loss = distil_loss / (total_samples if total_samples > 0 else 1)
        avg_contrastive_loss = contrastive_loss / (total_samples if total_samples > 0 else 1)

        metrics = {"{}_loss".format(metric_key_prefix): (avg_teach_loss + avg_student_loss + avg_distil_loss), "time": eval_time}
        # save eval loss for each Evaluation Dataset
        self.state.log_history.append({"{}_loss".format(metric_key_prefix): avg_teach_loss + avg_student_loss + avg_distil_loss})

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        metrics.update(
            trainer_utils.speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=num_examples,
                num_steps=math.ceil(num_examples / total_batch_size),
            )
        )
        logging.info(
            "{} for epoch {}, samples {}/{}, steps {}, loss {}, teach_loss {}, student_loss {}, distil_loss {}, contrastive_loss {}".format(
                metric_key_prefix,
                self.state.epoch,
                total_samples,
                num_examples,
                self.state.global_step,
                avg_loss,
                avg_teach_loss,
                avg_student_loss,
                avg_distil_loss,
                avg_contrastive_loss,
            )
        )

        self._memory_tracker.stop_and_update_metrics(metrics)
        return metrics


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    config_path: Optional[str] = field(default=None, metadata={"help": "setting files"})
    out_dir: Optional[str] = field(
        default=None, metadata={"help": "output dir for model"}
    )
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "train and dev data file in dir"}
    )
    resume_checkpoint: Optional[str] = field(
        default="none", metadata={"help": "resume the model from checkpoint"}
    )
    rank: Optional[int] = field(
        default=0, metadata={"help": "the rank for distributed training"}
    )
    world_size: Optional[int] = field(
        default=1, metadata={"help": "the total gpu number for distributed training"}
    )
    init_model_path: Optional[str] = field(
        default="none", metadata={"help": "init the model weight by other model"}
    )
    rag: Optional[bool] = field(
        default=False, metadata={"help": "use rag to generate text"}
    )
    def __post_init__(self):
        if self.config_path is None:
            raise ValueError("config path should not none")


def main():
    parser = HfArgumentParser(DataTrainingArguments)
    data_args = parser.parse_args_into_dataclasses()[0]
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    transformers.logging.set_verbosity_info()
    logging.info(data_args)

    # gpu_id = int(os.getenv("CUDA_VISIBLE_DEVICES"))
    # data_args.rank = data_args.rank - 1
    # logging.info("using gpu id:{}".format(gpu_id))

    # load config
    set_seed(20)
    with open(data_args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    ddp_dir = os.path.join(data_args.out_dir, "ddp")
    os.makedirs(ddp_dir, exist_ok=True)

    dataset_batch_size = config["dataset_conf"]["batch_size"]
    num_workers = config["dataset_conf"]["num_workers"]
    num_epoch = config["epochs"]
    training_args = TrainingArguments(
        output_dir=data_args.out_dir,
        seed=20,
        do_train=True,
        do_eval=True,
        dataloader_num_workers=num_workers,
        remove_unused_columns=False,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        # save_steps=1,
        # eval_steps=1,
        save_total_limit=5,
        greater_is_better=False,
        metric_for_best_model="eval_valid_loss",
        load_best_model_at_end=True,
        per_device_train_batch_size=dataset_batch_size,
        per_device_eval_batch_size=dataset_batch_size,
        save_safetensors=False,
        logging_dir=os.path.join(data_args.out_dir, "log"),
        max_grad_norm=config["clip_grad"],
        gradient_accumulation_steps=config["acc_grad"],
    )
    training_args = training_args.set_optimizer(
        config["optim_args"]["name"],
        learning_rate=config["optim_args"]["lr"],
        weight_decay=config["optim_args"]["weight_decay"],
    )
    training_args = training_args.set_lr_scheduler(
        "cosine", num_epoch, warmup_ratio=config["warmup_radio"]
    )
    training_args = training_args.set_logging(
        strategy="steps", steps=100, report_to=[], level="info", first_step=True
    )
    logging.info(training_args)

    train_data_file = os.path.join(
        os.path.join(data_args.data_dir, "train_caps.json")
    )
    val_data_file = os.path.join(
        os.path.join(data_args.data_dir, "val_caps.json")
    )

    logging.info(train_data_file)
    logging.info(val_data_file)

    sample_rate, max_length = (
        config["dataset_conf"]["sample_rate"],
        config["dataset_conf"]["max_len"],
    )

    train_dataset = AudioDataset(
        train_data_file, sample_rate=sample_rate, max_length=max_length, rag=data_args.rag, encoder_type=config['encoder']
    )
    val_dataset = AudioDataset(
        val_data_file, sample_rate=sample_rate, max_length=max_length, rag=data_args.rag, encoder_type=config['encoder']
    )
        
    eval_dataset = {"valid": val_dataset}

    model = CedLlama7BCaptionModel(config)
    if data_args.init_model_path != "none":
        init_state = torch.load(data_args.init_model_path, map_location="cpu")
        model.load_state_dict(init_state)
        # release memory
        del init_state
        logging.info("Loaded init weight from {}".format(data_args.init_model_path))

    logging.info(model)
    model.print_trainable_parameters()
    model.print_module_parameters()

    trainer = LoaeTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        compute_metrics=None,
    )

    checkpoint = None
    if data_args.resume_checkpoint != "none":
        checkpoint = data_args.resume_checkpoint

    trainer.train(resume_from_checkpoint=checkpoint)
    logging.info("Training done.")


if __name__ == "__main__":
    main()
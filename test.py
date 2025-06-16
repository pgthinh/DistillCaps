import argparse
import logging
import os

import torch
from torch import nn
import yaml
from torch.utils.data import DataLoader

from dataset import AudioDataset, collate_fn
from models.loae import CedLlama7BCaptionModel
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoFeatureExtractor, ClapAudioModel

def _exec_model_test(
    model_test, data_loader, res_dir, use_nucleus_sampling=False, beam_size=3, top_p=0.9
):
    model_test.eval()
    predict_result = {}
    with torch.no_grad():
        for i, batch_data in enumerate(data_loader):
            output = model_test.generate(
                samples=batch_data,
                use_nucleus_sampling=use_nucleus_sampling,
                num_beams=beam_size,
                top_p=top_p,
            )
            print(output[0])
            print("===========================================")
            audio_names = batch_data["names"]
            for j in range(len(audio_names)):
                predict_result[audio_names[j]] = output[j].replace("\n", " ")

    if use_nucleus_sampling:
        result_file = os.path.join(res_dir, "top-p_{}.txt".format(top_p))
    else:
        result_file = os.path.join(res_dir, "beam_{}.txt".format(beam_size))
    with open(result_file, "w+", encoding="utf8") as writer:
        for key in sorted(predict_result.keys()):
            writer.write(key + "\t" + predict_result[key] + "\n")
    return result_file


def do_beam_search():
    for beam in [1, 2, 3]:
        logging.info(
            "Start test for {}, beam {}, samples {}, total batch:{}".format(
                args.test_data, beam, len(test_dataset), len(test_dataloader)
            )
        )
        _exec_model_test(
            model,
            test_dataloader,
            args.result_dir,
            use_nucleus_sampling=False,
            beam_size=beam,
        )


def do_nucleus_sampling():
    for p in [0.85, 0.9, 0.95]:
        logging.info(
            "Start test for {}, top_p {}, samples {}, total batch:{}".format(
                args.test_data, p, len(test_dataset), len(test_dataloader)
            )
        )
        _exec_model_test(
            model, test_dataloader, args.result_dir, use_nucleus_sampling=True, top_p=p
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test with your special model")
    parser.add_argument("--config_path", required=True, help="config file")
    parser.add_argument("--checkpoint", required=True, help="checkpoint file")
    parser.add_argument("--test_data", required=True, help="test data file")
    parser.add_argument("--result_dir", required=True, help="asr result file")
    parser.add_argument("--rag", required=False, help="use rag model")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.info(args)

    os.makedirs(args.result_dir, exist_ok=True)
    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = CedLlama7BCaptionModel(config)
    logging.info(model)

    use_cuda = torch.cuda.is_available()
    # if use_cuda:
    #     logging.info("Checkpoint: loading from checkpoint %s for GPU" % args.checkpoint)
    #     checkpoint = torch.load(args.checkpoint)
    # else:
    logging.info("Checkpoint: loading from checkpoint %s for CPU" % args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    model.load_state_dict(checkpoint)
    logging.info("Finishing loading weight.")

    # model = nn.DataParallel(model, device_ids=[0, 1], output_device=1)

    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to("cuda")
    logging.info("start model recognise in {}".format(device))

    sample_rate, max_length, batch_size = (
        config["dataset_conf"]["sample_rate"],
        config["dataset_conf"]["max_len"],
        16,
    )
    test_dataset = AudioDataset(
        args.test_data, sample_rate=sample_rate, max_length=max_length, 
        rag=args.rag, encoder_type=config['encoder']#, do_train=False
    )
    # tokenizer = LlamaTokenizer.from_pretrained(
    #         "llama_weight/LLaMA-7B-HF"
    #     )
    # tokenizer.pad_token = tokenizer.unk_token
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        sampler=None,
        shuffle=False,
        collate_fn=collate_fn, #(tokenizer=tokenizer),
        drop_last=False,
    )

    do_beam_search()
    do_nucleus_sampling()

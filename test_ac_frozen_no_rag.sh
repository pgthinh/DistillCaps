torchrun --nnodes=1 --nproc_per_node=1 --master_port=25030 test.py \
    --config_path conf/finetune_ac.yaml \
    --checkpoint ./weight/ac_finetune_frozen_lora_rag_ds_same_FFT_teach_distill/checkpoint-31570/pytorch_model.bin \
    --test_data data/AudioCaps/ds_same/test_caps.json \
    --rag False \
    --result_dir result/ac_finetune_frozen_lora_no_rag_ds_same_FFT_teach_distill \

torchrun --nnodes=1 --nproc_per_node=4 --master_port=25005 train.py \
    --config_path conf/finetune_ac.yaml \
    --out_dir weight/ac_finetune_frozen_lora_rag_ds_same_FFT_teach_distill \
    --data_dir data/AudioCaps/ds_same \
    --rag True\
    --world_size 1


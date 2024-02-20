#!/bin/bash

# Finetune

python3 multitask_classifier.py --use_gpu --task sst --f lora_4_sst --lr 1e-3 --option finetune --enable_per_layer_finetune --batch_size 32 --lora 4 --epochs 20 --load_pretrain epoch_3_further-pretraining-sst-para-sts-lin.pt $@
python3 multitask_classifier.py --use_gpu --task sts --f lora_4_sts --lr 1e-3 --option finetune --enable_per_layer_finetune --batch_size 32 --lora 4 --epochs 20 --load_pretrain epoch_3_further-pretraining-sst-para-sts-lin.pt $@
python3 multitask_classifier.py --use_gpu --task lin --f lora_4_lin --lr 1e-3 --option finetune --enable_per_layer_finetune --batch_size 32 --lora 4 --epochs 20 --load_pretrain epoch_3_further-pretraining-sst-para-sts-lin.pt $@
python3 multitask_classifier.py --use_gpu --task para --f lora_4_para --lr 1e-3 --option finetune --enable_per_layer_finetune --batch_size 32 --lora 4 --epochs 20 --load_pretrain epoch_3_further-pretraining-sst-para-sts-lin.pt $@

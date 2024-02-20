#!/bin/bash

# Frozen
python3 multitask_classifier.py --use_gpu --task para --f baseline_pretrain --lr 1e-3 --option pretrain --batch_size 24 $@

# Finetune
python3 multitask_classifier.py --use_gpu --task para --f baseline_finetune --lr 1e-5 --option finetune --batch_size 24 $@

# Finetune and per-layer LR
python3 multitask_classifier.py --use_gpu --task para --f per_layer_finetune --lr 1e-5 --option finetune --enable_per_layer_finetune --batch_size 24 $@

# Finetune and per-layer LR and lora 1,2,4,8
#python3 multitask_classifier.py --use_gpu --task para --f lora_1_para --lr 1e-3 --option finetune --enable_per_layer_finetune --batch_size 24 --lora 1 --epochs 10 $@
#python3 multitask_classifier.py --use_gpu --task para --f lora_2_para --lr 1e-3 --option finetune --enable_per_layer_finetune --batch_size 24 --lora 2 --epochs 10 $@
#python3 multitask_classifier.py --use_gpu --task para --f lora_4_para --lr 1e-3 --option finetune --enable_per_layer_finetune --batch_size 24 --lora 4 --epochs 10 $@
#python3 multitask_classifier.py --use_gpu --task para --f lora_8_para --lr 1e-3 --option finetune --enable_per_layer_finetune --batch_size 24 --lora 8 --epochs 10 $@

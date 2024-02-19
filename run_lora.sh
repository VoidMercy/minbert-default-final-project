#!/bin/bash

# Finetune
python3 multitask_classifier.py --use_gpu --task sst --f lora_1_sst --lr 1e-3 --option finetune --enable_per_layer_finetune --batch_size 32 --lora 1 --epochs 20 $@
python3 multitask_classifier.py --use_gpu --task sts --f lora_1_sts --lr 1e-3 --option finetune --enable_per_layer_finetune --batch_size 32 --lora 1 --epochs 20 $@
python3 multitask_classifier.py --use_gpu --task lin --f lora_1_lin --lr 1e-3 --option finetune --enable_per_layer_finetune --batch_size 32 --lora 1 --epochs 20 $@

python3 multitask_classifier.py --use_gpu --task sst --f lora_2_sst --lr 1e-3 --option finetune --enable_per_layer_finetune --batch_size 32 --lora 2 --epochs 20 $@
python3 multitask_classifier.py --use_gpu --task sts --f lora_2_sts --lr 1e-3 --option finetune --enable_per_layer_finetune --batch_size 32 --lora 2 --epochs 20 $@
python3 multitask_classifier.py --use_gpu --task lin --f lora_2_lin --lr 1e-3 --option finetune --enable_per_layer_finetune --batch_size 32 --lora 2 --epochs 20 $@

python3 multitask_classifier.py --use_gpu --task sst --f lora_4_sst --lr 1e-3 --option finetune --enable_per_layer_finetune --batch_size 32 --lora 4 --epochs 20 $@
python3 multitask_classifier.py --use_gpu --task sts --f lora_4_sts --lr 1e-3 --option finetune --enable_per_layer_finetune --batch_size 32 --lora 4 --epochs 20 $@
python3 multitask_classifier.py --use_gpu --task lin --f lora_4_lin --lr 1e-3 --option finetune --enable_per_layer_finetune --batch_size 32 --lora 4 --epochs 20 $@

python3 multitask_classifier.py --use_gpu --task sst --f lora_8_sst --lr 1e-3 --option finetune --enable_per_layer_finetune --batch_size 32 --lora 8 --epochs 20 $@
python3 multitask_classifier.py --use_gpu --task sts --f lora_8_sts --lr 1e-3 --option finetune --enable_per_layer_finetune --batch_size 32 --lora 8 --epochs 20 $@
python3 multitask_classifier.py --use_gpu --task lin --f lora_8_lin --lr 1e-3 --option finetune --enable_per_layer_finetune --batch_size 32 --lora 8 --epochs 20 $@

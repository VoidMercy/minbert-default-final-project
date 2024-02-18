#!/bin/bash

# Finetune
python3 multitask_classifier.py --use_gpu --task sst --f per_layer_finetune --lr 1e-5 --option finetune --enable_per_layer_finetune
python3 multitask_classifier.py --use_gpu --task para --f per_layer_finetune --lr 1e-5 --option finetune --enable_per_layer_finetune
python3 multitask_classifier.py --use_gpu --task sts --f per_layer_finetune --lr 1e-5 --option finetune --enable_per_layer_finetune
python3 multitask_classifier.py --use_gpu --task lin --f per_layer_finetune --lr 1e-5 --option finetune --enable_per_layer_finetune

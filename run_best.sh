#!/bin/bash

# Finetune
#python3 multitask_classifier.py --use_gpu --task sst --f best --lr 1e-5 --option finetune --enable_per_layer_finetune --batch_size 64 $@
python3 multitask_classifier.py --use_gpu --task sts --f best --lr 1e-5 --option finetune --enable_per_layer_finetune --batch_size 32 $@
#python3 multitask_classifier.py --use_gpu --task lin --f best --lr 1e-5 --option finetune --enable_per_layer_finetune --batch_size 64 $@
python3 multitask_classifier.py --use_gpu --task para --f best --lr 1e-5 --option finetune --enable_per_layer_finetune --batch_size 24 $@

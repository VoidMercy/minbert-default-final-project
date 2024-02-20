#!/bin/bash

# Finetune
python3 multitask_classifier.py --use_gpu --task sst --f multitask_sst --lr 1e-5 --option finetune --enable_per_layer_finetune --batch_size 16 --multitask $@

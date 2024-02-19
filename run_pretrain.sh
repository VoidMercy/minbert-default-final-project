#!/bin/bash

# Pretrain
python3 multitask_classifier.py --use_gpu --task none --f pretrain-sst --lr 1e-5 --option finetune --pretrain_dataset sst --enable_pretrain --batch_size 4

#!/bin/bash

# Pretrain
python3 multitask_classifier.py --use_gpu --task sst --f baseline_pretrain --lr 1e-3 --option pretrain
python3 multitask_classifier.py --use_gpu --task para --f baseline_pretrain --lr 1e-3 --option pretrain
python3 multitask_classifier.py --use_gpu --task sts --f baseline_pretrain --lr 1e-3 --option pretrain
python3 multitask_classifier.py --use_gpu --task lin --f baseline_pretrain --lr 1e-3 --option pretrain

# Finetune
python3 multitask_classifier.py --use_gpu --task sst --f baseline_finetune --lr 1e-5 --option finetune
python3 multitask_classifier.py --use_gpu --task para --f baseline_finetune --lr 1e-5 --option finetune
python3 multitask_classifier.py --use_gpu --task sts --f baseline_finetune --lr 1e-5 --option finetune
python3 multitask_classifier.py --use_gpu --task lin --f baseline_finetune --lr 1e-5 --option finetune

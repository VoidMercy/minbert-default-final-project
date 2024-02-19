#!/bin/bash

# Pretrain SST-only
python3 multitask_classifier.py --use_gpu --task sst --f further-pretrain-sst --lr 2e-5 --option finetune --pretrain_dataset sst --enable_pretrain further-pretraining-sst.pt --pretrain_epochs 20
python3 multitask_classifier.py --use_gpu --task sst --f further-pretrain-sst --lr 2e-5 --option finetune --load_pretrain further-pretraining-sst.pt

# Pretrain Para-only
python3 multitask_classifier.py --use_gpu --task none --f further-pretrain-para --lr 2e-5 --option finetune --pretrain_dataset para --enable_pretrain further-pretraining-para.pt --pretrain_epochs 20
python3 multitask_classifier.py --use_gpu --task para --f further-pretrain-para --lr 2e-5 --option finetune --load_pretrain further-pretraining-para.pt

# Pretrain STS-only
python3 multitask_classifier.py --use_gpu --task none --f further-pretrain-sts --lr 2e-5 --option finetune --pretrain_dataset sts --enable_pretrain further-pretraining-sts.pt --pretrain_epochs 20
python3 multitask_classifier.py --use_gpu --task sts --f further-pretrain-sts --lr 2e-5 --option finetune --load_pretrain further-pretraining-sts.pt

# Pretrain Lin-only
python3 multitask_classifier.py --use_gpu --task none --f further-pretrain-lin --lr 2e-5 --option finetune --pretrain_dataset lin --enable_pretrain further-pretraining-lin.pt --pretrain_epochs 20
python3 multitask_classifier.py --use_gpu --task lin --f further-pretrain-lin --lr 2e-5 --option finetune --load_pretrain further-pretraining-lin.pt

# Pretrain All-four
python3 multitask_classifier.py --use_gpu --task none --f further-pretrain-sst-para-sts-lin --lr 2e-5 --option finetune --pretrain_dataset sst-para-sts-lin --enable_pretrain further-pretraining-sst-para-sts-lin.pt --pretrain_epochs 20

python3 multitask_classifier.py --use_gpu --task sst --f further-pretrain-sst-para-sts-lin-task-sst --lr 2e-5 --option finetune --load_pretrain further-pretraining-sst-para-sts-lin.pt
python3 multitask_classifier.py --use_gpu --task para --f further-pretrain-sst-para-sts-lin-task-para --lr 2e-5 --option finetune --load_pretrain further-pretraining-sst-para-sts-lin.pt
python3 multitask_classifier.py --use_gpu --task sts --f further-pretrain-sst-para-sts-lin-task-sts --lr 2e-5 --option finetune --load_pretrain further-pretraining-sst-para-sts-lin.pt
python3 multitask_classifier.py --use_gpu --task lin --f further-pretrain-sst-para-sts-lin-task-lin --lr 2e-5 --option finetune --load_pretrain further-pretraining-sst-para-sts-lin.pt

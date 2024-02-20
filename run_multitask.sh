#!/bin/bash

# Multitask finetune then finetune
# python3 multitask_classifier.py --use_gpu --task sst --f multitask_nopretrain_sst --lr 1e-5 --option finetune --enable_per_layer_finetune --batch_size 32 --multitask $@
python3 multitask_classifier.py --use_gpu --task sts --f multitask_nopretrain_sts --lr 1e-5 --option finetune --enable_per_layer_finetune --batch_size 32 --multitask $@
python3 multitask_classifier.py --use_gpu --task lin --f multitask_nopretrain_lin --lr 1e-5 --option finetune --enable_per_layer_finetune --batch_size 32 --multitask $@
python3 multitask_classifier.py --use_gpu --task para --f multitask_nopretrain_para --lr 1e-5 --option finetune --enable_per_layer_finetune --batch_size 24 --multitask $@

# Multitask finetinue then finetune with pre-training
python3 multitask_classifier.py --use_gpu --task sst --f multitask_pretrain_sst --lr 1e-5 --option finetune --enable_per_layer_finetune --batch_size 32 --multitask --multitask_filepath multitask_pretrain.pt --load_pretrain epoch_3_further-pretraining-sst-para-sts-lin.pt $@
python3 multitask_classifier.py --use_gpu --task sts --f multitask_pretrain_sts --lr 1e-5 --option finetune --enable_per_layer_finetune --batch_size 32 --multitask --multitask_filepath multitask_pretrain.pt --load_pretrain epoch_3_further-pretraining-sst-para-sts-lin.pt $@
python3 multitask_classifier.py --use_gpu --task lin --f multitask_pretrain_lin --lr 1e-5 --option finetune --enable_per_layer_finetune --batch_size 32 --multitask --multitask_filepath multitask_pretrain.pt --load_pretrain epoch_3_further-pretraining-sst-para-sts-lin.pt $@
python3 multitask_classifier.py --use_gpu --task para --f multitask_pretrain_para --lr 1e-5 --option finetune --enable_per_layer_finetune --batch_size 24 --multitask --multitask_filepath multitask_pretrain.pt --load_pretrain epoch_3_further-pretraining-sst-para-sts-lin.pt $@

# Do the above but with and lora 4
python3 multitask_classifier.py --use_gpu --task sst --f multitask_nopretrain4_sst --lr 1e-3 --option finetune --enable_per_layer_finetune --batch_size 32 --multitask --multitask_filepath multitask_nopretrain4.pt --lora 4 $@
python3 multitask_classifier.py --use_gpu --task sts --f multitask_nopretrain4_sts --lr 1e-3 --option finetune --enable_per_layer_finetune --batch_size 32 --multitask --multitask_filepath multitask_nopretrain4.pt --lora 4 $@
python3 multitask_classifier.py --use_gpu --task lin --f multitask_nopretrain4_lin --lr 1e-3 --option finetune --enable_per_layer_finetune --batch_size 32 --multitask --multitask_filepath multitask_nopretrain4.pt --lora 4 $@
python3 multitask_classifier.py --use_gpu --task para --f multitask_nopretrain4_para --lr 1e-3 --option finetune --enable_per_layer_finetune --batch_size 24 --multitask --multitask_filepath multitask_nopretrain4.pt --lora 4 $@

python3 multitask_classifier.py --use_gpu --task sst --f multitask_pretrain4_sst --lr 1e-3 --option finetune --enable_per_layer_finetune --batch_size 32 --multitask --multitask_filepath multitask_pretrain4.pt --load_pretrain epoch_3_further-pretraining-sst-para-sts-lin.pt --lora 4 $@
python3 multitask_classifier.py --use_gpu --task sts --f multitask_pretrain4_sts --lr 1e-3 --option finetune --enable_per_layer_finetune --batch_size 32 --multitask --multitask_filepath multitask_pretrain4.pt --load_pretrain epoch_3_further-pretraining-sst-para-sts-lin.pt --lora 4 $@
python3 multitask_classifier.py --use_gpu --task lin --f multitask_pretrain4_lin --lr 1e-3 --option finetune --enable_per_layer_finetune --batch_size 32 --multitask --multitask_filepath multitask_pretrain4.pt --load_pretrain epoch_3_further-pretraining-sst-para-sts-lin.pt --lora 4 $@
python3 multitask_classifier.py --use_gpu --task para --f multitask_pretrain4_para --lr 1e-3 --option finetune --enable_per_layer_finetune --batch_size 24 --multitask --multitask_filepath multitask_pretrain4.pt --load_pretrain epoch_3_further-pretraining-sst-para-sts-lin.pt --lora 4 $@

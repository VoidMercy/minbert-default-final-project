#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
module load anaconda3
module load cuda
module load cudnn
conda run --cwd /home/alexlin0/minbert-default-final-project -n pytorch python3 multitask_classifier.py --use_gpu --task lin --f lora_4_lin --lr 1e-3 --option finetune --enable_per_layer_finetune --batch_size 32 --lora 4 --epochs 20 --load_pretrain epoch_3_further-pretraining-sst-para-sts-lin.pt

#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --mem-per-cpu=8G
#SBATCH --time=2-0
module load anaconda3
module load cuda
module load cudnn
conda run --cwd /home/alexlin0/minbert-default-final-project -n pytorch python3 multitask_classifier.py --use_gpu --task none --f f-pretrain-para2 --lr 1e-5 --option finetune --pretrain_dataset para --enable_pretrain f-pretraining-para2.pt --pretrain_epochs 15 --pretrain_batch_size 10 --pretrain_path /farmshare/user_data/alexlin0

#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1-20
module load anaconda3
module load cuda
module load cudnn
conda run --cwd /home/alexlin0/minbert-default-final-project -n pytorch python3 multitask_classifier.py --use_gpu --task sts --f d18 --lr 1e-5 --option finetune --enable_per_layer_finetune --batch_size 24 --epochs 20 --sts_dev_out predictions/sts_dev_3.csv --load_pretrain /farmshare/user_data/alexlin0/epoch_29_f-pretraining-sts2.pt --multitask --test
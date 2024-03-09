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
conda run --cwd /home/alexlin0/minbert-default-final-project -n pytorch python3 multitask_classifier.py --use_gpu --task para --f lora_4_para --lr 1e-3 --option finetune --enable_per_layer_finetune --batch_size 16 --lora 4 --epochs 20 --load_pretrain epoch_3_further-pretraining-sst-para-sts-lin.pt

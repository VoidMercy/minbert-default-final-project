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
conda run --cwd /home/alexlin0/minbert-default-final-project -n pytorch python3 multitask_classifier.py --use_gpu --task sts --f multitask_nopretrain4_sts --lr 1e-3 --option finetune --enable_per_layer_finetune --batch_size 24 --multitask --multitask_filepath multitask_nopretrain4.pt --lora 4

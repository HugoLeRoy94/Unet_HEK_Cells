#!/bin/bash -l
#SBATCH --job-name=unet-training
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:01:00  # Set a proper time limit
#SBATCH --output=tensorboard-log-%J.out

module load gcc python openmpi py-tensorflow

ipnport=$(shuf -i8000-9999 -n1)
tensorboard --logdir logs --port=${ipnport} --bind_all &

python unet_training.py

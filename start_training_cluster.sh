#!/bin/bash -l
#SBATCH --job-name=unet-training
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:02:00  # Set a proper time limit
#SBATCH --output=log-%J.out

module load gcc python openmpi py-tensorflow py-keras

python multiclass_training_2.py

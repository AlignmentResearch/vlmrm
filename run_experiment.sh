#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --partition=single
#SBATCH --job-name=vlmrm
#SBATCH --output=logs/job_%j.log

mkdir -p logs

echo "Make sure you have WANDB_API_KEY set in .env"
export $(cat .env | xargs)

vlmrm train "$(cat config.yaml)"

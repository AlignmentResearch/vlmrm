#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --partition=single
#SBATCH --job-name=vlmrm
#SBATCH --output=logs/job_%j.log

if [ "$#" -ne 1 ]; then
    echo "Usage: sbatch run_experiment.sh [config path]"
    exit 1
fi

mkdir -p logs

echo "Make sure you have WANDB_API_KEY set in .env"
export $(cat .env | xargs)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlmrm

vlmrm train "$(cat $1)"

#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=2:00:00
#SBATCH --partition=single
#SBATCH --job-name=vlmrm
#SBATCH --output=logs/job_%j.log

models=("clip" "viclip" "s3d" "gpt4")

for model in "${models[@]}"; do
    python src/evaluation/evaluator.py -t data/evaluation/data.csv -m "$model" -r logit,projection -a 0.0,0.25,0.50,0.75,1.0 -n 32 -e standardized_improved --standardize
done

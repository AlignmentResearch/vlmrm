#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --partition=single
#SBATCH --job-name=vlmrm
#SBATCH --output=logs/job_%j.log

# Initialize default seed value
seed=""
port=""

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
    --seed)
        seed="$2"
        shift
        ;;
    --port)
        port="$2"
        shift
        ;;
    *)
        config_path="$1"
        ;;
    esac
    shift
done

if [ -z "$seed" ]; then
    echo "Seed value is required."
    exit 1
fi

if [ -z "$port" ]; then
    echo "Port value is required."
    exit 1
fi

if [ -z "$config_path" ]; then
    echo "Config path is required."
    exit 1
fi

# Create logs directory
mkdir -p logs

# Inform about WANDB_API_KEY
echo "Make sure you have WANDB_API_KEY set in .env"

# Export variables from .env file
export $(cat .env | xargs)

# Replace seed value in config
sed -i "s/seed: .*/seed: $seed/" "$config_path"

# Export the MASTER_PORT environment variable based on seed
export MASTER_PORT="$port"

# Execute the training command with the modified config file
vlmrm train "$(cat $config_path)"

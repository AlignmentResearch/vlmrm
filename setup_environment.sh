#!/bin/bash

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init bash
source ~/.bashrc

conda create -n vlmrm python=3.9 -y
conda activate vlmrm

pip install -e ".[dev]"

echo "Make sure you have WANDB_API_KEY set in .env"
echo "WANDB_API_KEY=..." >>.env

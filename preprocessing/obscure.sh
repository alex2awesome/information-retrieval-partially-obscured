#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=400G
#SBATCH --partition=isi

source /home1/spangher/.bashrc
cd /project/jonmay_231/spangher/Projects/information-retrieval-partially-obscured/information-retrieval-partially-obscured/preprocessing
conda activate vllm-py310

python obscure.py

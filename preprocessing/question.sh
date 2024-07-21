#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=60:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --constraint=a100-80gb
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=200G
#SBATCH --partition=gpu

echo "Successfully allocated resources"
module load python/3.11
pip install -r requirements.txt

python3 question.py

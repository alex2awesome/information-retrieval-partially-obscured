#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --mem-per-gpu=10GB
#SBATCH --cpus-per-gpu=20
#SBATCH --partition=gpu

module load python/3.11
pip install -r requirements.txt
python3 data_vllm_70b.py
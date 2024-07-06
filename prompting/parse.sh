#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80gb
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=100G
#SBATCH --partition=gpu

module load python/3.11
pip install -r requirements.txt
python3 parse.py --name sources_data_70b__110000_120000.txt


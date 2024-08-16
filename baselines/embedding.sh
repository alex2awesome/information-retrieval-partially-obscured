#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80gb
#SBATCH --mem=200GB
#SBATCH --cpus-per-gpu=10
#SBATCH --partition=gpu


module load conda
source conda activate rr

python3 embedding.py 

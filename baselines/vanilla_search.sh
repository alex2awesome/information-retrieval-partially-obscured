#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=200GB
#SBATCH --cpus-per-gpu=10
#SBATCH --partition=gpu

module load conda
source conda activate rr
conda env update -f env.yaml
pip install -r requirements.txt

python3 vanilla_search.py 

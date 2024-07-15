#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80gb
#SBATCH --mem=200GB
#SBATCH --cpus-per-gpu=10
#SBATCH --partition=gpu


module load python/3.11
module load conda
source conda activate rr

conda install -c pytorch -c nvidia faiss-gpu=1.8.0
conda list
pip install -r requirements.txt
python3 vanilla.py 

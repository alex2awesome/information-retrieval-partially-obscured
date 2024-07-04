#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --constraint=a100-80gb
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=100G
#SBATCH --partition=gpu

export HF_HOME="/project/jonmay_231/spangher/huggingface_cache"
export HF_DATASETS_CACHE="/project/jonmay_231/spangher/huggingface_cache"

# Run the Python script
python /project/jonmay_231/spangher/Projects/information-retrieval-partially-obscured/information-retrieval-partially-obscured/prompting/obscure_sources.py
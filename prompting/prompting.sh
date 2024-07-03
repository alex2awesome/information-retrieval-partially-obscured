#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu

export HF_HOME="/project/jonmay_231/spangher/huggingface_cache"
export HF_DATASETS_CACHE="/project/jonmay_231/spangher/huggingface_cache"

# Run the Python script
python /project/jonmay_231/spangher/Projects/information-retrieval-partially-obscured/information-retrieval-partially-obscured/prompting/obscure_sources.py
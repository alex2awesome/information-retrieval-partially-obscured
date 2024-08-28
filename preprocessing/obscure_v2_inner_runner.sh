#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=60:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=3
#SBATCH --mem=100G
#SBATCH --partition=isi


source /home1/spangher/.bashrc
conda activate vllm-py310

start_idx=$1
step=$2
iterations=$3
iterations=$((iterations + 1))
end_idx=$((start_idx + step))

for ((i=0; i<iterations; i++)); do
    python obscure_v2.py \
      --start_idx ${start_idx} \
      --end_idx ${end_idx} \
      --output_file v2_sources_obscured.jsonl

    start_idx=${end_idx}
    end_idx=$((start_idx + step))
done

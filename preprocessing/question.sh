#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --constraint=a100-80gb
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=100G
#SBATCH --partition=gpu

module load python/3.11
pip install -r requirements.txt

# List of source files to process
source_files=(
    "sources_data_70b__0_10000"
    "sources_data_70b__10000_20000"
    "sources_data_70b__80000_90000"
    "sources_data_70b__90000_100000"
    "sources_data_70b__100000_110000"
    "sources_data_70b__110000_120000"
    "sources_data_70b__120000_130000"
    "sources_data_70b__200000_200100"
    "sources_data_70b__200000_205000"
    "sources_data_70b__205000_210000"
    "sources_data_70b__210000_220000"
    "sources_data_70b__220000_230000"
    "sources_data_70b__230000_240000"
    "sources_data_70b__240000_250000"
    "sources_data_70b__310000_320000"
    "sources_data_70b__320000_330000"
    "sources_data_70b__330000_340000"
)

# Process each source file
for file in ""; do
    python3 lucas_prompting.py --source_file ""
    echo "Processed "
done

echo "All files processed successfully!"
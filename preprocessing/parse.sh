#!/bin/bash

module load python/3.11
pip install -r requirements.txt

start_idx=$1
step=$2
iterations=$3
end_idx=$((start_idx + step))

for ((i=0; i<iterations; i++)); do
    python3 parse.py --start_idx ${start_idx} --end_idx ${end_idx}
    start_idx=${end_idx}
    end_idx=$((start_idx + step))
done








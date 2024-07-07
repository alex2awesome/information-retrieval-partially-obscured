#!/bin/bash

module load python/3.11
pip install -r requirements.txt
python3 parse.py --name sources_data_70b__110000_120000.txt


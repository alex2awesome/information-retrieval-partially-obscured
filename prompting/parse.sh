#!/bin/bash

module load python/3.11
pip install -r requirements.txt
python3 parse.py --name sources_data_70b__0_10000
python3 parse.py --name sources_data_70b__10000_20000
python3 parse.py --name sources_data_70b__80000_90000
python3 parse.py --name sources_data_70b__90000_100000
python3 parse.py --name sources_data_70b__100000_110000
python3 parse.py --name sources_data_70b__110000_120000
python3 parse.py --name sources_data_70b__120000_130000
python3 parse.py --name sources_data_70b__200000_200100
python3 parse.py --name sources_data_70b__200000_205000
python3 parse.py --name sources_data_70b__205000_210000
python3 parse.py --name sources_data_70b__210000_220000

python3 parse.py --name sources_data_70b__220000_230000
python3 parse.py --name sources_data_70b__230000_240000
python3 parse.py --name sources_data_70b__220000_250000
python3 parse.py --name sources_data_70b__310000_320000
python3 parse.py --name sources_data_70b__320000_330000
python3 parse.py --name sources_data_70b__330000_340000











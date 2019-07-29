#!/bin/sh
source .env/bin/activate
pip install -r requirements.txt
python train.py -a "CNN1" -g "adam" -b "50"

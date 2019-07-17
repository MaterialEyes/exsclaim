#!/bin/sh
conda env create -f environment.yml
conda activate exsclaim_env



FILE=TextDetector/models/read_sflabel_5_CNN150_adam.pt
if [ ! -f "$FILE" ]; then
    mkdir TextDetector/models
    gdown  -O TextDetector/models/read_sflabel_5_CNN150_adam.pt https://drive.google.com/uc?id=1pkBWn0Ss0c9TeAgsOBFqFFJZ-5ntxzpb
fi

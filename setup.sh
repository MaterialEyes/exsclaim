#!/bin/sh
conda env create -f environment.yml
conda activate exsclaim_env



text_model=TextDetector/models/read_sflabel_5_CNN150_adam.pt
text_dir=TextDetector/models
if [ ! -f "$text_model" ]; then
    if [ ! -d "$text_dir" ]; then
        mkdir TextDetector/models
    fi
    gdown  -O TextDetector/models/read_sflabel_5_CNN150_adam.pt https://drive.google.com/uc?id=1pkBWn0Ss0c9TeAgsOBFqFFJZ-5ntxzpb
else
    echo "Already downloaded TextDetector model"
fi

object_model=ObjectDetector/checkpoints/snapshot930.ckpt
object_dir=ObjectDetector/checkpoints
if [ ! -f "$object_model" ]; then
    if [ ! -d "$object_dir" ] ; then
        mkdir ObjectDetector/checkpoints
    fi
    gdown -O ObjectDetector/checkpoints/snapshot930.ckpt.zip https://drive.google.com/uc?id=1xWxqQGDH_szfCe8eWDBwTcjzCmq7Bnf1
    unzip ObjectDetector/checkpoints/snapshot930.ckpt.zip -d ObjectDetector/checkpoints
    rm ObjectDetector/checkpoints/snapshot930.ckpt.zip
else
    echo "Already downloaded ObjectDetector model"
fi


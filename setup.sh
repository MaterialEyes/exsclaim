#!/bin/sh
conda env create -f environment.yml
conda activate exsclaim_env



text_model=exsclaim/text/models/read_sflabel_5_CNN150_adam.pt
text_dir=exsclaim/text/models
if [ ! -f "$text_model" ]; then
    if [ ! -d "$text_dir" ]; then
        mkdir exsclaim/text/models
    fi
    gdown  -O exsclaim/text/models/read_sflabel_5_CNN150_adam.pt https://drive.google.com/uc?id=1pkBWn0Ss0c9TeAgsOBFqFFJZ-5ntxzpb
else
    echo "Already downloaded TextDetector model"
fi

object_model=exsclaim/objects/checkpoints/snapshot930.ckpt
object_dir=exsclaim/objects/checkpoints
if [ ! -f "$object_model" ]; then
    if [ ! -d "$object_dir" ] ; then
        mkdir exsclaim/objects/checkpoints
    fi
    gdown -O exsclaim/objects/checkpoints/snapshot930.ckpt.zip https://drive.google.com/uc?id=1xWxqQGDH_szfCe8eWDBwTcjzCmq7Bnf1
    unzip exsclaim/objects/checkpoints/snapshot930.ckpt.zip -d exsclaim/objects/checkpoints
    rm exsclaim/objects/checkpoints/snapshot930.ckpt.zip
else
    echo "Already downloaded ObjectDetector model"
fi


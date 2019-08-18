#!/bin/sh

# Detect the platform 
OS="`uname`"
case $OS in
  'Linux')
    OS='linux'
    alias ls='ls --color=auto'
    ;;
  'WindowsNT')
    OS='windows'
    ;;
  'Darwin')
    OS='mac'
    ;;
  *) ;;
esac

# Get command line input (regular pipeline = 'min', or train/test models = 'dev')
USE=${1:-min}

# Create conda environment 
envfile="envs/"$OS"/"$USE".yml"
conda env create -f "$envfile";

# Get the environment name from the yaml file
envname=$(grep "name: *" $envfile | sed -n -e 's/name: //p')
source activate "$envname";

# Download the text detector model
text_model=exsclaim/text/models/read_sflabel_5_CNN150_adam.pt
text_dir=exsclaim/text/models
if [ ! -f "$text_model" ]; then
    if [ ! -d "$text_dir" ]; then
        mkdir exsclaim/text/models
    fi
    # Changle link below if a new model is trained
    gdown  -O exsclaim/text/models/read_sflabel_5_CNN150_adam.pt https://drive.google.com/uc?id=1pkBWn0Ss0c9TeAgsOBFqFFJZ-5ntxzpb
else
    echo "Already downloaded TextDetector model"
fi

# Download the object detector model
object_model=exsclaim/objects/checkpoints/snapshot930.ckpt
object_dir=exsclaim/objects/checkpoints
if [ ! -f "$object_model" ]; then
    if [ ! -d "$object_dir" ] ; then
        mkdir exsclaim/objects/checkpoints
    fi
    # Change link below if a new model is trained
    gdown -O exsclaim/objects/checkpoints/snapshot930.ckpt.zip https://drive.google.com/uc?id=1xWxqQGDH_szfCe8eWDBwTcjzCmq7Bnf1
    unzip exsclaim/objects/checkpoints/snapshot930.ckpt.zip -d exsclaim/objects/checkpoints
    rm exsclaim/objects/checkpoints/snapshot930.ckpt.zip
else
    echo "Already downloaded ObjectDetector model"
fi

python -m spacy download en_core_web_sm

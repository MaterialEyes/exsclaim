#!/bin/bash
CWD="$(pwd)"
FIGURE_MODELS_SAVE_DIR=$CWD/exsclaim/figures/checkpoints/
mkdir $FIGURE_MODELS_SAVE_DIR

# classifier_model.pt
gdown https://docs.google.com/uc\?export\=download\&id\=16BxXXGJyHfMtzhDEufIwMkcDoBftZosh -O $FIGURE_MODELS_SAVE_DIR
# object_detection_model.pt
gdown https://docs.google.com/uc\?export\=download\&id\=1HbzvNvhPcvUKh_RCddjKIvdRFt6T4PEH -O $FIGURE_MODELS_SAVE_DIR
# text_recognition_model.pt
gdown https://docs.google.com/uc\?export\=download\&id\=1p9miOnR_dUxO5jpIv1hKtsZQuFHAaQpX -O $FIGURE_MODELS_SAVE_DIR
# scale_bar_detection_model.pt
gdown https://docs.google.com/uc\?export\=download\&id\=11Kfu9xEbjG0Mw2u0zCJlKy_4C6K21qqg -O $FIGURE_MODELS_SAVE_DIR
# scale_label_recognition_model.pt
gdown https://docs.google.com/uc\?export\=download\&id\=1AND30sQpSrph2CGl86k2aWNqNp-0vLnR -O $FIGURE_MODELS_SAVE_DIR

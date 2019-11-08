# # Download the text detector model
# text_model=exsclaim/imagetexts/models/read_sflabel_5_CNN150_adam.pt
# text_dir=exsclaim/imagetexts/models
# if [ ! -f "$text_model" ]; then
#     if [ ! -d "$text_dir" ]; then
#         mkdir exsclaim/imagetexts/models
#     fi
#     # Changle link below if a new model is trained
#     gdown  -O exsclaim/imagetexts/models/read_sflabel_5_CNN150_adam.pt https://drive.google.com/uc?id=1pkBWn0Ss0c9TeAgsOBFqFFJZ-5ntxzpb
# else
#     echo "Already downloaded TextDetector model"
# fi

# Download the object detector model
object_model=exsclaim/figures/models/checkpoints/snapshot20000.ckpt
object_dir=exsclaim/figures/models/checkpoints
if [ ! -f "$object_model" ]; then
    if [ ! -d "$object_dir" ] ; then
        mkdir exsclaim/figures/models/checkpoints
    fi
    # Change link below if a new model is trained
    gdown -O exsclaim/figures/models/checkpoints/snapshot20000.ckpt https://drive.google.com/uc?id=1DlYH9TVr0ucgbInHqLd8HFWRkKEWGQCj
    # unzip exsclaim/figures/models/checkpoints/snapshot20000.ckpt -d exsclaim/figures/models/checkpoints
    # rm exsclaim/figures/models/checkpoints/snapshot20000.ckpt.zip
else
    echo "Already downloaded ObjectDetector model"
fi
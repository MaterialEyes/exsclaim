# MaterialEye_Framework_v2

### Description

Train YOLOv3 for detecting subfigure label/scale bar/image/nonimage, adapted from https://github.com/DeNA/PyTorch_YOLOv3 


## Automated Run

Requires pip, best if you run after activating a virtualenv. 
run
```
bash setup.sh
```
(this installs all required packages in requirements.txt and downloads and unzips the .ckpt file)

Then to run the model on the default images in input_images directory, 
```
bash run.sh
```
or if you have no gpu
```
cd detection_compound_figure
bash run_no_gpu.sh
```

## Semi Automated

### How to put weights file?

Download the snapshot930.ckpt.zip file from https://drive.google.com/file/d/1xWxqQGDH_szfCe8eWDBwTcjzCmq7Bnf1/view?usp=sharing, unzip it.

```bash
git clone git@github.com:WeixinGithubJiang/MaterialEye_Framework_v2.git
cd ObjectDetector
mkdir checkpoints
cp /path-to-weights_file/snapshot930.ckpt ./checkpoints
bash test.sh
```

### test.sh
```bash
> --detect_thresh       float number between 0-1, default 0.5
> --ckpt                path of trained weights, default checkpoints/snapshot930.ckpt
> --image               path of input image
> --image_dir           folder path of input images
> --result_dir          folder for saving results
> --image_extend        image extensions, default png
> --gpu			enter 0 if running on machine with no GPU
```

pip install -r requirements.txt
gdown https://drive.google.com/uc?id=1xWxqQGDH_szfCe8eWDBwTcjzCmq7Bnf1
unzip snapshot930.ckpt.zip
mkdir detection_compound_figure/checkpoints
mv snapshot930.ckpt detection_compound_figure/checkpoints
cd detection_compound_figures
## write desired image directory in here
python test.py --detect_thresh 0.5 --ckpt checkpoints/snapshot930.ckpt --image_dir ./sample_input/ --result_dir ./sample_results/ --image_extend png --gpu -1

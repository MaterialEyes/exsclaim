## for multi image detection
python3 run.py --detect_thresh 0.5 --ckpt checkpoints/snapshot930.ckpt --image_dir ./input_images/ --result_dir ./sample_results/ --image_extend jpg --gpu 0

## for single image detection
#python3 run.py --detect_thresh 0.5 --ckpt checkpoints/snapshot930.ckpt --image ./sample_input/complex_1.png --result_dir ./sample_results/ --image_extend png --gpu 0



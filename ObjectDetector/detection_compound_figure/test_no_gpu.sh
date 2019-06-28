## for multi image detection
python3 test.py --detect_thresh 0.5 --ckpt checkpoints/snapshot930.ckpt --image_dir ./sample_input/ --result_dir ./sample_results/ --image_extend png --gpu -1

## for single image detection
#python3 test.py --detect_thresh 0.5 --ckpt checkpoints/snapshot930.ckpt --image ./sample_input/complex_1.png --result_dir ./sample_results/ --image_extend png --gpu -1



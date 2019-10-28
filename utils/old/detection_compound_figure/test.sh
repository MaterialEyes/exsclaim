## for multi images detection
# python test.py --detect_thresh 0.5 --ckpt checkpoints/snapshot930.ckpt --image_dir ./sample_input/ --result_dir ./sample_results/ --image_extend png

## for single image detection
python test.py --detect_thresh 0.5 --ckpt checkpoints/snapshot950.ckpt --image ./sample_input/complex_1.png --result_dir ./sample_results/ --image_extend png


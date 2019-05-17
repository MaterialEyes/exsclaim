# run on a bunch of images while given the folder path
# python test.py --detect_thresh 0.5 --ckpt checkpoints/snapshot200.ckpt --image_dir  ./data/ --result_dir /home/weixin/Documents/MyProjects/PhDResearch/ArgonneProjects/MaterialEyes_Framework/data/subfigures/ --image_extend png --save_bbox --save_image

# test with different weights
# python test.py --detect_thresh 0.5 --ckpt checkpoints_2000/snapshot200.ckpt --image_dir  ./data/ --result_dir /home/weixin/Documents/MyProjects/PhDResearch/ArgonneProjects/MaterialEyes_Framework/data/subfigures/ --image_extend png


# run on single image
python test.py --detect_thresh 0.5 --ckpt checkpoints/snapshot200.ckpt --image  ./data/complex_1.png --result_dir /home/weixin/Documents/MyProjects/PhDResearch/ArgonneProjects/MaterialEyes_Framework/data/subfigures/ --image_extend png --save_bbox --save_image --bbox_expand 0.1
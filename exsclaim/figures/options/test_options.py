import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/yolov3_default.cfg')
    parser.add_argument('--clear_folder', action="store_true")
    parser.add_argument('--ckpt', type=str, help='path to the checkpoint file')
    parser.add_argument('--classifier_ckpt', type=str, help='path to the checkpoint file of classifier')
    parser.add_argument('--image', type=str, help="path to test image")
    parser.add_argument('--image_dir', type=str, help="folder path to test image")
    parser.add_argument('--detect_thresh', type=float,
                        default=0.5, help='confidence threshold')
    parser.add_argument('--result_dir', type=str, help="path to save results", 
                       default="./results")
    parser.add_argument('--gpu_id', type=int, default=0)
    return parser.parse_args()
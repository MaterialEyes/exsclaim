import argparse
import yaml
import cv2
import torch
from torch.autograd import Variable
from models.yolov3 import *
from utils.utils import *
from utils.parse_yolo_weights import parse_yolo_weights
import glob
import os
from dataset.figsepdataset import *
import time
import pytesseract


def parse_command_line_arguments():
    """ reads command line arguments and returns them in dictionary """
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cfg', type=str, default='config/yolov3_default.cfg')
    parser.add_argument('--ckpt', type=str,
                        help='path to the checkpoint file')
    parser.add_argument('--weights_path', type=str,
                        default=None, help='path to weights file')
    parser.add_argument('--image', type=str)
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--image_extend', type=str,default="jpg")
    parser.add_argument('--groundtruth', action="store_true")
    parser.add_argument('--gt_data_dir', type=str)
    parser.add_argument('--gt_anno_dir', type=str)
    parser.add_argument('--max_img', type=int,default=2000)
    parser.add_argument('--background', action='store_true',
                        default=False, 
						help='background(no-display mode. save "./output.png")')
    parser.add_argument('--detect_thresh', type=float,
                        default=None, help='confidence threshold')
    parser.add_argument('--result_dir', type=str)
    args = parser.parse_args()
    return vars(args)
	
def load_model(config_file = "config/yolov3_eval.cfg", detection_threshold = None):
    with open(config_file, 'r') as f:
        cfg = yaml.load(f)

    img_size = cfg['TEST']['IMGSIZE']
    model = YOLOv3(cfg['MODEL'])

    confidence_threshold = cfg['TEST']['CONFTHRE'] 
    nmsthre = cfg['TEST']['NMSTHRE']

    if detection_threshold:
        confidence_threshold = detection_threshold
	
    return model, confidence_threshold, nmsthre, img_size

	
def run_model(model, confidence_threshold, nms_threshold, image_size,
        images_path = "./input_images", extension = "png",
        checkpoint = "./checkpoints/snapshot930.ckpt", gpu = 0):
    """ Runs 

    """
    ## Sets up list of image names
    if os.path.isdir(images_path):
        img_names = glob.glob(os.path.join(images_path, "*." + extension))
        print("num of images in directory:{}".format(len(img_names)))
    else:
        img_names = [images_path]
        print("num of images:{}".format(len(img_names)))
 
    if gpu > 0:
        model.cuda(args[gpu])
        #image = Variable(image.type(torch.cuda.FloatTensor))
        print("loading checkpoint %s" % (checkpoint))
        model.load_state_dict(torch.load(checkpoint)["model_state_dict"])
    else:
        #image = Variable(image.type(torch.FloatTensor))
        print("loading checkpoint %s" % (checkpoint))
        model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model_state_dict"])
          
    model.eval()

   
    #print(img_names)
    ## Runs model on each image and stores the result in a dictionary
    image_to_result = {}    
    for image_name in img_names:    
        #print(image_name)
        image = cv2.imread(image_name)
        image_raw = image.copy()[:, :, ::-1].transpose((2, 0, 1))
        image, info_image = preprocess(image, image_size, jitter=0)  # info = (h, w, nh, nw, dx, dy)
        image = np.transpose(image / 255., (2, 0, 1))
        image = torch.from_numpy(image).float().unsqueeze(0)
        if gpu > 0:
            image = Variable(image.type(torch.cuda.FloatTensor))
        else:
            image = Variable(image.type(torch.FloatTensor))
        
        with torch.no_grad():
            outputs = model(image)
            outputs = postprocess(outputs, 80, confidence_threshold, nms_threshold)
        image_to_result[image_name] = (outputs, info_image)
	
    return image_to_result

def load_and_run_model(input_images = "./input_images", gpu = 0, extension = "png",
        checkpoint = "checkpoints/snapshot930.ckpt", config_file = "config/yolov3_eval.cfg",
        detection_threshold = None):
    """ runs model on images described by input_images path on model """
    model, confidence_threshold, nms_threshold, image_size = load_model(config_file, detection_threshold)
    image_to_result = run_model(model, confidence_threshold, nms_threshold,
            image_size, input_images, extension, checkpoint, gpu)
    return image_to_result

def generate_single_image_dictionary(image_data):
    outputs, info_image = image_data
    coco_class_names, coco_class_ids, coco_class_colors = get_coco_label_names()
     
    classes = {}
    for name, ID, color in zip(coco_class_names, coco_class_ids, coco_class_colors):
        classes[ID-1] = (name, color)
    	
    bboxes = {}

    for x1, y1, x2, y2, conf, cls_conf, cls_pred in outputs[0]:
        y_1, x_1, y_2, x_2 = yolobox2label([y1, x1, y2, x2], info_image)
        location = [{"x" : x_1, "y" : y_1}, {"x" : x_1, "y" : y_2},
                    {"x" : x_2, "y" : y_2}, {"x" : x_2, "y" : y_1}] 
        object_entry = {"geometry" : location, "confidence" : float(cls_conf)}
        box_class = classes[int(cls_pred)][0]
        bbox = bboxes.get(box_class, [])
        bbox.append(object_entry)
        bboxes[box_class] = bbox 
    return bboxes

def generate_json(image_to_result):
    results_json = {}
    for image_name in image_to_result:
        bboxes = generate_single_image_dictionary(image_to_result[image_name])
        results_json[image_name] = {"unassigned" : bboxes}
    return results_json


def run_and_save():
    images_to_outputs = load_and_run_model()
    print(generate_json(images_to_outputs))

def main():
    """
    Visualize the detection result for the given image and the pre-trained model.
    """
    ## get arguments passed in through command line
    args = parse_command_line_arguments()
	
    ## creates the model
        
if __name__ == '__main__':
    start_time = time.time()
    main()
    print("time elapsed = ",time.time()-start_time)

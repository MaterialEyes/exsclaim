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

<<<<<<< HEAD
def parse_arguments():
	""" reads command line arguments and returns them in NameSpace object """
	parser = argparse.ArgumentParser()
=======
def parse_command_line_arguments():
    """ reads command line arguments and returns them in dictionary """
    parser = argparse.ArgumentParser()
>>>>>>> ee30b67... Updated test to output in desired format, but breaks test.sh temporarily
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
<<<<<<< HEAD
	return args
=======
    return vars(args)
>>>>>>> ee30b67... Updated test to output in desired format, but breaks test.sh temporarily
	
def load_model_verbose(args):
    if args["groundtruth"]:
        args["cfg"] = 'config/yolov3_eval.cfg'
        with open(args["cfg"], 'r') as f:
            cfg = yaml.load(f)

        imgsize = cfg['TEST']['IMGSIZE']
        model = YOLOv3(cfg['MODEL'])

        confthre = cfg['TEST']['CONFTHRE'] 
        nmsthre = cfg['TEST']['NMSTHRE']
        batch_size = cfg['TRAIN']['BATCHSIZE']
        
        if args["detect_thresh"]:
            confthre = args["detect_thresh"]
            
        dataset = FigSepDataset(model_type=cfg['MODEL']['TYPE'],
                              data_dir=args["gt_data_dir"],
                              anno_dir =args["gt_anno_dir"],
                              img_size=cfg['TRAIN']['IMGSIZE'],
                              augmentation=cfg['AUGMENTATION'],
                              debug=False)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        dataiterator = iter(dataloader)
        
        cuda = torch.cuda.is_available()
        
        if cuda:
            print("using cuda") 
            model = model.cuda()
        
        dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        
        total_loss = 0
        for i in range(args["max_img"]):
            try:
                imgs, targets, _, _ = next(dataiterator)
            except StopIteration:
                print("{} images has been assessed".format(i+1))
                break
            imgs = Variable(imgs.type(dtype))
            targets = Variable(targets.type(dtype), requires_grad=False)
            loss = model(imgs, targets)
            total_loss += loss.data.cpu().numpy()
        print("loss is {}".format(total_loss/(i+1)))
      
    with open(args["cfg"], 'r') as f:
        cfg = yaml.load(f)

    imgsize = cfg['TEST']['IMGSIZE']
    model = YOLOv3(cfg['MODEL'])

    confthre = cfg['TEST']['CONFTHRE'] 
    nmsthre = cfg['TEST']['NMSTHRE']

    if args["detect_thresh"]:
        confthre = args["detect_thresh"]
	
    return model, confthre, nmsthre, imgsize

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

def generate_single_image(image_name, outputs, info_image, output_directory = "./formatted_outputs"):
    coco_class_names, coco_class_ids, coco_class_colors = get_coco_label_names()

    bboxes = list()
    classes = list()
    colors = list()
    confidences = []
		
    subfigure_labels = {}
    scalebar_labels = {}

    for x1, y1, x2, y2, conf, cls_conf, cls_pred in outputs[0]:

        cls_id = coco_class_ids[int(cls_pred)]
        #print(int(x1), int(y1), int(x2), int(y2), float(conf), int(cls_pred))
        #print('\t+ Label: %s, Conf: %.5f' %
        #      (coco_class_names[cls_id], cls_conf.item()))
        box = yolobox2label([y1, x1, y2, x2], info_image)
        bboxes.append(box)
        classes.append(cls_id)
        colors.append(coco_class_colors[int(cls_pred)])
        confidences.append("%.3f"%(cls_conf.item()))

        
    from PIL import Image
    test_img = Image.open(image_name).convert("RGB")
    sub_figures = []
    labels = []
    scalebars = []
		
    for i in range(len(bboxes)):
        box = bboxes[i]
        h,w = test_img.size
        y0 = int(max(box[0].data.cpu().numpy(),0))
        x0 = int(max(box[1].data.cpu().numpy(),0))
        y1 = int(min(box[2].data.cpu().numpy(),w-1))
        x1 = int(min(box[3].data.cpu().numpy(),h-1))
        #print(x0,y0,x1,y1,type(x0))
        sub_figure = test_img.crop((x0,y0,x1,y1))
        if classes[i] == 1:
            sub_figures.append(sub_figure)
        elif classes[i] == 3:
            labels.append(sub_figure)
        elif classes[i] == 4:
            scalebars.append(sub_figure)
	
    if not os.path.exists(os.path.join(output_directory, "image")):
        os.makedirs(os.path.join(output_directory, "sub"))
        os.makedirs(os.path.join(output_directory, "scale"))
        os.makedirs(os.path.join(output_directory, "image"))
    
    for i in range(len(sub_figures)):
        sub_figures[i].save(os.path.join(output_directory, "image/image_{}_".format(i+1) + image_name.split("/")[-1].split(".")[0]+".png"))
        #sub_figures[i].save(os.path.join(os.path.join(args.result_dir,sample_img.split("/")[-1].split(".")[0]),"{}_output.png".format(i+1)))
    for i in range(len(labels)):
        new_image_name = "sub/sub_{}_".format(i+1)+image_name.split("/")[-1].split(".")[0]+".png"
        labels[i].save(os.path.join(output_directory, new_image_name))
        text = pytesseract.image_to_string(labels[i], config = "--psm 10 --oem 3")
        subfigure_labels[new_image_name] = text
	
    for i in range(len(scalebars)):
        new_image_name = "scale/scale_{}_".format(i+1)+image_name.split("/")[-1].split(".")[0]+".png"
        scalebars[i].save(os.path.join(output_directory, new_image_name))			
        text = pytesseract.image_to_string(labels[i], config = "--psm 10 --oem 3")
        scalebar_labels[new_image_name] = text
        
    from PIL import Image, ImageDraw, ImageFont
    test_img = Image.open(image_name).convert("RGB")
    draw = ImageDraw.Draw(test_img)
    font_type = "DejaVuSansMono.ttf"
    font = ImageFont.truetype(font_type, 10)

#         print(classes)
    for i in range(len(bboxes)):
        box = bboxes[i]
        h,w = test_img.size
        y0 = max(box[0].data.cpu().numpy(),0)
        x0 = max(box[1].data.cpu().numpy(),0)
        y1 = min(box[2].data.cpu().numpy(),w-1)
        x1 = min(box[3].data.cpu().numpy(),h-1)
        if classes[i] ==1:
            rect_color = "blue"
        elif classes[i] ==2:
            rect_color = "green"
        elif classes[i] ==3:
            rect_color = "red"
        else:
            rect_color = "white"
        draw.rectangle([x0,y0,x1,y1],outline=rect_color)#tuple(colors[i]))
        label = coco_class_names[classes[i]]
        conf = confidences[i]
        text = label+":"+conf
        text_width, text_height = font.getsize(text)
        draw.text((x0,y0),text,fill=rect_color,font=font)
    del draw
    #if not os.path.exists(os.path.join(output_directory, image_name.split("/")[-1].split(".")[0])):
    #    os.makedirs(os.path.join(output_directory, image_name.split("/")[-1].split(".")[0]))
    test_img.save(os.path.join(output_directory, image_name.split("/")[-1]))
    #test_img.save(os.path.join(os.path.join(output_directory,image_name.split("/")[-1].split(".")[0]) + ".png"))
    
    return scalebar_labels, subfigure_labels

def generate_output_files(images_to_outputs):
    scalebar_labels = {}
    subfigure_labels = {}
    
    for image_name in images_to_outputs:
        scalebars, subfigures = generate_single_image(image_name, images_to_outputs[image_name][0], images_to_outputs[image_name][1])
        scalebar_labels.update(scalebars)
        subfigure_labels.update(subfigures)

    ## save labels
    with open("formatted_outputs/sub/image_labels.yaml", "w") as outfile:
        yaml.dump(subfigure_labels, outfile)
    with open("formatted_outputs/scale/image_labels.yaml", "w") as outfile:
        yaml.dump(scalebar_labels, outfile)
	
def run_and_save():
    images_to_outputs = load_and_run_model()
    generate_output_files(images_to_outputs)


def run_model_verbose(args, model, confthre, nmsthre):
    if args["image_dir"]:
        img_names = glob.glob(os.path.join(args.image_dir,"*."+args["image_extend"]))
        print("num of images:{}".format(len(img_names)))
    else:
        img_names = [args["image"]]
        print("num of images:{}".format(len(img_names)))
                
    if not os.path.exists(args["result_dir"]):
        os.makedirs(args["result_dir"])
        
    for sample_img in img_names:
        args["image"] = sample_img
    
        img = cv2.imread(args["image"])
        img_raw = img.copy()[:, :, ::-1].transpose((2, 0, 1))
        img, info_img = preprocess(img, imgsize, jitter=0)  # info = (h, w, nh, nw, dx, dy)
        img = np.transpose(img / 255., (2, 0, 1))
        img = torch.from_numpy(img).float().unsqueeze(0)

<<<<<<< HEAD
<<<<<<< HEAD
def main():
    """
    Visualize the detection result for the given image and the pre-trained model.
    """
    ## get arguments passed in through command line
	args = parse_arguments()
=======
        if args.gpu >= 0:
            model.cuda(args.gpu)
=======
        if args["gpu"] >= 0:
            model.cuda(args["gpu"])
>>>>>>> ee30b67... Updated test to output in desired format, but breaks test.sh temporarily
            img = Variable(img.type(torch.cuda.FloatTensor))
        else:
            img = Variable(img.type(torch.FloatTensor))
>>>>>>> 1b906c0... test.py now saves scale bars and labels too in desired format

        if args["weights_path"]:
            print("loading yolo weights %s" % (args.weights_path))
            parse_yolo_weights(model, args["weights_path"])
        elif args["gpu"] < 0:
            print("loading checkpoint %s" % (args["ckpt"]))
            model.load_state_dict(torch.load(args["ckpt"], map_location="cpu")["model_state_dict"])
        else:
            print("loading checkpoint %s" % (args["ckpt"]))
            model.load_state_dict(torch.load(args["ckpt"])["model_state_dict"])

        model.eval()

        with torch.no_grad():
            outputs = model(img)
            print("outputs:",outputs.size())
            outputs = postprocess(outputs, 80, confthre, nmsthre)

        if outputs[0] is None:
            print("No Objects Deteted!!")
            continue
	
    return outputs
	
def generate_images_old(yolo_output):
    coco_class_names, coco_class_ids, coco_class_colors = get_coco_label_names()

    bboxes = list()
    classes = list()
    colors = list()
    confidences = []
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in outputs[0]:

        cls_id = coco_class_ids[int(cls_pred)]
        print(int(x1), int(y1), int(x2), int(y2), float(conf), int(cls_pred))
        print('\t+ Label: %s, Conf: %.5f' %
             (coco_class_names[cls_id], cls_conf.item()))
        box = yolobox2label([y1, x1, y2, x2], info_img)
        bboxes.append(box)
        classes.append(cls_id)
        colors.append(coco_class_colors[int(cls_pred)])
        confidences.append("%.3f"%(cls_conf.item()))

        
        from PIL import Image
        test_img = Image.open(args.image).convert("RGB")
        sub_figures = []
        for i in range(len(bboxes)):
            box = bboxes[i]
            h,w = test_img.size
            y0 = int(max(box[0].data.cpu().numpy(),0))
            x0 = int(max(box[1].data.cpu().numpy(),0))
            y1 = int(min(box[2].data.cpu().numpy(),w-1))
            x1 = int(min(box[3].data.cpu().numpy(),h-1))
            print(x0,y0,x1,y1,type(x0))
            sub_figure = test_img.crop((x0,y0,x1,y1))
            if classes[i] == 1:
                sub_figures.append(sub_figure)
        if not os.path.exists(os.path.join(args.result_dir,sample_img.split("/")[-1].split(".")[0])):
            os.makedirs(os.path.join(args.result_dir,sample_img.split("/")[-1].split(".")[0]))
        for i in range(len(sub_figures)):
            sub_figures[i].save(os.path.join(args.result_dir,sample_img.split("/")[-1].split(".")[0]+"----{}_output.png".format(i+1)))
            sub_figures[i].save(os.path.join(os.path.join(args.result_dir,sample_img.split("/")[-1].split(".")[0]),"{}_output.png".format(i+1)))
        
        
        from PIL import Image, ImageDraw, ImageFont
        test_img = Image.open(args.image).convert("RGB")
        draw = ImageDraw.Draw(test_img)
        font_type = "DejaVuSansMono.ttf"
        font = ImageFont.truetype(font_type, 10)

#        print(classes)
        for i in range(len(bboxes)):
            box = bboxes[i]
            h,w = test_img.size
            y0 = max(box[0].data.cpu().numpy(),0)
            x0 = max(box[1].data.cpu().numpy(),0)
            y1 = min(box[2].data.cpu().numpy(),w-1)
            x1 = min(box[3].data.cpu().numpy(),h-1)
            if classes[i] ==1:
                rect_color = "blue"
            elif classes[i] ==2:
                rect_color = "green"
            elif classes[i] ==3:
                rect_color = "red"
            else:
                rect_color = "white"
            draw.rectangle([x0,y0,x1,y1],outline=rect_color)#tuple(colors[i]))
            label = coco_class_names[classes[i]]
            conf = confidences[i]
            text = label+":"+conf
            text_width, text_height = font.getsize(text)
            draw.text((x0,y0),text,fill=rect_color,font=font)
        del draw
        if not os.path.exists(os.path.join(args.result_dir,sample_img.split("/")[-1].split(".")[0])):
            os.makedirs(os.path.join(args.result_dir,sample_img.split("/")[-1].split(".")[0]))
        test_img.save(os.path.join(args.result_dir,sample_img.split("/")[-1]))
        test_img.save(os.path.join(os.path.join(args.result_dir,sample_img.split("/")[-1].split(".")[0]),"output.png"))

def main():
    """
    Visualize the detection result for the given image and the pre-trained model.
    """
    ## get arguments passed in through command line
    args = parse_arguments()
	
    ## creates the model
    model, confthre, nmsthre, imgsize = load_model(args)
    
    

    if args.image_dir:
        img_names = glob.glob(os.path.join(args.image_dir,"*."+args.image_extend))
        print("num of images:{}".format(len(img_names)))
    else:
        img_names = [args.image]
        print("num of images:{}".format(len(img_names)))  
    
    
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
        
    for sample_img in img_names:
        args.image = sample_img
    
        img = cv2.imread(args.image)
        img_raw = img.copy()[:, :, ::-1].transpose((2, 0, 1))
        img, info_img = preprocess(img, imgsize, jitter=0)  # info = (h, w, nh, nw, dx, dy)
        img = np.transpose(img / 255., (2, 0, 1))
        img = torch.from_numpy(img).float().unsqueeze(0)

        if args.gpu >= 0:
            model.cuda(args.gpu)
            img = Variable(img.type(torch.cuda.FloatTensor))
        else:
            img = Variable(img.type(torch.FloatTensor))

        if args.weights_path:
            print("loading yolo weights %s" % (args.weights_path))
            parse_yolo_weights(model, args.weights_path)
        elif args.gpu < 0:
            print("loading checkpoint %s" % (args.ckpt))
            model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model_state_dict"])
        else:
            print("loading checkpoint %s" % (args.ckpt))
            model.load_state_dict(torch.load(args.ckpt)["model_state_dict"])

        model.eval()

        with torch.no_grad():
            outputs = model(img)
            print("outputs:",outputs.size())
            outputs = postprocess(outputs, 80, confthre, nmsthre)

        if outputs[0] is None:
            print("No Objects Deteted!!")
            continue

        coco_class_names, coco_class_ids, coco_class_colors = get_coco_label_names()

        bboxes = list()
        classes = list()
        colors = list()
        confidences = []
		
        subfigure_labels = {}
        scalebar_labels = {}

        for x1, y1, x2, y2, conf, cls_conf, cls_pred in outputs[0]:

            cls_id = coco_class_ids[int(cls_pred)]
            print(int(x1), int(y1), int(x2), int(y2), float(conf), int(cls_pred))
            print('\t+ Label: %s, Conf: %.5f' %
                  (coco_class_names[cls_id], cls_conf.item()))
            box = yolobox2label([y1, x1, y2, x2], info_img)
            bboxes.append(box)
            classes.append(cls_id)
            colors.append(coco_class_colors[int(cls_pred)])
            confidences.append("%.3f"%(cls_conf.item()))

        
<<<<<<< HEAD
         from PIL import Image
         test_img = Image.open(args.image).convert("RGB")
         sub_figures = []
         for i in range(len(bboxes)):
             box = bboxes[i]
             h,w = test_img.size
             y0 = int(max(box[0].data.cpu().numpy(),0))
             x0 = int(max(box[1].data.cpu().numpy(),0))
             y1 = int(min(box[2].data.cpu().numpy(),w-1))
             x1 = int(min(box[3].data.cpu().numpy(),h-1))
             print(x0,y0,x1,y1,type(x0))
             sub_figure = test_img.crop((x0,y0,x1,y1))
             if classes[i] == 1:
                 sub_figures.append(sub_figure)
         if not os.path.exists(os.path.join(args.result_dir,sample_img.split("/")[-1].split(".")[0])):
             os.makedirs(os.path.join(args.result_dir,sample_img.split("/")[-1].split(".")[0]))
         for i in range(len(sub_figures)):
             sub_figures[i].save(os.path.join(args.result_dir,sample_img.split("/")[-1].split(".")[0]+"----{}_output.png".format(i+1)))
             sub_figures[i].save(os.path.join(os.path.join(args.result_dir,sample_img.split("/")[-1].split(".")[0]),"{}_output.png".format(i+1)))
        
=======
        from PIL import Image
        test_img = Image.open(args.image).convert("RGB")
        sub_figures = []
        labels = []
        scalebars = []
		
        for i in range(len(bboxes)):
            box = bboxes[i]
            h,w = test_img.size
            y0 = int(max(box[0].data.cpu().numpy(),0))
            x0 = int(max(box[1].data.cpu().numpy(),0))
            y1 = int(min(box[2].data.cpu().numpy(),w-1))
            x1 = int(min(box[3].data.cpu().numpy(),h-1))
            print(x0,y0,x1,y1,type(x0))
            sub_figure = test_img.crop((x0,y0,x1,y1))
            if classes[i] == 1:
                sub_figures.append(sub_figure)
            elif classes[i] == 3:
                labels.append(sub_figure)
            elif classes[i] == 4:
                scalebars.append(sub_figure)
			
        if not os.path.exists(os.path.join(args.result_dir,sample_img.split("/")[-1].split(".")[0])):
            os.makedirs(os.path.join(args.result_dir,sample_img.split("/")[-1].split(".")[0]))
        for i in range(len(sub_figures)):
            sub_figures[i].save(os.path.join(args.result_dir,"image_{}_".format(i+1)+sample_img.split("/")[-1].split(".")[0]+".png"))
            #sub_figures[i].save(os.path.join(os.path.join(args.result_dir,sample_img.split("/")[-1].split(".")[0]),"{}_output.png".format(i+1)))
        for i in range(len(labels)):
            new_image_name = "scale_{}_".format(i+1)+sample_img.split("/")[-1].split(".")[0]+".png"
            labels[i].save(os.path.join(args.result_dir,new_image_name))
            text = pytesseract.image_to_string(labels[i], config = "--psm 10 --oem 3")
            subfigure_labels[new_image_name] = text
			
        for i in range(len(scalebars)):
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
            scalebars[i].save(os.path.join(args.result_dir,"scale_{}_".format(i+1)+sample_img.split("/")[-1].split(".")[0]))			
>>>>>>> 1b906c0... test.py now saves scale bars and labels too in desired format
=======
            scalebars[i].save(os.path.join(args.result_dir,"scale_{}_".format(i+1)+sample_img.split("/")[-1].split(".")[0]+".png"))			
>>>>>>> cb96c78... fixed indentation, file extensions
=======
			new_image_name = "scale_{}_".format(i+1)+sample_img.split("/")[-1].split(".")[0]+".png"
=======
            new_image_name = "scale_{}_".format(i+1)+sample_img.split("/")[-1].split(".")[0]+".png"
>>>>>>> 76872d1... Added ability to run on directories
            scalebars[i].save(os.path.join(args.result_dir,new_image_name))			
            text = pytesseract.image_to_string(labels[i], config = "--psm 10 --oem 3")
            scalebar_labels[new_image_name] = text

>>>>>>> 4210dd1... Added text output
        
        from PIL import Image, ImageDraw, ImageFont
        test_img = Image.open(args.image).convert("RGB")
        draw = ImageDraw.Draw(test_img)
        font_type = "DejaVuSansMono.ttf"
        font = ImageFont.truetype(font_type, 10)

#         print(classes)
        for i in range(len(bboxes)):
            box = bboxes[i]
            h,w = test_img.size
            y0 = max(box[0].data.cpu().numpy(),0)
            x0 = max(box[1].data.cpu().numpy(),0)
            y1 = min(box[2].data.cpu().numpy(),w-1)
            x1 = min(box[3].data.cpu().numpy(),h-1)
            if classes[i] ==1:
                rect_color = "blue"
            elif classes[i] ==2:
                rect_color = "green"
            elif classes[i] ==3:
                rect_color = "red"
            else:
                rect_color = "white"
            draw.rectangle([x0,y0,x1,y1],outline=rect_color)#tuple(colors[i]))
            label = coco_class_names[classes[i]]
            conf = confidences[i]
            text = label+":"+conf
            text_width, text_height = font.getsize(text)
            draw.text((x0,y0),text,fill=rect_color,font=font)
        del draw
        if not os.path.exists(os.path.join(args.result_dir,sample_img.split("/")[-1].split(".")[0])):
            os.makedirs(os.path.join(args.result_dir,sample_img.split("/")[-1].split(".")[0]))
        test_img.save(os.path.join(args.result_dir,sample_img.split("/")[-1]))
        test_img.save(os.path.join(os.path.join(args.result_dir,sample_img.split("/")[-1].split(".")[0]),"output.png"))
    
	
	## save labels
    with open("subfigure_labels.yaml", "w") as outfile:
        yaml.dump(subfigure_labels, outfile)
    with open("scalebar_labels.yaml", "w") as outfile:
        yaml.dump(scalebar_labels, outfile)	
    
#     print("bbbox=",bboxes)
#     print("classes=",classes)
#     print("colors=",colors)
#     print(tuple(colors[0]))

#     if args.background:
#         import matplotlib
#         matplotlib.use('Agg')

#     from utils.vis_bbox import vis_bbox
#     import matplotlib.pyplot as plt

#     vis_bbox(
#         img_raw, bboxes, label=classes, label_names=coco_class_names,
#         instance_colors=colors, linewidth=2)
#     plt.show()

#     if args.background:
#         plt.savefig('output.png')


if __name__ == '__main__':
    start_time = time.time()
    #main()
    run_and_save()
    print("time elapsed = ",time.time()-start_time)

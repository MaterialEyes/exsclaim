import os
import cv2
import json
import yaml
import glob
import torch
import shutil

import numpy as np
import torch.nn.functional as F

from skimage import io
from scipy.special import softmax
from torch.autograd import Variable
from PIL import Image, ImageDraw, ImageFont

from .figures.models.yolov3 import *
from .figures.utils.utils import *

def load_subfigure_model(model_path=str) -> "figure_separator_model":
    """
    Opens and extracts model snapshot from configuration file + checkpoints

    Args:
        model_path: A path to the model files

    Returns:
        figure_separator_model: A tuple (model, confidence_threshold, nms_threshold, img_size, gpu)
    """

    # Paths to config/checkpoint files
    # objd_ckpt = model_path + "checkpoints/snapshot6500.ckpt" # object detector
    objd_ckpt = model_path + "checkpoints/snapshot13400.ckpt"
    clsf_ckpt = model_path + "checkpoints/snapshot260.ckpt"  # classifier
    cnfg_file = model_path + "config/yolov3_default_subfig.cfg"
        
    # Open the config file
    with open(cnfg_file, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    # print("Successfully loaded config file: \n", cfg)

    # Assign values to important varables
    image_size            = cfg['TEST']['IMGSIZE']
    nms_threshold         = cfg['TEST']['NMSTHRE']
    confidence_threshold  = 0.0001
    gpu_id                = 1

    # load object_detect model
    model = YOLOv3(cfg['MODEL'])

    # load classifier model
    classifier_model = None
    for m in model.module_list:
        if hasattr(m,"classifier_model"):
            classifier_model = m.classifier_model
            break
    assert classifier_model != None

    cuda = torch.cuda.is_available() and (gpu_id >= 0)
    if objd_ckpt:
        if cuda:
            model.load_state_dict(torch.load(objd_ckpt)["model_state_dict"])
            classifier_model.load_state_dict(torch.load(clsf_ckpt)["model_state_dict"])
        else:
            model.load_state_dict(torch.load(objd_ckpt,map_location='cpu')["model_state_dict"])
            classifier_model.load_state_dict(torch.load(clsf_ckpt, map_location='cpu')["model_state_dict"])
    dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    if cuda:
        print("using cuda: ", args.gpu_id) 
        torch.cuda.set_device(device=args.gpu_id)
        model = model.cuda()
        classifier_model = classifier_model.cuda()

    return (model, classifier_model, dtype, confidence_threshold, nms_threshold, image_size, gpu_id)

def load_masterimg_model(model_path=str) -> "figure_separator_model":
    """
    Opens and extracts model snapshot from configuration file + checkpoints

    Args:
        model_path: A path to the model files

    Returns:
        figure_separator_model: A tuple (model, confidence_threshold, nms_threshold, img_size, gpu)
    """
    # Paths to config/checkpoint files
    objd_ckpt = model_path + "checkpoints/snapshot12000.ckpt" # object detector
    cnfg_file = model_path + "config/yolov3_default_master.cfg"
        
    # Open the config file
    with open(cnfg_file, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    # print("Successfully loaded config file: \n", cfg)

    # Assign values to important varables
    image_size = cfg['TEST']['IMGSIZE']
    gpu_id     = 1

    # load object_detect model
    model = YOLOv3img(cfg['MODEL'])

    cuda = torch.cuda.is_available() and (gpu_id >= 0)
    if objd_ckpt:
        if cuda:
            model.load_state_dict(torch.load(objd_ckpt)["model_state_dict"])
        else:
            model.load_state_dict(torch.load(objd_ckpt,map_location='cpu')["model_state_dict"])

    dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    if cuda:
        print("using cuda: ", args.gpu_id) 
        torch.cuda.set_device(device=args.gpu_id)
        model = model.cuda()
    
    return (model, image_size)


def get_figure_paths(search_query: dict) -> list:
    """
    Get a list of paths to figures extracted using the search_query

    Args:
        search_query: A query json
    Returns:
        A list of figure paths
    """
    extensions = ['.png','jpg','.gif']
    paths = []
    for ext in extensions:
        paths+=glob.glob(search_query['results_dir']+'figures/*'+ext)
    return paths


def extract_image_objects(subfigure_label_model=tuple, master_image_model=tuple, figure_path=str, save_path="") -> "figure_dict":
    """
    Find individual image objects within a figure and classify based on functionality

    Args:
        figure_separator_model: A tuple (model, confidence_threshold, nms_threshold, img_size, gpu)
        figure_path: A path to the figure to separate

    Returns:
        figure_dict: A dictionary with classified image_objects extracted from figure
    """
    # Unpack models and eval
    model, classifier_model, dtype, confidence_threshold, nms_threshold, image_size, _ = subfigure_label_model
    mi_model, _ = master_image_model

    model.eval()
    classifier_model.eval()
    mi_model.eval()

    os.makedirs(save_path+"/extractions", exist_ok=True)

    # label_names = ["background","microscopy","parent","graph","illustration","diffraction","None",
    #                "OtherMaster","OtherSubfigure","a","b","c","d","e","f"]

    label_names = ["background","microscopy","parent","graph","illustration","diffraction","basic_photo",
                   "unclear","OtherSubfigure","a","b","c","d","e","f"]

    img = io.imread(figure_path)
    if len(np.shape(img)) == 2:
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img,cv2.COLOR_RGBA2RGB)
    
    img, info_img = preprocess(img, image_size, jitter=0)
    
    img = np.transpose(img / 255., (2, 0, 1))
    img = torch.from_numpy(img).float().unsqueeze(0)
    img = Variable(img.type(dtype))

    # prediction
    img_raw = Image.open(figure_path).convert("RGB")
    width, height = img_raw.size

    with torch.no_grad():
        outputs = model(img)
        outputs = postprocess(outputs, dtype=dtype, 
                    conf_thre=confidence_threshold, nms_thre=nms_threshold)
    if outputs[0] is None:
        print("No Objects Deteted!!")

    bboxes = list()
    confidences = []
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in outputs[0]:
        box = yolobox2label([y1.data.cpu().numpy(), x1.data.cpu().numpy(), y2.data.cpu().numpy(), x2.data.cpu().numpy()], info_img)
        box[0] = int(min(max(box[0],0),width-1))
        box[1] = int(min(max(box[1],0),height-1))
        box[2] = int(min(max(box[2],0),width))
        box[3] = int(min(max(box[3],0),height))
        if box[2]-box[0] > 5 and box[3]-box[1] > 5:
            bboxes.append(box)
            confidences.append("%.3f"%(cls_conf.item()))

    # save results
    sample_image_name = figure_path.split("/")[-1].split(".")[0]
    # img_raw.save(os.path.join(inpt_dir,sample_image_name+".png"))
    width, height = img_raw.size
    binary_img = np.zeros((height,width,1))
    pair_dtype = [("x1",float),("y1",float),("x2",float),("y2",float),("cat",int)]
    subfigure_pair_info_bboxes = []
    # with open(os.path.join(outpt_dir,sample_image_name+".txt"),"a+") as results_file:
        # results_file.write("z 1 0 0 1 1\n")
    detected_labels = []
    detected_bboxes = []
    for i in range(len(bboxes)):
        img_patch = img_raw.crop(tuple(bboxes[i]))
        img_patch = np.array(img_patch)[:,:,::-1]
        img_patch, _ = preprocess(img_patch, 28, jitter=0)
        img_patch = np.transpose(img_patch / 255., (2, 0, 1))
        img_patch = torch.from_numpy(img_patch).type(dtype).unsqueeze(0)
        label_prediction = classifier_model(img_patch)
        label_conf = np.amax(F.softmax(label_prediction, dim=1).data.cpu().numpy())
        label_value = chr(label_prediction.argmax(dim=1).data.cpu().numpy()[0]+ord("a"))
        if label_value == "z":
            pass
        else:
            x1,y1,x2,y2 = int(bboxes[i][0]),int(bboxes[i][1]),int(bboxes[i][2]),int(bboxes[i][3])
            conf = float(confidences[i])*label_conf
            if label_value in detected_labels:
                label_index = detected_labels.index(label_value)
                if conf > detected_bboxes[label_index][0]:
                    detected_bboxes[label_index] = [conf,x1,y1,x2,y2]
            else:
                detected_labels.append(label_value)
                detected_bboxes.append([conf,x1,y1,x2,y2])
    
    # post processing
    assert len(detected_labels) == len(detected_bboxes)
    for i in range(len(detected_labels)):
        label_value = detected_labels[i]
        if (ord(label_value) - ord("a")) < (len(detected_labels)+2):
            conf,x1,y1,x2,y2 = detected_bboxes[i]
            if (x2-x1) < 50 and (y2-y1)< 50:
                binary_img[y1:y2,x1:x2] = 255
                text = "%s %f %d %d %d %d\n"%(label_value, conf, x1, y1, x2, y2)
                subfigure_pair_info_bboxes.append(tuple([x1/width,y1/height,x2/width,y2/height, ord(label_value)-ord("a")]))
                # with open(os.path.join(outpt_dir,sample_image_name+".txt"),"a+") as results_file:
                #     results_file.write(text)
        else:
            pass

    # save concatenate image and pair info
    subfigure_pair_info_bboxes = np.array(subfigure_pair_info_bboxes, dtype=pair_dtype)
    concate_img = np.concatenate((np.array(img_raw),binary_img),axis=2)
    # np.save(os.path.join(concat_dir,sample_image_name+".npy"),concate_img)
    
    pair_info = []
    pair_info.append([width, height])
    for bbox in subfigure_pair_info_bboxes:
        pair_info.append([bbox,bbox])
    # np.save(os.path.join(pair_info_dir,sample_image_name+".npy"),pair_info)
    
    # documentation
    json_info = {}
    json_info["figure_separator_results"] = []
        
    img = concate_img[...,:3].copy()
    mask = concate_img[...,3:].copy()

    img, info_img = preprocess(img, image_size, jitter=0)
    img = np.transpose(img / 255., (2, 0, 1))
    mask = preprocess_mask(mask, image_size, info_img)
    mask = np.transpose(mask / 255., (2, 0, 1))
    new_concate_img = np.concatenate((img,mask),axis=0)
    
    img = torch.from_numpy(new_concate_img).float().unsqueeze(0)
    img = Variable(img.type(dtype))

    # documentation
    current_info = {}
    current_info["figure_name"] = figure_path.split("/")[-1]
    current_info["master_images"] = []
    current_info["unassigned"] = []
    
    width, height = pair_info[0]
    subfigure_labels = []
    for i in range(1, len(pair_info)):
        subfigure, master = pair_info[i]
        
        x1,y1,x2,y2,c = subfigure
        x1, x2 = float(x1*width), float(x2*width)
        y1, y2 = float(y1*height), float(y2*height)
        subfigure_labels.append([])
        subfigure_labels[-1].append(c)
        subfigure_labels[-1].extend([x1,y1,x2-x1,y2-y1])
    
    subfigure_labels_copy = subfigure_labels.copy()
    
    subfigure_padded_labels = np.zeros((80, 5))
    if len(subfigure_labels) > 0:
        subfigure_labels = np.stack(subfigure_labels)
        subfigure_labels = label2yolobox(subfigure_labels, info_img, image_size, lrflip=False)
        subfigure_padded_labels[range(len(subfigure_labels))[:80]
                      ] = subfigure_labels[:80]
    subfigure_padded_labels = (torch.from_numpy(subfigure_padded_labels)).float().unsqueeze(0)
    subfigure_padded_labels = Variable(subfigure_padded_labels.type(dtype))
    
    padded_label_list = [None, subfigure_padded_labels]
    assert subfigure_padded_labels.size()[0] == 1

    # prediction
    img_raw = Image.fromarray(np.uint8(concate_img[...,:3].copy()[...,::-1]))
    width, height = img_raw.size
    with torch.no_grad():
        outputs = mi_model(img, padded_label_list)

    # select the 13x13 grid as feature map
    feature_size = [13,26,52]
    feature_index = 0
    preds = outputs[feature_index]
    preds = preds[0].data.cpu().numpy()
    
    result_image = Image.new(mode="RGB",size=(200,len(pair_info)*100-50))
    draw = ImageDraw.Draw(result_image)
    font = ImageFont.load_default()

    # headline text
    text = "labels"
    draw.text((10,10),text,fill="white",font=font)
    text = "master image"
    draw.text((110,10),text,fill="white",font=font)
    
    label_img = img_raw.copy()
    img_draw = ImageDraw.Draw(label_img)
    # font_draw = ImageFont.truetype("FreeSerifBoldItalic.ttf", 15)

    for subfigure_id in range(0, len(pair_info)-1):
        sub_cat,x,y,w,h = (subfigure_padded_labels[0,subfigure_id]* feature_size[feature_index] ).to(torch.int16).data.cpu().numpy()
        best_anchor = np.argmax(preds[:,y,x,4])
        tx,ty = np.array(preds[best_anchor,y,x,:2]/32,np.int32)
        best_anchor = np.argmax(preds[:,ty,tx,4])
        x,y,w,h = preds[best_anchor,ty,tx,:4]
        cls = np.argmax(preds[best_anchor,int(ty),int(tx),5:])
        master_cls_conf = max(softmax(preds[best_anchor,int(ty),int(tx),5:]))
        master_obj_conf = preds[best_anchor,ty,tx,4]
        # print("CLASS: {0} ({1})".format(label_names[cls],master_cls_conf))

        x1 = (x-w/2)
        x2 = (x+w/2)
        y1 = (y-h/2)
        y2 = (y+h/2)
 
        x1,y1,x2,y2 = yolobox2label([y1,x1,y2,x2], info_img)
        # visualization
        patch = img_raw.crop((int(x1),int(y1),int(x2),int(y2)))
        
        master_label = label_names[cls]
        subfigure_label = chr(int(sub_cat/feature_size[feature_index])+ord("a"))
        
        text = "%s %f %d %d %d %d\n"%(master_label, master_cls_conf*master_obj_conf, int(x1), int(y1), int(x2), int(y2))
        with open(os.path.join(save_path+"/extractions/",sample_image_name+".txt"),"a+") as results_file:
            results_file.write(text)
        
        img_draw.line([(x1,y1),(x1,y2),(x2,y2),(x2,y1),(x1,y1)], fill=(255,0,0), width=3)
        img_draw.rectangle((x2-100,y2-30,x2,y2),fill=(0,255,0))
        img_draw.text((x2-100+2,y2-30+2),"{}, {}".format(master_label,subfigure_label),fill=(255,0,0))
        
        text = "%s\n%s"%(master_label,subfigure_label)
        draw.text((10,60+100*subfigure_id),text,fill="white",font=font)
        
        pw,ph = patch.size
        if pw>ph:
            ph = ph/pw*80
            pw = 80
        else:
            pw = pw/ph*80
            ph = 80
            
        patch = patch.resize((int(pw),int(ph)))
        result_image.paste(patch,box=(110,60+100*subfigure_id))

        # documentation
        master_image_info = {}
        master_image_info["classification"] = master_label
        master_image_info["confidence"] = float("{0:.4f}".format(master_cls_conf))
        master_image_info["geometry"] = []
        for x in [int(x1), int(x2)]:
            for y in [int(y1), int(y2)]:
                geometry = {}
                geometry["x"] = x
                geometry["y"] = y
                master_image_info["geometry"].append(geometry)
        subfigure_label_info = {}
        subfigure_label_info["text"] = subfigure_label
        subfigure_label_info["geometry"] = []
        _,x1,y1,x2,y2 = subfigure_labels_copy[subfigure_id]
        x2 += x1
        y2 += y1
        for x in [int(x1), int(x2)]:
            for y in [int(y1), int(y2)]:
                geometry = {}
                geometry["x"] = x
                geometry["y"] = y
                subfigure_label_info["geometry"].append(geometry)
        master_image_info["subfigure_label"] = subfigure_label_info
        current_info["master_images"].append(master_image_info)
        
    json_info["figure_separator_results"]=[current_info]
        
    del draw
    result_image.save(os.path.join(save_path+"/extractions/"+sample_image_name+".png"))

    return json_info
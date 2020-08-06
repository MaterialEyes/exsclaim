import os
import cv2
import json
import yaml
import glob
import torch
import shutil
import pathlib
import time

import numpy as np
import torch.nn.functional as F

from skimage import io
from scipy.special import softmax
from torch.autograd import Variable
from PIL import Image, ImageDraw, ImageFont

from .figures.models.yolov3 import *
from .figures.utils.utils import *
from .figures.models.network import *
from .tool import ExsclaimTool
from . import utils

class FigureSeparator(ExsclaimTool):
    """ 
    FigureSeparator object.
    Separate subfigure images from full figure image
    using CNN trained on crowdsourced labeled figures
    Parameters:
    None
    """
    def __init__(self , model_path=""):
        super().__init__(model_path)

    def _load_model(self):
        ## Set configuration variables
        model_path = os.path.dirname(__file__)+'/figures/'
        configuration_file = model_path + "config/yolov3_default_subfig.cfg"
        with open(configuration_file, 'r') as f:
            configuration = yaml.load(f, Loader=yaml.FullLoader)

        self.image_size            = configuration['TEST']['IMGSIZE']
        self.nms_threshold         = configuration['TEST']['NMSTHRE']
        self.confidence_threshold  = 0.0001
        self.gpu_id                = 1
        self.cuda = torch.cuda.is_available() and (gpu_id >= 0)
        self.dtype = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        if self.cuda:
            print("using cuda: ", args.gpu_id) 
            torch.cuda.set_device(device=args.gpu_id)

        ## Load object detection model
        object_detection_checkpoint = model_path + "checkpoints/object_detection_model.pt"
        object_detection_model = YOLOv3(configuration['MODEL'])
        if self.cuda:
            object_detection_model.load_state_dict(torch.load(object_detection_checkpoint))
            object_detection_model = object_detection_model.cuda()
        else:
            object_detection_model.load_state_dict(torch.load(object_detection_checkpoint,map_location='cpu'))
        self.object_detection_model = object_detection_model

        ## Load text recognition model
        text_recognition_checkpoint = pathlib.Path(__file__).parent / 'figures' / 'checkpoints' / 'text_recognition_model.pt'
        text_recognition_model = None
        for m in object_detection_model.module_list:
            if hasattr(m,"classifier_model"):
                text_recognition_model = m.classifier_model
                break
        assert text_recognition_model != None
        if self.cuda:
            text_recognition_model.load_state_dict(torch.load(text_recognition_checkpoint))
            text_recognition_model = text_recognition_model.cuda()  
        else:
            text_recognition_model.load_state_dict(torch.load(text_recognition_checkpoint, map_location='cpu'))
        self.text_recognition_model = text_recognition_model

        ## Load classification model
        classifier_checkpoint = model_path + "checkpoints/classifier_model.pt" 
        master_config_file = model_path + "config/yolov3_default_master.cfg"     
        with open(master_config_file, 'r') as f:
            master_config = yaml.load(f, Loader=yaml.FullLoader)
        classifier_model = YOLOv3img(master_config['MODEL'])
        if self.cuda:
            classifier_model.load_state_dict(torch.load(classifier_checkpoint))
            classifier_model = model.cuda()
        else:
            classifier_model.load_state_dict(torch.load(classifier_checkpoint,map_location='cpu'))
        self.classifier_model = classifier_model


    def _update_exsclaim(self,exsclaim_dict,figure_name,figure_dict):
        figure_name = figure_name.split("/")[-1]
        for master_image in figure_dict['figure_separator_results'][0]['master_images']:
            exsclaim_dict[figure_name]['master_images'].append(master_image)

        for unassigned in figure_dict['figure_separator_results'][0]['unassigned']:
            exsclaim_dict[figure_name]['unassigned']['master_images'].append(unassigned)
        return exsclaim_dict

    def _appendJSON(self,filename,json_dict):
        with open(filename,'w') as f: 
            json.dump(json_dict, f, indent=3)

    def run(self,search_query,exsclaim_dict):
        utils.Printer("Running Figure Separator\n")
        os.makedirs(search_query['results_dir'], exist_ok=True)
        t0 = time.time()
        self._load_model()
        counter = 1
        figures = self.get_figure_paths(search_query)
        for figure_name in figures:
            utils.Printer(">>> ({0} of {1}) ".format(counter,+\
                len(figures))+\
                "Extracting images from: "+figure_name.split("/")[-1])
            #try:
            figure_dict = self.extract_image_objects(figure_name, search_query['results_dir'])
            exsclaim_dict = self._update_exsclaim(exsclaim_dict,figure_name,figure_dict)
            #except:
            #    utils.Printer("<!> ERROR: An exception occurred in FigureSeparator\n")
            
            # Save to file every N iterations (to accomodate restart scenarios)
            if counter%500 == 0:
                self._appendJSON(search_query['results_dir']+'_fs.json',exsclaim_dict)
            counter += 1
        
        t1 = time.time()
        utils.Printer(">>> Time Elapsed: {0:.2f} sec ({1} figures)\n".format(t1-t0,int(counter-1)))
        # -------------------------------- #  
        # -- Save current exsclaim_dict -- #
        # -------------------------------- # 
        with open(search_query['results_dir']+'_fs.json', 'w') as f:
            json.dump(exsclaim_dict, f, indent=3)
        # -------------------------------- #  
        # -------------------------------- # 
        # -------------------------------- # 
        return exsclaim_dict


    def get_figure_paths(self, search_query: dict) -> list:
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

    def detect_subfigure_boundaries(self, figure_path):
        # Preprocess the figure for the models
        img = io.imread(figure_path)
        if len(np.shape(img)) == 2:
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img,cv2.COLOR_RGBA2RGB)
        
        img, info_img = preprocess(img, self.image_size, jitter=0)
        
        img = np.transpose(img / 255., (2, 0, 1))
        img = np.copy(img)
        img = torch.from_numpy(img).float().unsqueeze(0)
        img = Variable(img.type(self.dtype))

        # prediction
        img_raw = Image.open(figure_path).convert("RGB")
        width, height = img_raw.size
        with torch.no_grad():
            outputs = self.object_detection_model(img)
            outputs = postprocess(outputs, dtype=self.dtype, 
                        conf_thre=self.confidence_threshold, nms_thre=self.nms_threshold)

        bboxes = list()
        confidences = []

        if outputs[0] is None:
            print("No Objects Detected!!")
        else:
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in outputs[0]:
                box = yolobox2label([y1.data.cpu().numpy(), x1.data.cpu().numpy(), y2.data.cpu().numpy(), x2.data.cpu().numpy()], info_img)
                box[0] = int(min(max(box[0],0),width-1))
                box[1] = int(min(max(box[1],0),height-1))
                box[2] = int(min(max(box[2],0),width))
                box[3] = int(min(max(box[3],0),height))
                if box[2]-box[0] > 5 and box[3]-box[1] > 5:
                    bboxes.append(box)
                    confidences.append("%.3f"%(cls_conf.item()))
        return bboxes, confidences, img_raw

    def detect_subfigure_labels(self, bboxes, confidences, img_raw):
        width, height = img_raw.size
        binary_img = np.zeros((height,width,1))

        detected_labels = []
        detected_bboxes = []
        for i in range(len(bboxes)):
            img_patch = img_raw.crop(tuple(bboxes[i]))
            img_patch = np.array(img_patch)[:,:,::-1]
            img_patch, _ = preprocess(img_patch, 28, jitter=0)
            img_patch = np.transpose(img_patch / 255., (2, 0, 1))
            img_patch = torch.from_numpy(img_patch).type(self.dtype).unsqueeze(0)
            label_prediction = self.text_recognition_model(img_patch)
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
        pair_info = []
        # pair_info is list of tuples, each tuple being (x1, y1, x2, y2, label's alphabetical number) where coords are percentages
        for i in range(len(detected_labels)):
            label_value = detected_labels[i]
            if (ord(label_value) - ord("a")) < (len(detected_labels)+2):
                conf,x1,y1,x2,y2 = detected_bboxes[i]
                if (x2-x1) < 64 and (y2-y1)< 64: # Made this bigger because it was missing some images with labels
                    binary_img[y1:y2,x1:x2] = 255
                    text = "%s %f %d %d %d %d\n"%(label_value, conf, x1, y1, x2, y2)
                    pair_info.append(tuple([x1/width,y1/height,x2/width,y2/height, ord(label_value)-ord("a")]))
        # save concatenate image and pair info
        concate_img = np.concatenate((np.array(img_raw),binary_img),axis=2)
        
        return pair_info, concate_img

    def classify_subfigures(self, pair_info, concate_img, figure_path, img_raw):
        label_names = ["background","microscopy","parent","graph","illustration","diffraction","basic_photo",
                    "unclear","OtherSubfigure","a","b","c","d","e","f"]
        width, height = img_raw.size

        img = concate_img[...,:3].copy()
        mask = concate_img[...,3:].copy()

        img, info_img = preprocess(img, self.image_size, jitter=0)
        img = np.transpose(img / 255., (2, 0, 1))
        mask = preprocess_mask(mask, self.image_size, info_img)
        mask = np.transpose(mask / 255., (2, 0, 1))
        new_concate_img = np.concatenate((img,mask),axis=0)
        img = torch.from_numpy(new_concate_img).float().unsqueeze(0)
        img = Variable(img.type(self.dtype))

        subfigure_labels = []
        for subfigure in pair_info:        
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
            # convert coco labels to yolo
            subfigure_labels = label2yolobox(subfigure_labels, info_img, self.image_size, lrflip=False)
            subfigure_padded_labels[range(len(subfigure_labels))[:80]
                        ] = subfigure_labels[:80]
        subfigure_padded_labels = (torch.from_numpy(subfigure_padded_labels)).float().unsqueeze(0)
        subfigure_padded_labels = Variable(subfigure_padded_labels.type(self.dtype))
        padded_label_list = [None, subfigure_padded_labels]
        assert subfigure_padded_labels.size()[0] == 1

        # prediction
        img_raw = Image.fromarray(np.uint8(concate_img[...,:3].copy()[...,::-1]))
        width, height = img_raw.size
        with torch.no_grad():
            outputs = self.classifier_model(img, padded_label_list)

        # select the 13x13 grid as feature map
        feature_size = [13,26,52]
        feature_index = 0
        preds = outputs[feature_index]
        preds = preds[0].data.cpu().numpy()
        
        ## List of tuples where each tuple represents a subfigures as
        ## (x1, y1, x2, y2, subfigure label, )
        subfigure_info = []

        ## Documentation
        current_info = {}
        current_info["figure_name"] = figure_path.split("/")[-1]
        current_info["master_images"] = []
        current_info["unassigned"] = []
        json_info = {}
        json_info["figure_separator_results"] = []

        full_figure_is_master = True if len(pair_info) == 0 else False

        # max to handle case where pair info has only 1 (the full figure is the master image)
        for subfigure_id in range(0, max(len(pair_info), 1)):   
            sub_cat,x,y,w,h = (subfigure_padded_labels[0,subfigure_id] * 13 ).to(torch.int16).data.cpu().numpy()
            best_anchor = np.argmax(preds[:,y,x,4])
            tx,ty = np.array(preds[best_anchor,y,x,:2]/32,np.int32)
            best_anchor = np.argmax(preds[:,ty,tx,4])
            x,y,w,h = preds[best_anchor,ty,tx,:4]
            cls = np.argmax(preds[best_anchor,int(ty),int(tx),5:])
            master_label = label_names[cls]
            subfigure_label = chr(int(sub_cat/feature_size[feature_index])+ord("a"))

            master_cls_conf = max(softmax(preds[best_anchor,int(ty),int(tx),5:]))
            master_obj_conf = preds[best_anchor,ty,tx,4]

            if full_figure_is_master:
                x1 = 0
                x2 = np.shape(img_raw)[1]
                y1 = 0
                y2 = np.shape(img_raw)[0]
                subfigure_label = "0"

            else: 
                x1 = (x-w/2)
                x2 = (x+w/2)
                y1 = (y-h/2)
                y2 = (y+h/2)
    
                x1,y1,x2,y2 = yolobox2label([y1,x1,y2,x2], info_img)
    
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

            if not full_figure_is_master:
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
        
        return json_info


    def make_visualization(self, image_raw, x1, x2, y1, y2, master_cls_conf, master_obj_conf,
                        master_label, subfigure_label):
        sample_image_name = ".".join(figure_path.split("/")[-1].split(".")[0:-1])
        # ## Make and save images
        # result_image = Image.new(mode="RGB",size=(200,(len(pair_info)+1)*100-50))
        # draw = ImageDraw.Draw(result_image)
        # font = ImageFont.load_default()

        # # headline text
        # text = "labels"
        # draw.text((10,10),text,fill="white",font=font)
        # text = "master image"
        # draw.text((110,10),text,fill="white",font=font)
        
        # label_img = img_raw.copy()
        # img_draw = ImageDraw.Draw(label_img)

        # # max to handle case where pair info has only 1 (the full figure is the master image)
        # for subfigure_id in range(0, max(len(pair_info), 1)):   
            
        #     # visualization
        #     patch = img_raw.crop((int(x1),int(y1),int(x2),int(y2)))
            
        #     text = "%s %f %d %d %d %d\n"%(master_label, master_cls_conf*master_obj_conf, int(x1), int(y1), int(x2), int(y2))

        #     os.makedirs(save_path+"/extractions", exist_ok=True)
        #     with open(os.path.join(save_path+"/extractions/",sample_image_name+".txt"),"a+") as results_file:
        #         results_file.write(text)
        #     img_draw.line([(x1,y1),(x1,y2),(x2,y2),(x2,y1),(x1,y1)], fill=(255,0,0), width=3)
        #     img_draw.rectangle((x2-100,y2-30,x2,y2),fill=(0,255,0))
        #     img_draw.text((x2-100+2,y2-30+2),"{}, {}".format(master_label,subfigure_label),fill=(255,0,0))
            
        #     text = "%s\n%s"%(master_label,subfigure_label)
        #     draw.text((10,60+100*subfigure_id),text,fill="white",font=font)
            
        #     pw,ph = patch.size
        #     if pw>ph:
        #         ph = max(1,ph/pw*80)
        #         pw = 80
        #     else:
        #         pw = max(1,pw/ph*80)
        #         ph = 80

        #     patch = patch.resize((int(pw),int(ph)))
        #     result_image.paste(patch,box=(110,60+100*subfigure_id))

        # del draw
        # result_image.save(os.path.join(save_path+"/extractions/"+sample_image_name+".png"))   


    def extract_image_objects(self, figure_path=str, save_path="") -> "figure_dict":
        """
        Find individual image objects within a figure and classify based on functionality

        Args:
            figure_separator_model: A tuple (model, confidence_threshold, nms_threshold, img_size, gpu)
            figure_path: A path to the figure to separate

        Returns:
            figure_dict: A dictionary with classified image_objects extracted from figure
        """
        # Unpack models and eval
        self.object_detection_model.eval()
        self.text_recognition_model.eval()
        self.classifier_model.eval()
        
        ## Detect the bounding boxes of each subfigure
        bboxes, confidences, img_raw = self.detect_subfigure_boundaries(figure_path)

        ## Detect the subfigure labels on each of the bboxes found
        pair_info, concate_img = self.detect_subfigure_labels(bboxes, confidences, img_raw)
            
        json_info = self.classify_subfigures(pair_info, concate_img, figure_path, img_raw)
            
        return json_info
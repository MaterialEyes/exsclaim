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
import torchvision.models.detection
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import models

from .figures.models.yolov3 import *
from .figures.utils import *
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
    def __init__(self , search_query):
        self._load_model()
        self.exsclaim_json = {}


    def _load_model(self):
        """ Load relevant models for the object detection tasks """
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
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
        text_recognition_model = resnet152()
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
            classifier_model = classifier_model.cuda()
        else:
            classifier_model.load_state_dict(torch.load(classifier_checkpoint,map_location='cpu'))
        self.classifier_model = classifier_model

        ## Load scale bar detection model
        scale_bar_detection_checkpoint = model_path + "checkpoints/scale_bar_detection_model.pt"
        # load an object detection model pre-trained on COCO
        scale_bar_detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        input_features = scale_bar_detection_model.roi_heads.box_predictor.cls_score.in_features
        number_classes = 3      # background, scale bar, scale bar label
        scale_bar_detection_model.roi_heads.box_predictor = FastRCNNPredictor(input_features, number_classes)
        if self.cuda:
            scale_bar_detection_model.load_state_dict(torch.load(scale_bar_detection_checkpoint))
            scale_bar_detection_model = scale_bar_detection_model.cuda()
        else:
            scale_bar_detection_model.load_state_dict(torch.load(scale_bar_detection_checkpoint, map_location='cpu'))
        self.scale_bar_detection_model = scale_bar_detection_model

        ## Code to set up scale label reading model(s)
        full_scale_bar_reader = os.path.join(model_path, "checkpoints", "all_18-120.pt")
        unit_scale_bar_reader = os.path.join(model_path, "checkpoints", "unit_data_18-335.pt")
        number_scale_bar_reader = os.path.join(model_path, "checkpoints", "scale_all_18-124.pt")

        # load models of choice
        self.full_scale_bar_reader = self.get_classification_model(full_scale_bar_reader)
        self.unit_scale_bar_reader = self.get_classification_model(unit_scale_bar_reader)
        self.number_scale_bar_reader = self.get_classification_model(number_scale_bar_reader)

        ## Save scale laber reader transforms
        self.label_reader_transforms = T.Compose([T.Resize((224, 224)),
                                        T.ToTensor(),])


    def get_classification_model(self, scale_label_recognition_checkpoint, pretrained=True):
        """ """
        all = {0: '0.1 A', 1: '0.1 nm', 2: '0.1 um', 3: '0.2 A', 4: '0.2 nm', 5: '0.2 um', 6: '0.3 A', 7: '0.3 nm', 8: '0.3 um', 9: '0.4 A', 10: '0.4 nm', 11: '0.4 um', 12: '0.5 A', 13: '0.5 nm', 14: '0.5 um', 15: '0.6 A', 16: '0.6 nm', 17: '0.6 um', 18: '0.7 A', 19: '0.7 nm', 20: '0.7 um', 21: '0.8 A', 22: '0.8 nm', 23: '0.8 um', 24: '0.9 A', 25: '0.9 nm', 26: '0.9 um', 27: '1 A', 28: '1 nm', 29: '1 um', 30: '10 A', 31: '10 nm', 32: '10 um', 33: '100 A', 34: '100 nm', 35: '100 um', 36: '2 A', 37: '2 nm', 38: '2 um', 39: '2.5 A', 40: '2.5 nm', 41: '2.5 um', 42: '20 A', 43: '20 nm', 44: '20 um', 45: '200 A', 46: '200 nm', 47: '200 um', 48: '25 A', 49: '25 nm', 50: '25 um', 51: '250 A', 52: '250 nm', 53: '250 um', 54: '3 A', 55: '3 nm', 56: '3 um', 57: '30 A', 58: '30 nm', 59: '30 um', 60: '300 A', 61: '300 nm', 62: '300 um', 63: '4 A', 64: '4 nm', 65: '4 um', 66: '40 A', 67: '40 nm', 68: '40 um', 69: '400 A', 70: '400 nm', 71: '400 um', 72: '5 A', 73: '5 nm', 74: '5 um', 75: '50 A', 76: '50 nm', 77: '50 um', 78: '500 A', 79: '500 nm', 80: '500 um', 81: '6 A', 82: '6 nm', 83: '6 um', 84: '60 A', 85: '60 nm', 86: '60 um', 87: '600 A', 88: '600 nm', 89: '600 um', 90: '7 A', 91: '7 nm', 92: '7 um', 93: '70 A', 94: '70 nm', 95: '70 um', 96: '700 A', 97: '700 nm', 98: '700 um', 99: '8 A', 100: '8 nm', 101: '8 um', 102: '80 A', 103: '80 nm', 104: '80 um', 105: '800 A', 106: '800 nm', 107: '800 um', 108: '9 A', 109: '9 nm', 110: '9 um', 111: '90 A', 112: '90 nm', 113: '90 um', 114: '900 A', 115: '900 nm', 116: '900 um'}
        some = {0: '0.1 A', 1: '0.1 nm', 2: '0.1 um', 3: '0.2 A', 4: '0.2 nm', 5: '0.2 um', 6: '0.3 A', 7: '0.3 nm', 8: '0.3 um', 9: '0.4 A', 10: '0.4 nm', 11: '0.4 um', 12: '0.5 A', 13: '0.5 nm', 14: '0.5 um', 15: '1 A', 16: '1 nm', 17: '1 um', 18: '10 A', 19: '10 nm', 20: '10 um', 21: '100 A', 22: '100 nm', 23: '100 um', 24: '2 A', 25: '2 nm', 26: '2 um', 27: '2.5 A', 28: '2.5 nm', 29: '2.5 um', 30: '20 A', 31: '20 nm', 32: '20 um', 33: '200 A', 34: '200 nm', 35: '200 um', 36: '25 A', 37: '25 nm', 38: '25 um', 39: '250 A', 40: '250 nm', 41: '250 um', 42: '3 A', 43: '3 nm', 44: '3 um', 45: '30 A', 46: '30 nm', 47: '30 um', 48: '300 A', 49: '300 nm', 50: '300 um', 51: '4 A', 52: '4 nm', 53: '4 um', 54: '40 A', 55: '40 nm', 56: '40 um', 57: '400 A', 58: '400 nm', 59: '400 um', 60: '5 A', 61: '5 nm', 62: '5 um', 63: '50 A', 64: '50 nm', 65: '50 um', 66: '500 A', 67: '500 nm', 68: '500 um'}        
        scale_some = {0: '0.1', 1: '0.2', 2: '0.3', 3: '0.4', 4: '0.5', 5: '1', 6: '10', 7: '100', 8: '2', 9: '2.5', 10: '20', 11: '200', 12: '25', 13: '250', 14: '3', 15: '30', 16: '300', 17: '4', 18: '40', 19: '400', 20: '5', 21: '50', 22: '500'}
        scale_all = {0: '0.1', 1: '0.2', 2: '0.3', 3: '0.4', 4: '0.5', 5: '0.6', 6: '0.7', 7: '0.8', 8: '0.9', 9: '1', 10: '10', 11: '100', 12: '2', 13: '2.5', 14: '20', 15: '200', 16: '25', 17: '250', 18: '3', 19: '30', 20: '300', 21: '4', 22: '40', 23: '400', 24: '5', 25: '50', 26: '500', 27: '6', 28: '60', 29: '600', 30: '7', 31: '70', 32: '700', 33: '8', 34: '80', 35: '800', 36: '9', 37: '90', 38: '900'}
        unit_data = {0: 'A', 1: 'mm', 2: 'nm', 3: 'um'}
        dataset_to_label_dict = {
            "all": all,
            "some" : some,
            "scale_some": scale_some,
            "scale_all": scale_all,
            "unit_data": unit_data
        }
        filename = scale_label_recognition_checkpoint.split(".")[0].split("/")[-1]
        model_name, epoch = filename.split("-")
        tokens = model_name.split("_")
        dataset = "_".join(tokens[:-1])
        depth = int(tokens[-1])
        self.id_to_class_full = all if dataset == 'all' else some
        self.id_to_class_number = scale_all if dataset == 'scale_all' else scale_some
        self.id_to_class_unit = unit_data
        classes = len(dataset_to_label_dict[dataset])
        # load an object detection model pre-trained on COCO
        if depth == 18:
            model = models.resnet18(pretrained=pretrained)
        elif depth == 50:
            model = models.resnet50(pretrained=pretrained)
        elif depth == 152:
            model = models.resnet152(pretrained=pretrained)
        
        if depth == 18:
            model.fc = nn.Sequential(nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(512, classes),
                                    nn.LogSoftmax(dim=1))
        else:
            model.fc = nn.Sequential(nn.Linear(2048, 512),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(512, classes),
                                    nn.LogSoftmax(dim=1))
        model.to(self.device)
        
        if self.cuda:
            model.load_state_dict(torch.load(scale_label_recognition_checkpoint))
            model = model.cuda()
        else:
            model.load_state_dict(torch.load(scale_label_recognition_checkpoint, map_location='cpu')["model_state_dict"])
        
        model.eval()
        return model


    def _update_exsclaim(self, exsclaim_dict, figure_name, figure_dict):
        figure_name = figure_name.split("/")[-1]
        for master_image in figure_dict['figure_separator_results'][0]['master_images']:
            exsclaim_dict[figure_name]['master_images'].append(master_image)

        for unassigned in figure_dict['figure_separator_results'][0]['unassigned']:
            exsclaim_dict[figure_name]['unassigned']['master_images'].append(unassigned)
        return exsclaim_dict


    def _appendJSON(self, results_directory, exsclaim_json, figures_separated):
        """ Commit updates to EXSCLAIM JSON and updates list of separated figures

        Args:
            results_directory (string): Path to results directory
            exsclaim_json (dict): Updated EXSCLAIM JSON
            figures_separated (set): Figures which have already been separated
        """
        with open(results_directory + "exsclaim.json", 'w') as f: 
            json.dump(exsclaim_json, f, indent=3)
        with open(results_directory + "_figures", "a+") as f:
            for figure in figures_separated:
                f.write("%s\n" % figure.split("/")[-1])


    def run(self, search_query, exsclaim_dict):
        """ Run the models relevant to manipulating article figures
        """
        utils.Printer("Running Figure Separator\n")
        os.makedirs(search_query['results_dir'], exist_ok=True)
        self.exsclaim_json = exsclaim_dict
        t0 = time.time()
        ## List figures that have already been separated
        if os.path.isfile(search_query["results_dir"] + "_figures"):
            with open(search_query["results_dir"] + "_figures", "r") as f:
                contents = f.readlines()
            figures_separated = {f.strip() for f in contents}
        else:
            figures_separated = set()
        new_figures_separated = set()

        counter = 1
        figures = ([self.exsclaim_json[figure]["figure_path"] for figure in self.exsclaim_json 
                    if self.exsclaim_json[figure]["figure_name"] not in figures_separated])
        for figure_name in figures:
            utils.Printer(">>> ({0} of {1}) ".format(counter,+\
                len(figures))+\
                "Extracting images from: "+ figure_name.split("/")[-1])
            try:
                self.extract_image_objects(figure_name)
                new_figures_separated.add(figure_name)
            except:
                utils.Printer("<!> ERROR: An exception occurred in FigureSeparator\n")
            
            # Save to file every N iterations (to accomodate restart scenarios)
            if counter%500 == 0:
                self._appendJSON(search_query['results_dir'], self.exsclaim_json, new_figures_separated)
            counter += 1
        
        t1 = time.time()
        utils.Printer(">>> Time Elapsed: {0:.2f} sec ({1} figures)\n".format(t1-t0,int(counter-1)))
        self._appendJSON(search_query["results_dir"], self.exsclaim_json, new_figures_separated)
        return self.exsclaim_json


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
        """ Detects the bounding boxes of subfigures in figure_path

        Args:
            figure_path: A string, path to an image of a figure
                from a scientific journal
        Returns:
            subfigure_info (list of lists): Each inner list is
                x1, y1, x2, y2, confidence 
        """
        ## Preprocess the figure for the models
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

        img_raw = Image.open(figure_path).convert("RGB")
        width, height = img_raw.size
        
        ## Run model on figure
        with torch.no_grad():
            outputs = self.object_detection_model(img)
            outputs = postprocess(outputs, dtype=self.dtype, 
                        conf_thre=self.confidence_threshold, nms_thre=self.nms_threshold)

        ## Reformat model outputs to display bounding boxes in our desired format
        ## List of lists where each inner list is [x1, y1, x2, y2, confidence]
        subfigure_info = list()

        if outputs[0] is None:
            print("No Objects Detected!!")
            return subfigure_info

        for x1, y1, x2, y2, conf, cls_conf, cls_pred in outputs[0]:
            box = yolobox2label([y1.data.cpu().numpy(), x1.data.cpu().numpy(), y2.data.cpu().numpy(), x2.data.cpu().numpy()], info_img)
            box[0] = int(min(max(box[0],0),width-1))
            box[1] = int(min(max(box[1],0),height-1))
            box[2] = int(min(max(box[2],0),width))
            box[3] = int(min(max(box[3],0),height))
            # ensures no extremely small (likely incorrect) boxes are counted
            small_box_threshold = 5
            if (box[2]-box[0] > small_box_threshold and 
                box[3]-box[1] > small_box_threshold):
                box.append("%.3f"%(cls_conf.item()))
                subfigure_info.append(box)
        return subfigure_info


    def detect_subfigure_labels(self, figure_path, subfigure_info):
        """ Uses text recognition to read subfigure labels from figure_path
        
        Note: 
            To get sensible results, should be run only after
            detect_subfigure_boundaries has been run
        Args:
            figure_path (str): A path to the image (.png, .jpg, or .gif)
                file containing the article figure
            subfigure_info (list of lists): Details about bounding boxes
                of each subfigure from detect_subfigure_boundaries(). Each
                inner list has format [x1, y1, x2, y2, confidence] where
                x1, y1 are upper left bounding box coordinates as ints, 
                x2, y2, are lower right, and confidence the models confidence
        Returns:
            subfigure_info (list of tuples): Details about bounding boxes and 
                labels of each subfigure in figure. Tuples for each subfigure are
                (x1, y1, x2, y2, label) where x1, y1 are upper left x and y coord
                divided by image width/height and label is the an integer n 
                meaning the label is the nth letter
            concate_img (np.ndarray): A numpy array representing the figure.
                Used in classify_subfigures. Ideally this will be removed to 
                increase modularity. 
        """
        img_raw = Image.open(figure_path).convert("RGB")
        img_raw = img_raw.copy()
        width, height = img_raw.size
        binary_img = np.zeros((height,width,1))

        detected_labels = []
        detected_bboxes = []
        for subfigure in subfigure_info:
            ## Preprocess the image for the model
            bbox = tuple(subfigure[:4])
            img_patch = img_raw.crop(bbox)
            img_patch = np.array(img_patch)[:,:,::-1]
            img_patch, _ = preprocess(img_patch, 28, jitter=0)
            img_patch = np.transpose(img_patch / 255., (2, 0, 1))
            img_patch = torch.from_numpy(img_patch).type(self.dtype).unsqueeze(0)

            ## Run model on figure
            label_prediction = self.text_recognition_model(img_patch)
            label_confidence = np.amax(F.softmax(label_prediction, dim=1).data.cpu().numpy())
            label_value = chr(label_prediction.argmax(dim=1).data.cpu().numpy()[0]+ord("a"))
            if label_value == "z":
                continue

            ## Reformat results for to desired format
            x1,y1,x2,y2, box_confidence = subfigure
            total_confidence = float(box_confidence)*label_confidence
            if label_value in detected_labels:
                label_index = detected_labels.index(label_value)
                if total_confidence > detected_bboxes[label_index][0]:
                    detected_bboxes[label_index] = [total_confidence,x1,y1,x2,y2]
            else:
                detected_labels.append(label_value)
                detected_bboxes.append([total_confidence,x1,y1,x2,y2])
        assert len(detected_labels) == len(detected_bboxes)

        ## subfigure_info (list of tuples): [(x1, y1, x2, y2, label) 
        ##  where x1, y1 are upper left x and y coord divided by image width/height
        ##  and label is the an integer n meaning the label is the nth letter  
        subfigure_info = []
        for i, label_value in enumerate(detected_labels):
            if (ord(label_value) - ord("a")) >= (len(detected_labels) + 2):
                continue
            conf,x1,y1,x2,y2 = detected_bboxes[i]
            if (x2-x1) < 64 and (y2-y1)< 64: # Made this bigger because it was missing some images with labels
                binary_img[y1:y2,x1:x2] = 255
                label = ord(label_value) - ord("a")
                subfigure_info.append((label, float(x1), float(y1), float(x2-x1), float(y2-y1)))
        # concate_img needed for classify_subfigures
        concate_img = np.concatenate((np.array(img_raw),binary_img),axis=2)
        
        return subfigure_info, concate_img


    def classify_subfigures(self, figure_path, subfigure_labels, concate_img):
        """ Classifies the type of image each subfigure in figure_path 

        Note: 
            To get sensible results, should be run only after
            detect_subfigure_boundaries and detect_subfigure_labels have run
        Args:
            figure_path (str): A path to the image (.png, .jpg, or .gif)
                file containing the article figure
            subfigure_labels (list of tuples): Information about each subfigure.
                Each tuple represents a single subfigure in the figure_path
                figure. The tuples are (label, x, y, width, height) where 
                label is the n for the nth letter in the alphabet and x, y,
                width, and height are percentages of the image width and height
            concate_img (np.ndarray): A numpy array representing the figure.
                Has been modified in detect_subfigure_labels. Ideally this 
                parameter will be removed to increase modularity.
        Returns:
            figure_json (dict): A figure json describing the data collected.
        Modifies:
            self.exsclaim_json (dict): Adds figure_json to exsclaim_json
        """
        label_names = ["background", "microscopy", "parent", "graph", 
                       "illustration", "diffraction", "basic_photo", "unclear",
                       "OtherSubfigure", "a", "b", "c", "d", "e", "f"]
        img = concate_img[...,:3].copy()
        mask = concate_img[...,3:].copy()

        img, info_img = preprocess(img, self.image_size, jitter=0)
        img = np.transpose(img / 255., (2, 0, 1))
        mask = preprocess_mask(mask, self.image_size, info_img)
        mask = np.transpose(mask / 255., (2, 0, 1))
        new_concate_img = np.concatenate((img,mask),axis=0)
        img = torch.from_numpy(new_concate_img).float().unsqueeze(0)
        img = Variable(img.type(self.dtype))

        subfigure_labels_copy = subfigure_labels.copy()
        
        subfigure_padded_labels = np.zeros((80, 5))
        if len(subfigure_labels) > 0:
            subfigure_labels = np.stack(subfigure_labels)
            # convert coco labels to yolo
            subfigure_labels = label2yolobox(subfigure_labels, info_img, self.image_size, lrflip=False)
            # make the beginning of subfigure_padded_labels subfigure_labels
            subfigure_padded_labels[:len(subfigure_labels)] = subfigure_labels[:80]
        # conver labels to tensor and add dimension
        subfigure_padded_labels = (torch.from_numpy(subfigure_padded_labels)).unsqueeze(0)
        subfigure_padded_labels = Variable(subfigure_padded_labels.type(self.dtype))
        padded_label_list = [None, subfigure_padded_labels]
        assert subfigure_padded_labels.size()[0] == 1

        # prediction
        with torch.no_grad():
            outputs = self.classifier_model(img, padded_label_list)

        # select the 13x13 grid as feature map
        feature_size = [13,26,52]
        feature_index = 0
        preds = outputs[feature_index]
        preds = preds[0].data.cpu().numpy()

        ## Documentation
        figure_name = figure_path.split("/")[-1]
        figure_json = self.exsclaim_json.get(figure_name, {})
        figure_json["figure_name"] = figure_name
        figure_json.get("master_images", [])
        # create an unassigned field with a master images field if it doesn't exist
        figure_json.get("unassigned", {'master_images' : []}).get('master_images', [])


        full_figure_is_master = True if len(subfigure_labels) == 0 else False

        # max to handle case where pair info has only 1 (the full figure is the master image)
        for subfigure_id in range(0, max(len(subfigure_labels), 1)):   
            sub_cat,x,y,w,h = (subfigure_padded_labels[0,subfigure_id] * 13 ).to(torch.int16).data.cpu().numpy()
            best_anchor = np.argmax(preds[:,y,x,4])
            tx,ty = np.array(preds[best_anchor,y,x,:2]/32,np.int32)
            best_anchor = np.argmax(preds[:,ty,tx,4])
            x,y,w,h = preds[best_anchor,ty,tx,:4]
            classification = np.argmax(preds[best_anchor,int(ty),int(tx),5:])
            master_label = label_names[classification]
            subfigure_label = chr(int(sub_cat/feature_size[feature_index])+ord("a"))
            master_cls_conf = max(softmax(preds[best_anchor,int(ty),int(tx),5:]))

            if full_figure_is_master:
                img_raw = Image.fromarray(np.uint8(concate_img[...,:3].copy()[...,::-1]))
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

            ## Saving the data into a json. Eventually it would be good to make the json
            ## be updated in each model's function. This could eliminate the need to pass
            ## arguments from function to function. Currently the coordinates in 
            ## subfigure_info are different from those output from classifier model. Also
            ## concate_image depends on operations performed in detect_subfigure_labels()
            master_image_info = {}
            master_image_info["classification"] = master_label
            master_image_info["confidence"] = float("{0:.4f}".format(master_cls_conf))
            master_image_info["height"] = y2 - y1
            master_image_info["width"] = x2 -x1
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
                        geometry = { "x" : x, "y" : y}
                        subfigure_label_info["geometry"].append(geometry)
            master_image_info["subfigure_label"] = subfigure_label_info
            figure_json.get("master_images", []).append(master_image_info)
            
        self.exsclaim_json[figure_name] = figure_json
        return figure_json

    def read_scale_bar(self, cropped_image):
        """ Outputs the text of an image cropped to a scale bar label bbox 

        Args:
            cropped_image (Image): An PIL RGB image cropped to the bounding box
                of a scale bar label. 
        Returns:
            label_text (string): The text of the scale bar label
        """
        return self.read_scale_bar_parts(cropped_image)

    def read_scale_bar_full(self, cropped_image):
        """ Outputs the text of an image cropped to a scale bar label bbox 

        Args:
            cropped_image (Image): An PIL RGB image cropped to the bounding box
                of a scale bar label. 
        Returns:
            label_text (string): The text of the scale bar label
        """
        cropped_image = self.label_reader_transforms(cropped_image)
        cropped_image =  cropped_image.unsqueeze(0)

        with torch.no_grad():
            log_probabilities = self.full_scale_bar_reader(cropped_image)
        probabilities = torch.exp(log_probabilities)
        probabilities = list(probabilities.numpy()[0])
        predicted_idx = probabilities.index(max(probabilities))
        label_text = self.id_to_class_full[predicted_idx]
        return label_text, probabilities[predicted_idx]

    def read_scale_bar_parts(self, cropped_image):
        """ Outputs the text of an image cropped to a scale bar label bbox 

        Args:
            cropped_image (Image): An image cropped to the bounding box
                of a scale bar label. 
        Returns:
            label_text (string): The text of the scale bar label
        """
        cropped_image = self.label_reader_transforms(cropped_image)
        cropped_image = cropped_image.unsqueeze(0)

        ## Find predicted scale unit
        with torch.no_grad():
            log_probabilities = self.unit_scale_bar_reader(cropped_image)
        probabilities = torch.exp(log_probabilities)
        probabilities = list(probabilities.numpy()[0])
        predicted_idx = probabilities.index(max(probabilities))
        unit_confidence = probabilities[predicted_idx]
        unit_text = self.id_to_class_unit[predicted_idx]

        ## Find predicted scale number
        with torch.no_grad():
            log_probabilities = self.number_scale_bar_reader(cropped_image)
        probabilities = torch.exp(log_probabilities)
        probabilities = list(probabilities.numpy()[0])
        predicted_idx = probabilities.index(max(probabilities))
        number_text = self.id_to_class_number[predicted_idx]

        return number_text + " " + unit_text, probabilities[predicted_idx] * unit_confidence

    def create_scale_bar_objects(self, scale_bar_lines, scale_bar_labels):
        """ Match scale bar lines with labels to create scale bar jsons
        
        Args:
            scale_bar_lines (list of dicts): A list of dictionaries
                representing predicted scale bars with 'geometry', 'length',
                and 'confidence' attributes.
            scale_bar_labels (list of dicts): A list of dictionaries
                representing predicted scale bar labesl with 'geometry',
                'text', 'confidence', 'box_confidence', 'nm' attributes.
        Returns:
            scale_bar_jsons (list of Scale Bar JSONS): Scale Bar JSONS that
                were made from pairing scale labels and scale lines
            unassigned_labels (list of dicts): List of dictionaries
                representing scale bar labels that were not matched.
        """
        scale_bar_jsons = []
        paired_labels = set()
        for line in scale_bar_lines:
            x_line, y_line = utils.find_box_center(line["geometry"])
            best_distance = 1000000
            best_label = None
            for label_index, label in enumerate(scale_bar_labels):
                x_label, y_label = utils.find_box_center(label["geometry"])
                distance = (x_label - x_line)**2 + (y_label - y_line)**2
                if distance < best_distance:
                    best_distance = distance
                    best_index = label_index
                    best_label = label
            # If the best match is not very good, keep this line unassigned
            if best_distance > 5000:
                best_index = -1
                best_label = None
                best_distance = -1
            paired_labels.add(best_index)
            scale_bar_json = {
                "label" : best_label,
                "geometry" : line["geometry"],
                "confidence" : float(line.get("confidence", None)),
                "length" : line.get("length", None),
                "label_line_distance" : best_distance
            }
            scale_bar_jsons.append(scale_bar_json)
        # Check which labels were left unassigned
        unassigned_labels = []
        for i, label in enumerate(scale_bar_labels):
            if i not in paired_labels:
                unassigned_labels.append(label)
        return scale_bar_jsons, unassigned_labels

    def assign_scale_objects_to_subfigures(self,
                                           master_image,
                                           scale_objects):
        """ Assign scale bar objects to master images 

        Args:
            master_image (Master Image Json): A Master Image JSON
            scale_objects (list of Scale Object JSON): candidate scale objects
        Returns:
            master_image (Master Image JSON): updated with scale objects
            scale_objects: updated with assigned objects removed
        """
        geomtery = master_image["geometry"]
        unassigned_scale_objects = []
        assigned_scale_objects = []
        for scale_object in scale_objects:
            if utils.is_contained(scale_object["geometry"], geomtery):
                assigned_scale_objects.append(scale_object)
            else:
                unassigned_scale_objects.append(scale_object)
        master_image["scale_bars"] = assigned_scale_objects
        # find if there is one unique scale bar label
        nm_to_pixel = 0
        label = ""
        scale_labels = set()
        for scale_object in master_image["scale_bars"]:
            if scale_object["label"]:
                scale_labels.add(scale_object["label"]["nm"])
                nm_to_pixel = (scale_object["label"]["nm"]
                               / float(scale_object["length"]))
                label = scale_object["label"]["text"]
        if len(scale_labels) == 1:
            master_image["nm_height"] = nm_to_pixel * master_image["height"]
            master_image["nm_width"] = nm_to_pixel * master_image["width"]
            master_image["scale_label"] = label

        return master_image, unassigned_scale_objects

    def detect_scale_objects(self, image):
        """ Detects bounding boxes of scale bars and scale bar labels 

        Args:
            image (PIL Image): A PIL image object
        Returns:
            scale_bar_info (list): A list of lists with the following 
                pattern: [[x1,y1,x2,y2, confidence, label],...] where
                label is 1 for scale bars and 2 for scale bar labelss 
        """
        # prediction
        self.scale_bar_detection_model.eval()
        with torch.no_grad():
            outputs = self.scale_bar_detection_model([image])
        # post-process 
        scale_bar_info = []
        for i, box in enumerate(outputs[0]["boxes"]):
            confidence = outputs[0]["scores"][i]
            if confidence > 0.5:
                x1, y1, x2, y2 = box
                label = outputs[0]['labels'][i]
                scale_bar_info.append([x1, y1, x2, y2, confidence, label])
        scale_bar_info = non_max_suppression_malisiewicz(np.asarray(scale_bar_info), 0.4)
        return scale_bar_info

    def determine_scale(self, figure_path, figure_json):
        """ Adds scale information to figure by reading and measuring scale bars 

        Args:
            figure_path (str): A path to the image (.png, .jpg, or .gif)
                file containing the article figure
            figure_json (dict): A Figure JSON
        Returns:
            figure_json (dict): A dictionary with classified image_objects
                extracted from figure
        """
        convert_to_nm = {
            "a"  : 0.1,
            "nm" : 1.0,
            "um" : 1000.0,
            "mm" : 1000000.0,
            "cm" : 10000000.0,
            "m"  : 1000000000.0,
        }
        unassigned = figure_json.get("unassigned", {})
        unassigned_scale_labels = unassigned.get("scale_bar_labels", [])
        master_images = figure_json.get("master_images", [])
        image = Image.open(figure_path).convert("RGB")
        tensor_image = T.ToTensor()(image)
        # Detect scale bar objects
        scale_bar_info = self.detect_scale_objects(tensor_image)
        label_names = ["background", "scale bar", "scale label"]
        scale_bars = []
        scale_labels = []
        for scale_object in scale_bar_info:
            x1, y1, x2, y2, confidence, classification = scale_object
            geometry = utils.convert_coords_to_labelbox([int(x1), int(y1),
                                                        int(x2), int(y2)])
            if label_names[int(classification)] == "scale bar":
                scale_bar_json = {
                    "geometry" : geometry,
                    "confidence" : float(confidence),
                    "length" : int(x2 - x1)
                }
                scale_bars.append(scale_bar_json)
            elif label_names[int(classification)] == "scale label":

                scale_bar_label_image = image.crop((int(x1), int(y1),
                                                    int(x2), int(y2)))
                ## Read Scale Text
                scale_label_text, label_confidence = self.read_scale_bar(
                    scale_bar_label_image)
                magnitude, unit = scale_label_text.split(" ")
                magnitude = float(magnitude)
                length_in_nm = magnitude * convert_to_nm[unit.strip().lower()]
                label_json = {
                    "geometry" : geometry,
                    "text" : scale_label_text,
                    "label_confidence" : float(label_confidence),
                    "box_confidence" : float(confidence),
                    "nm" : length_in_nm
                }
                scale_labels.append(label_json)
        # Match scale bars to labels and to subfigures (master images)
        scale_bar_jsons, unassigned_labels = (
            self.create_scale_bar_objects(scale_bars, scale_labels))
        for master_image in master_images:
            master_image, scale_bar_jsons = (
                self.assign_scale_objects_to_subfigures(master_image,
                                                        scale_bar_jsons))                                                      

        # Save info to JSON
        unassigned["scale_bar_labels"] = unassigned_scale_labels
        unassigned["scale_bar_objects"] = scale_bar_jsons
        figure_json["unassigned"] = unassigned
        figure_json["master_images"] = master_images
        return figure_json
    
    def extract_image_objects(self, figure_path=str) -> "figure_dict":
        """ Separate and classify subfigures in an article figure

        Args:
            figure_path (str): A path to the image (.png, .jpg, or .gif)
                file containing the article figure
        Returns:
            figure_json (dict): A dictionary with classified image_objects
                extracted from figure
        """
        ## Set models to evaluation mode
        self.object_detection_model.eval()
        self.text_recognition_model.eval()
        self.classifier_model.eval()
        self.scale_bar_detection_model.eval()
        
        ## Detect the bounding boxes of each subfigure
        subfigure_info = self.detect_subfigure_boundaries(figure_path)

        ## Detect the subfigure labels on each of the bboxes found
        subfigure_info, concate_img = self.detect_subfigure_labels(figure_path, subfigure_info)
            
        ## Classify the subfigures
        figure_json = self.classify_subfigures(figure_path, subfigure_info, concate_img)

        ## Detect scale bar lines and labels
        figure_json = self.determine_scale(figure_path, figure_json)

        return figure_json
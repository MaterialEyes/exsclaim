import os
import cv2
import json
from numpy.lib import utils
from torch._C import device
import yaml
import glob
import torch
import shutil
import pathlib
import time
import requests
import warnings
import logging

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
from .figures.separator import process
from .figures.scale.process import non_max_suppression_malisiewicz
from .figures.models.network import *
from .tool import ExsclaimTool
from .utilities import boxes, download
from .utilities.logging import Printer
from .figures.scale import ctc
from .figures.models.crnn import CRNN

def convert_to_rgb(image):
    return image.convert("RGB")


class FigureSeparator(ExsclaimTool):
    """ 
    FigureSeparator object.
    Separate subfigure images from full figure image
    using CNN trained on crowdsourced labeled figures
    Parameters:
    None
    """

    ## stores the google drive file IDs of default neural network checkpoints
    ## replace these if you wish to change a model
    model_names_to_googleids = {
        "classifier_model.pt":              "1ZodeH37Nd4ZbA0_1G_MkLKuuiyk7VUXR",
        "object_detection_model.pt":        "1Hh7IPTEc-oTWDGAxI9o0lKrv9MBgP4rm",
        "text_recognition_model.pt":        "1rZaxCPEWKGwvwYYa8jLINpUt20h0jo8y",
        "scale_bar_detection_model.pt":     "1B4_rMbP3a1XguHHX4EnJ6tSlyCCRIiy4",
        "scale_label_recognition_model.pt": "1oGjPG698LdSGvv3FhrLYh_1FhcmYYKpu"
    }

    def __init__(self , search_query):
        self.logger = logging.getLogger(__name__)
        self.initialize_query(search_query)
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
        # This suppresses warning if user has no CUDA device initialized,
        # which is unneccessary as we are explicitly checking. This may not
        # be necessary in the future, described in:
        # https://github.com/pytorch/pytorch/issues/47038
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.cuda = torch.cuda.is_available()
        self.dtype = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        if self.cuda:
            self.logger.info("using cuda") 
            #torch.cuda.set_device(device=args.gpu_id)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        ## Load object detection model
        object_detection_model = YOLOv3(configuration['MODEL'])
        self.object_detection_model = self.load_model_from_checkpoint(
                                            object_detection_model,
                                            "object_detection_model.pt")
        ## Load text recognition model
        text_recognition_model = resnet152()
        self.text_recognition_model = self.load_model_from_checkpoint(
                                            text_recognition_model,
                                            'text_recognition_model.pt')
        ## Load classification model
        master_config_file = model_path + "config/yolov3_default_master.cfg"     
        with open(master_config_file, 'r') as f:
            master_config = yaml.load(f, Loader=yaml.FullLoader)
        classifier_model = YOLOv3img(master_config['MODEL'])
        self.classifier_model = self.load_model_from_checkpoint(
                                            classifier_model,
                                            "classifier_model.pt")
        ## Load scale bar detection model
        # load an object detection model pre-trained on COCO
        scale_bar_detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        input_features = scale_bar_detection_model.roi_heads.box_predictor.cls_score.in_features
        number_classes = 3      # background, scale bar, scale bar label
        scale_bar_detection_model.roi_heads.box_predictor = FastRCNNPredictor(input_features, number_classes)
        self.scale_bar_detection_model = self.load_model_from_checkpoint(
                                            scale_bar_detection_model,
                                            "scale_bar_detection_model.pt")
        ## Load scale label recognition model
        parent_dir = pathlib.Path(__file__).resolve(strict=True).parent
        config_path = (
            parent_dir / "figures" / "config" / "scale_label_reader.json"
        )
        with open(config_path, "r") as f:
            configuration_file = json.load(f)
        configuration = configuration_file["theta"]
        scale_label_recognition_model = CRNN(configuration=configuration)
        self.scale_label_recognition_model = self.load_model_from_checkpoint(
                                                scale_label_recognition_model,
                                                "scale_label_recognition_model.pt")

    def load_model_from_checkpoint(self, model, model_name):
        """ load checkpoint weights into model """
        checkpoints_path = pathlib.Path(__file__).parent / 'figures' / 'checkpoints'
        checkpoint = checkpoints_path / model_name
        model.to(self.device)
        # download the model if isn't already
        if not os.path.isfile(checkpoint):
            os.makedirs(checkpoints_path, exist_ok=True)
            file_id = FigureSeparator.model_names_to_googleids[model_name]
            self.display_info(("Downloading: https://drive.google.com/uc?export=download&id={}"
                   " to {}".format(file_id, checkpoint)))
            download.download_file_from_google_drive(file_id, checkpoint)  
        if self.cuda:
            model.load_state_dict(torch.load(checkpoint))
            model = model.cuda()
        else:
            model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        return model

    def _update_exsclaim(self, exsclaim_dict, figure_name, figure_dict):
        figure_name = figure_name.split("/")[-1]
        for master_image in figure_dict['figure_separator_results'][0]['master_images']:
            exsclaim_dict[figure_name]['master_images'].append(master_image)

        for unassigned in figure_dict['figure_separator_results'][0]['unassigned']:
            exsclaim_dict[figure_name]['unassigned']['master_images'].append(unassigned)
        return exsclaim_dict


    def _appendJSON(self, exsclaim_json, figures_separated):
        """ Commit updates to EXSCLAIM JSON and updates list of separated figures

        Args:
            results_directory (string): Path to results directory
            exsclaim_json (dict): Updated EXSCLAIM JSON
            figures_separated (set): Figures which have already been separated
        """
        with open(self.results_directory / "exsclaim.json", 'w', encoding="utf-8") as f:
            json.dump(exsclaim_json, f, indent=3)
        with open(self.results_directory / "_figures", "a+", encoding="utf-8") as f:
            for figure in figures_separated:
                f.write("%s\n" % figure)


    def run(self, search_query, exsclaim_dict):
        """ Run the models relevant to manipulating article figures
        """
        self.display_info("Running Figure Separator\n")
        os.makedirs(self.results_directory, exist_ok=True)
        self.exsclaim_json = exsclaim_dict
        t0 = time.time()
        ## List figures that have already been separated
        figures_file = self.results_directory / "_figures"
        if os.path.isfile(figures_file):
            with open(figures_file, "r", encoding="utf-8") as f:
                contents = f.readlines()
            figures_separated = {f.strip() for f in contents}
        else:
            figures_separated = set()

        with open(figures_file, "w", encoding="utf-8") as f:
            for figure in figures_separated:
                f.write("%s\n" % str(pathlib.Path(figure).name))
        new_figures_separated = set()

        counter = 1
        figures_path = self.results_directory / 'figures'
        figures = ([figures_path / self.exsclaim_json[figure]["figure_name"] for figure in self.exsclaim_json 
                    if self.exsclaim_json[figure]["figure_name"] not in figures_separated])
        for figure_path in figures:
            self.display_info(">>> ({0} of {1}) ".format(counter,+\
                len(figures))+\
                "Extracting images from: "+ str(figure_path))
            try:
                self.extract_image_objects(figure_path)
                new_figures_separated.add(figure_path.name)
            except Exception as e:
                if self.print:
                    Printer(("<!> ERROR: An exception occurred in"
                    " FigureSeparator on figure: {}".format(figure_path)))
                self.logger.exception(("<!> ERROR: An exception occurred in"
                    " FigureSeparator on figure: {}".format(figure_path)))
            
            # Save to file every N iterations (to accomodate restart scenarios)
            if counter % 100 == 0:
                self._appendJSON(self.exsclaim_json, new_figures_separated)
                new_figures_separated = set()
            counter += 1
        
        t1 = time.time()
        self.display_info(">>> Time Elapsed: {0:.2f} sec ({1} figures)\n".format(t1-t0,int(counter-1)))
        self._appendJSON(self.exsclaim_json, new_figures_separated)
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
            paths+=glob.glob(os.path.join(search_query['results_dir'],'figures','*'+ext))
        return paths,


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
        img, info_img = process.preprocess(img, self.image_size, jitter=0)
        img = np.transpose(img / 255., (2, 0, 1))
        img = np.copy(img)
        img = torch.from_numpy(img).float().unsqueeze(0)
        img = Variable(img.type(self.dtype))

        img_raw = Image.open(figure_path).convert("RGB")
        width, height = img_raw.size
        
        ## Run model on figure
        with torch.no_grad():
            outputs = self.object_detection_model(img.to(self.device))
            outputs = process.postprocess(outputs, dtype=self.dtype, 
                        conf_thre=self.confidence_threshold, nms_thre=self.nms_threshold)

        ## Reformat model outputs to display bounding boxes in our desired format
        ## List of lists where each inner list is [x1, y1, x2, y2, confidence]
        subfigure_info = list()

        if outputs[0] is None:
            self.display_info("No Objects Detected! in {}".format(figure_path))
            return subfigure_info

        for x1, y1, x2, y2, conf, cls_conf, cls_pred in outputs[0]:
            box = process.yolobox2label([y1.data.cpu().numpy(), x1.data.cpu().numpy(), y2.data.cpu().numpy(), x2.data.cpu().numpy()], info_img)
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
            img_patch, _ = process.preprocess(img_patch, 28, jitter=0)
            img_patch = np.transpose(img_patch / 255., (2, 0, 1))
            img_patch = torch.from_numpy(img_patch).type(self.dtype).unsqueeze(0)

            ## Run model on figure
            label_prediction = self.text_recognition_model(img_patch.to(self.device))
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

        img, info_img = process.preprocess(img, self.image_size, jitter=0)
        img = np.transpose(img / 255., (2, 0, 1))
        mask = process.preprocess_mask(mask, self.image_size, info_img)
        mask = np.transpose(mask / 255., (2, 0, 1))
        new_concate_img = np.concatenate((img,mask),axis=0)
        img = torch.from_numpy(new_concate_img).float().unsqueeze(0)
        img = Variable(img.type(self.dtype))

        subfigure_labels_copy = subfigure_labels.copy()
        
        subfigure_padded_labels = np.zeros((80, 5))
        if len(subfigure_labels) > 0:
            subfigure_labels = np.stack(subfigure_labels)
            # convert coco labels to yolo
            subfigure_labels = process.label2yolobox(subfigure_labels, info_img, self.image_size, lrflip=False)
            # make the beginning of subfigure_padded_labels subfigure_labels
            subfigure_padded_labels[:len(subfigure_labels)] = subfigure_labels[:80]
        # conver labels to tensor and add dimension
        subfigure_padded_labels = (torch.from_numpy(subfigure_padded_labels)).unsqueeze(0)
        subfigure_padded_labels = Variable(subfigure_padded_labels.type(self.dtype))
        padded_label_list = [None, subfigure_padded_labels]
        assert subfigure_padded_labels.size()[0] == 1

        # prediction
        with torch.no_grad():
            outputs = self.classifier_model(img.to(self.device), padded_label_list)

        # select the 13x13 grid as feature map
        feature_size = [13,26,52]
        feature_index = 0
        preds = outputs[feature_index]
        preds = preds[0].data.cpu().numpy()

        ## Documentation
        figure_name = figure_path.name
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
    
                x1,y1,x2,y2 = process.yolobox2label([y1,x1,y2,x2], info_img)

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
        resize_transform = T.Compose([
            T.Resize((128, 512)),
            T.Lambda(convert_to_rgb),
            T.ToTensor(),
            T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
        ])       
        
        classes = "0123456789mMcCuUnN .A"
        idx_to_class = classes + "-"
        image = resize_transform(cropped_image)
        image = image.unsqueeze(0)
        # run image on model
        logps = self.scale_label_recognition_model(image.to(self.device))
        probs = torch.exp(logps)
        probs = probs.squeeze(0)
        magnitude, unit, confidence = ctc.run_ctc(probs, classes)
        return magnitude, unit, float(confidence)

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
            x_line, y_line = boxes.find_box_center(line["geometry"])
            best_distance = 1000000
            best_label = None
            for label_index, label in enumerate(scale_bar_labels):
                x_label, y_label = boxes.find_box_center(label["geometry"])
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
        x1, y1, x2, y2 = boxes.convert_labelbox_to_coords(geomtery)
        unassigned_scale_objects = []
        assigned_scale_objects = []
        for scale_object in scale_objects:
            if boxes.is_contained(scale_object["geometry"], geomtery):
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
            master_image["nm_height"] = int(nm_to_pixel * master_image.get("height", y2-y1) * 10) / 10
            master_image["nm_width"] = int(nm_to_pixel * master_image.get("width", x2-x1) * 10) / 10
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
            outputs = self.scale_bar_detection_model([image.to(self.device)])
        # post-process 
        scale_bar_info = []
        for i, box in enumerate(outputs[0]["boxes"]):
            confidence = outputs[0]["scores"][i]
            if confidence > 0.5:
                x1, y1, x2, y2 = box
                label = outputs[0]['labels'][i]
                scale_bar_info.append([x1.data.cpu(), y1.data.cpu(), x2.data.cpu(), y2.data.cpu(), confidence.data.cpu(), label.data.cpu()])
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
            geometry = boxes.convert_coords_to_labelbox([int(x1), int(y1),
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
                magnitude, unit, label_confidence = self.read_scale_bar(
                    scale_bar_label_image)
                # 0 is never correct and -1 is the error value
                if magnitude > 0:
                    length_in_nm = magnitude * convert_to_nm[unit.strip().lower()]
                    label_json = {
                        "geometry" : geometry,
                        "text" : str(magnitude) + " " + unit,
                        "label_confidence" : float(label_confidence),
                        "box_confidence" : float(confidence),
                        "nm" : int(length_in_nm * 100) / 100
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

        # Get full path to figure
        full_figure_path = self.results_directory.parent / figure_path
        
        ## Detect the bounding boxes of each subfigure
        subfigure_info = self.detect_subfigure_boundaries(full_figure_path)

        ## Detect the subfigure labels on each of the bboxes found
        subfigure_info, concate_img = self.detect_subfigure_labels(full_figure_path, subfigure_info)
            
        ## Classify the subfigures
        figure_json = self.classify_subfigures(full_figure_path, subfigure_info, concate_img)

        ## Detect scale bar lines and labels
        figure_json = self.determine_scale(full_figure_path, figure_json)

        return figure_json
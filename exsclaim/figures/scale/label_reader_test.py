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
import torch.nn as nn

from skimage import io
from scipy.special import softmax
from torch.autograd import Variable
from PIL import Image, ImageDraw, ImageFont
import torchvision.models.detection
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import models


class ScaleBarReaderTest():

    #### Utility Functions ####
    def is_number(self, n):
        """ returns true if a string n represents a float """
        try:
            float(n)
        except ValueError:
            return False
        return True

    def is_valid_scale_bar_label(self, text):
        """ returns True if label is of form "[0-9]* [nm|um|mm|A]" """
        if self.is_number(text) or "/" in text:
            return False
        if len(text.split(" ")) != 2:
            return False
        if not self.is_number(text.split(" ")[0]):
            return False
        return True

    def all_divide(self, text):
        return text

    def scale_divide(self, text):
        return text.split()[0]

    def unit_divide(self, text):
        return text.split()[-1]

    def get_classification_model(self, scale_label_recognition_checkpoint, classes, depth, pretrained=True):
        """ """
        ## Load scale bar label reading model
        
        # load an object detection model pre-trained on COCO
        if depth == 18:
            model = models.resnet18(pretrained=pretrained)
        elif depth == 50:
            model = models.resnet50(pretrained=pretrained)
        elif depth == 152:
            model = models.resnet152(pretrained=pretrained)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        cuda = torch.cuda.is_available() and (gpu_id >= 0)
        
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
        model.to(device)
        
        if cuda:
            model.load_state_dict(torch.load(scale_label_recognition_checkpoint))
            model = model.cuda()
        else:
            model.load_state_dict(torch.load(scale_label_recognition_checkpoint, map_location='cpu')["model_state_dict"])
        
        model.eval()
        return model

    def set_up_model(self, checkpoint_path):
        """ set FigureSeparator values """
        print("starting {}".format(checkpoint_path))
        # Break up checkpoint path
        pretrained = True if checkpoint_path.split("/")[-2] == "pretrained" else False
        model_name = checkpoint_path.split("/")[-1].split(".")[0]
        model_type, epoch = model_name.split("-")
        chunks = model_type.split("_")
        dataset_name = "_".join(chunks[:-1])
        depth = int(chunks[-1])

        ## Code to set up scale label reading model(s)
        # id_to_class dictionaries for model outputs
        all = {0: '0.1 A', 1: '0.1 nm', 2: '0.1 um', 3: '0.2 A', 4: '0.2 nm', 5: '0.2 um', 6: '0.3 A', 7: '0.3 nm', 8: '0.3 um', 9: '0.4 A', 10: '0.4 nm', 11: '0.4 um', 12: '0.5 A', 13: '0.5 nm', 14: '0.5 um', 15: '0.6 A', 16: '0.6 nm', 17: '0.6 um', 18: '0.7 A', 19: '0.7 nm', 20: '0.7 um', 21: '0.8 A', 22: '0.8 nm', 23: '0.8 um', 24: '0.9 A', 25: '0.9 nm', 26: '0.9 um', 27: '1 A', 28: '1 nm', 29: '1 um', 30: '10 A', 31: '10 nm', 32: '10 um', 33: '100 A', 34: '100 nm', 35: '100 um', 36: '2 A', 37: '2 nm', 38: '2 um', 39: '2.5 A', 40: '2.5 nm', 41: '2.5 um', 42: '20 A', 43: '20 nm', 44: '20 um', 45: '200 A', 46: '200 nm', 47: '200 um', 48: '25 A', 49: '25 nm', 50: '25 um', 51: '250 A', 52: '250 nm', 53: '250 um', 54: '3 A', 55: '3 nm', 56: '3 um', 57: '30 A', 58: '30 nm', 59: '30 um', 60: '300 A', 61: '300 nm', 62: '300 um', 63: '4 A', 64: '4 nm', 65: '4 um', 66: '40 A', 67: '40 nm', 68: '40 um', 69: '400 A', 70: '400 nm', 71: '400 um', 72: '5 A', 73: '5 nm', 74: '5 um', 75: '50 A', 76: '50 nm', 77: '50 um', 78: '500 A', 79: '500 nm', 80: '500 um', 81: '6 A', 82: '6 nm', 83: '6 um', 84: '60 A', 85: '60 nm', 86: '60 um', 87: '600 A', 88: '600 nm', 89: '600 um', 90: '7 A', 91: '7 nm', 92: '7 um', 93: '70 A', 94: '70 nm', 95: '70 um', 96: '700 A', 97: '700 nm', 98: '700 um', 99: '8 A', 100: '8 nm', 101: '8 um', 102: '80 A', 103: '80 nm', 104: '80 um', 105: '800 A', 106: '800 nm', 107: '800 um', 108: '9 A', 109: '9 nm', 110: '9 um', 111: '90 A', 112: '90 nm', 113: '90 um', 114: '900 A', 115: '900 nm', 116: '900 um'}
        some = {0: '0.1 A', 1: '0.1 nm', 2: '0.1 um', 3: '0.2 A', 4: '0.2 nm', 5: '0.2 um', 6: '0.3 A', 7: '0.3 nm', 8: '0.3 um', 9: '0.4 A', 10: '0.4 nm', 11: '0.4 um', 12: '0.5 A', 13: '0.5 nm', 14: '0.5 um', 15: '1 A', 16: '1 nm', 17: '1 um', 18: '10 A', 19: '10 nm', 20: '10 um', 21: '100 A', 22: '100 nm', 23: '100 um', 24: '2 A', 25: '2 nm', 26: '2 um', 27: '2.5 A', 28: '2.5 nm', 29: '2.5 um', 30: '20 A', 31: '20 nm', 32: '20 um', 33: '200 A', 34: '200 nm', 35: '200 um', 36: '25 A', 37: '25 nm', 38: '25 um', 39: '250 A', 40: '250 nm', 41: '250 um', 42: '3 A', 43: '3 nm', 44: '3 um', 45: '30 A', 46: '30 nm', 47: '30 um', 48: '300 A', 49: '300 nm', 50: '300 um', 51: '4 A', 52: '4 nm', 53: '4 um', 54: '40 A', 55: '40 nm', 56: '40 um', 57: '400 A', 58: '400 nm', 59: '400 um', 60: '5 A', 61: '5 nm', 62: '5 um', 63: '50 A', 64: '50 nm', 65: '50 um', 66: '500 A', 67: '500 nm', 68: '500 um'}        
        scale_some = {0: '0.1', 1: '0.2', 2: '0.3', 3: '0.4', 4: '0.5', 5: '1', 6: '10', 7: '100', 8: '2', 9: '2.5', 10: '20', 11: '200', 12: '25', 13: '250', 14: '3', 15: '30', 16: '300', 17: '4', 18: '40', 19: '400', 20: '5', 21: '50', 22: '500'}
        scale_all = {0: '0.1', 1: '0.2', 2: '0.3', 3: '0.4', 4: '0.5', 5: '0.6', 6: '0.7', 7: '0.8', 8: '0.9', 9: '1', 10: '10', 11: '100', 12: '2', 13: '2.5', 14: '20', 15: '200', 16: '25', 17: '250', 18: '3', 19: '30', 20: '300', 21: '4', 22: '40', 23: '400', 24: '5', 25: '50', 26: '500', 27: '6', 28: '60', 29: '600', 30: '7', 31: '70', 32: '700', 33: '8', 34: '80', 35: '800', 36: '9', 37: '90', 38: '900'}
        unit_data = {0: 'A', 1: 'mm', 2: 'nm', 3: 'um'}
        dataset_to_dict = {"all" : all, "some": some, "scale_all": scale_all, "scale_some": scale_some, "unit_data": unit_data}
        self.idx_to_class = dataset_to_dict[dataset_name]
        self.class_to_idx = { v : k for k, v in self.idx_to_class.items()}

        self.model = self.get_classification_model(checkpoint_path, len(self.idx_to_class), depth, pretrained)

        get_actual_text_funcs = {'all': self.all_divide, 'some': self.all_divide, 'scale_all': self.scale_divide, "scale_some": self.scale_divide, "unit_data": self.unit_divide}
        self.get_actual_text = get_actual_text_funcs[dataset_name]
        
    def run_model(self, cropped_image):
        label_reader_transforms = T.Compose([T.Resize((224, 224)),
                                T.ToTensor(),
                                T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),])
        cropped_image = label_reader_transforms(cropped_image)
        cropped_image = cropped_image.unsqueeze(0)

        ## Find predicted scale unit
        with torch.no_grad():
            log_probabilities = self.model(cropped_image)
        probabilities = torch.exp(log_probabilities)
        probabilities = list(probabilities.numpy()[0])
        predicted_idx = probabilities.index(max(probabilities))
        text = self.idx_to_class[predicted_idx]
        return predicted_idx, text, probabilities[predicted_idx]


    def test_single_model(self, checkpoint_path):
        """ Tests the accuracy and validity of reading scale bar labels """
        self.set_up_model(checkpoint_path)
        scale_label_data = pathlib.Path(__file__).resolve(strict=True)
        scale_label_data = scale_label_data.parent.parent.parent  / 'tests' / 'data' / 'scale_label_dataset'
        correct = 0
        incorrect = 0
        predicted_classes = []
        predicted_idxs = []
        actual_classes = []
        actual_idxs = []
        confidences = []
        skipped = 0
        for label_dir in os.listdir(scale_label_data):
            label = str(label_dir)
            actual_text = self.get_actual_text(label)
            if actual_text in self.class_to_idx:
                actual_idx = self.class_to_idx[actual_text]
            else:
                skipped += 1
                actual_idx = -1
            for image_file in os.listdir(scale_label_data / label):
                scale_label_image = Image.open(scale_label_data / label / image_file).convert("RGB")
                predicted_idx, predicted_text, confidence = self.run_model(scale_label_image)
                predicted_idxs.append(predicted_idx)
                predicted_classes.append(predicted_text)
                actual_classes.append(actual_text)
                actual_idxs.append(actual_idx)
                confidences.append(float(confidence))
                if predicted_idx == actual_idx:
                    correct += 1
                else:
                    incorrect += 1  

        accuracy = correct / float(correct + incorrect + 0.0000000000000001)
        return {"predicted_class": predicted_classes, 
                "predicted_idx": predicted_idxs, 
                "actual_class": actual_classes,
                "actual_idx": actual_idxs,
                "confidence": confidences, 
                "accuracy": accuracy}

    def test_many_models(self, checkpoint_dir):
        """ tests accuracy of diffferent scale label reading methods """
        results_dict = {}
        for filename in os.listdir(checkpoint_dir):
            model_name, filetype = str(filename).split(".")
            if model_name not in {"some_18-120", "all_18-148", "scale_all_18-124", "scale_some_18-127", "unit_data_18-250"}:
                continue
            if filetype == "pt":
                checkpoint_path = checkpoint_dir / filename
                single_result_dict = self.test_single_model(str(checkpoint_path))
                results_dict[filename] = single_result_dict
                with open("results_breif.txt", "w") as f:
                    json.dump(results_dict, f)



if __name__ == '__main__':
    checkpoint_dir = pathlib.Path(__file__).parent / 'checkpoints' / 'pretrained'
    test = ScaleBarReaderTest()
    test.test_many_models(checkpoint_dir)

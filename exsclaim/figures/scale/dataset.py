import json
import torch
import numpy as np
import os
from PIL import Image
import random
from ...utilities.boxes import convert_labelbox_to_coords

class ScaleLabelDataset():
    """ Dataset used to train CRNN to read scale bar labels """
    def make_encoding(self, label):
        max_length = 8
        char_to_int = {
            "0":    0,
            "1":    1,
            "2":    2,
            "3":    3,
            "4":    4,
            "5":    5,
            "6":    6,
            "7":    7,
            "8":    8,
            "9":    9,
            "m":    10,
            "M":    11,
            "c":    12,
            "C":    13,
            "u":    14,
            "U":    15,
            "n":    16,
            "N":    17,
            " ":    18,
            ".":    19,
            "A":    20,
            "empty": 21
        }
        target = torch.zeros(max_length)
        for i in range(max_length):
            try:
                character = label[i]
                number = char_to_int[character]
            except:
                number = 21
            target[i] = number
        return target            

    def __init__(self, root, transforms, test=True):
        self.root = root
        self.transforms = transforms
        if test:
            scale_bar_dataset = os.path.join(root, "test")
        else:
            scale_bar_dataset = os.path.join(root, "train")

        self.image_paths = []
        for label in os.listdir(scale_bar_dataset):
            label_folder = os.path.join(scale_bar_dataset, label)
            for image in os.listdir(label_folder):
                image_path = os.path.join(label_folder, image)
                self.image_paths.append(image_path)
  
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image_transformed = self.transforms(image)
        image.close()
        label = image_path.split("/")[-2]
        target = self.make_encoding(label)
        return image_transformed, target

    def __len__(self):
        return len(self.image_paths)

class ScaleBarDataset():
    """ Dataset used to train Faster-RCNN to detect scale labels and lines """
    def __init__(self, root, transforms, test=True, size=None):
        ## initiates a dataset from a json
        self.root = root
        self.transforms = transforms
        if test:
            scale_bar_dataset = os.path.join(root, "scale_bars_dataset_test.json")
        else:
            scale_bar_dataset = os.path.join(root, "scale_bars_dataset_train.json")

        self.test = test
        with open(scale_bar_dataset, "r") as f:
            self.data = json.load(f)
        all_figures = os.path.join(root, "images", "labeled_data")
        self.images = [figure for figure in self.data 
                       if os.path.isfile(os.path.join(all_figures,
                                                      figure))]
        if size != None:
            self.images = random.sample(self.images, size)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.root,"images", "labeled_data", self.images[idx])
        with Image.open(image_path).convert("RGB") as image:
            image_name = self.images[idx]

            boxes = []
            labels = []
            for scale_bar in self.data[image_name].setdefault("scale_bars", []):
                boxes.append(convert_labelbox_to_coords(scale_bar["geometry"]))
                labels.append(1)
            for scale_label in self.data[image_name].setdefault("scale_labels", []):
                boxes.append(convert_labelbox_to_coords(scale_label["geometry"]))
                labels.append(2)
            
            num_objs = len(boxes)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd

            if self.transforms is not None:
                new_image = self.transforms(image)

        return new_image, target
        
    def __len__(self):
        return len(self.images)
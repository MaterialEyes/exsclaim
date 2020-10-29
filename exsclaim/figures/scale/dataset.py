import json
import torch
import numpy as np
import os
from PIL import Image
from utils import convert_box_format

class ScaleBarDataset():
    def __init__(self, root, transforms):
        ## initiates a dataset from a json
        self.root = root
        self.transforms = transforms

        with open("scale_bars_dataset.json", "r") as f:
            self.data = json.load(f)

        self.images = [figure for figure in self.data if os.path.isfile(os.path.join("all-figures", figure))]
        print(len(self.images))

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, "all-figures", self.images[idx])
        image = Image.open(image_path).convert("RGB")
        image_name = self.images[idx]

        boxes = []
        labels = []
        for scale_bar in self.data[image_name].setdefault("scale_bars", []):
            boxes.append(convert_box_format(scale_bar["geometry"]))
            labels.append(1)
        for scale_label in self.data[image_name].setdefault("scale_labels", []):
            boxes.append(convert_box_format(scale_label["geometry"]))
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
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.images)
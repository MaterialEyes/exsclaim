import os
import numpy as np

import torch
from torch.utils.data import Dataset
import cv2
# from pycocotools.coco import COCO

from ..utils.utils import *
import glob
import xml.etree.ElementTree as ET


class FigSepDataset(Dataset):
    """
    COCO dataset class.
    """
    def __init__(self, model_type, data_dir='FigSep/JPEGImages', anno_dir='FigSep/Annotations/',
                 name='SepFig_100', img_size=416,
                 augmentation=None, min_size=1, debug=False):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            model_type (str): model name specified in config file
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            min_size (int): bounding boxes smaller than this are ignored
            debug (bool): if True, only one data id is selected from the dataset
        """
        self.data_dir = data_dir
        self.anno_dir = anno_dir
#         self.json_file = json_file
        self.model_type = model_type
#         self.coco = COCO(self.data_dir+'annotations/'+self.json_file)
#         self.ids = self.coco.getImgIds()
#         num_annotations = glob.glob("FigSep/Annotations/*.xml")
        num_annotations = glob.glob(os.path.join(self.anno_dir,"*.xml"))
        self.ids = np.arange(len(num_annotations))
        self.anno_list = num_annotations
#         if debug:
#             self.ids = self.ids[1:2]
#             print("debug mode...", self.ids)
#         self.class_ids = sorted(self.coco.getCatIds())
        self.class_ids = np.arange(2) #single figure or not

        self.name = name
        self.max_labels = 50
        self.img_size = img_size
        self.min_size = min_size
        self.lrflip = augmentation['LRFLIP']
        self.jitter = augmentation['JITTER']
        self.random_placing = augmentation['RANDOM_PLACING']
        self.hue = augmentation['HUE']
        self.saturation = augmentation['SATURATION']
        self.exposure = augmentation['EXPOSURE']
        self.random_distort = augmentation['RANDOM_DISTORT']


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up \
        and pre-processed.
        Args:
            index (int): data index
        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data. \
                The shape is :math:`[self.max_labels, 5]`. \
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            id_ (int): same as the input index. Used for evaluation.
        """
        id_ = self.ids[index]
        
#         in_file = open("./FigSep/Annotations/fig{}img_.xml".format(id_))
#         in_file = open(os.path.join(self.anno_dir,"fig{}img_.xml".format(id_)))
        in_file = open(self.anno_list[id_])
        root = ET.parse(in_file).getroot()
        img_file_name = root.find("filename").text
        img_size = root.find("size")
        img_size = [int(img_size.find("width").text),int(img_size.find("height").text),int(img_size.find("depth").text)]
        bboxes = []
        for obj in root.iter('object'):
            current = list()

            xmlbox = obj.find('bndbox')
            xn = (float(xmlbox.find('xmin').text))
            xx = (float(xmlbox.find('xmax').text))
            yn = (float(xmlbox.find('ymin').text))
            yx = (float(xmlbox.find('ymax').text))
            current = [xn,yn,xx-xn,yx-yn]
            bboxes += [current]

#         anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=None)
#         annotations = self.coco.loadAnns(anno_ids)

        lrflip = False
        if np.random.rand() > 0.5 and self.lrflip == True:
            lrflip = True

        # load image and preprocess
        img_file = os.path.join(self.data_dir,img_file_name)
        img = cv2.imread(img_file)

#         if self.json_file == 'instances_val5k.json' and img is None:
#             img_file = os.path.join(self.data_dir, 'train2017',
#                                     '{:012}'.format(id_) + '.jpg')
#             img = cv2.imread(img_file)
#         assert img is not None

        img, info_img = preprocess(img, self.img_size, jitter=self.jitter,
                                   random_placing=self.random_placing)

        if self.random_distort:
            img = random_distort(img, self.hue, self.saturation, self.exposure)

        img = np.transpose(img / 255., (2, 0, 1))

        if lrflip:
            img = np.flip(img, axis=2).copy()

        # load labels
        labels = []
        for bbox in bboxes:
            if bbox[2] > self.min_size and bbox[3] > self.min_size:
                labels.append([])
                labels[-1].append(1)
                labels[-1].extend(bbox)

        padded_labels = np.zeros((self.max_labels, 5))
        if len(labels) > 0:
            labels = np.stack(labels)
            if 'YOLO' in self.model_type:
                labels = label2yolobox(labels, info_img, self.img_size, lrflip)
            padded_labels[range(len(labels))[:self.max_labels]
                          ] = labels[:self.max_labels]
        padded_labels = torch.from_numpy(padded_labels)

        return img, padded_labels, info_img, id_

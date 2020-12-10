import unittest
import json
import pathlib
import os
from PIL import Image
import random
import torchvision.transforms as T
import torch
import numpy as np

from .. import figure
from ..figures.scale.dataset import ScaleBarDataset
from ..figures.scale.engine import evaluate
from ..figures.scale.utils import collate_fn

class TestScaleDetection(unittest.TestCase):

    def setUp(self):
        """ Instantiates a test search query and FigureSeparator to test """
        nature_json = pathlib.Path(__file__).parent / 'data' / 'nature_test.json'
        with open(nature_json, "r") as f:
            query = json.load(f)

        self.query = query
        self.figure_separator = figure.FigureSeparator(query)
        self.current_directory = pathlib.Path(__file__).resolve(strict=True).parent
                
    def test_scale_object_detection_accuracy(self):
        """ Tests the accuracy and validity of scale bar object detection """
        test_image_directory = self.current_directory / 'data'
        dataset = ScaleBarDataset(test_image_directory, T.ToTensor(), True, 5)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                                 collate_fn=collate_fn, num_workers=0,
                                                 shuffle=False)
        coco_eval = evaluate(self.figure_separator.scale_bar_detection_model,
                             dataloader, self.figure_separator.device)

    def test_scale_label_reading_accuracy(self):
        """ Tests the accuracy of reading scale bar labels """
        # set constants
        scale_label_data = pathlib.Path(__file__).parent / 'data' / 'scale_label_dataset'
        min_low_test_images = 100
        min_high_test_images = 50
        max_test_images = 250
        high_confidence_threshold = 0.9
        expected_high_confidence_correct = 0.8
        low_confidence_threshold = 0.6
        expected_low_confidence_correct = 0.5
        
        # keep track of accuracy
        high_correct = 0
        high_total = 0
        low_correct = 0
        low_total = 0
        total = 0

        # test on random images
        while ((high_total < min_high_test_images or 
               low_total < min_low_test_images) and 
               total < max_test_images):
            total += 1
            # randomly pick label
            label_dir = random.choice(os.listdir(scale_label_data))
            label = str(label_dir)
            # randomly pick image with selected label
            image_file = random.choice(os.listdir(scale_label_data / label))
            scale_label_image = Image.open(scale_label_data / label / image_file).convert("RGB")
            result, confidence = self.figure_separator.read_scale_bar(scale_label_image)
            
            if confidence < low_confidence_threshold:
                continue
            # if confidence above lower threshold
            low_total += 1
            if result == label:
                low_correct += 1
            # if confidence above higher threshold
            if confidence >= high_confidence_threshold:
                high_total += 1
                if result == label:
                    high_correct += 1
        # accuracy tests
        self.assertGreaterEqual(low_total, min_low_test_images, 
            ("Only {} images had a confidence of greater than {}".format(
                low_total, low_confidence_threshold)))
        self.assertGreaterEqual(high_total, min_high_test_images, 
            ("Only {} images had a confidence of greater than {}".format(
                high_total, high_confidence_threshold)))
        high_accuracy = high_correct / float(high_total)
        low_accuracy = low_correct / float(low_total)
        self.assertGreater(low_accuracy, expected_low_confidence_correct,
            ("Scale label reading had poor accuracy with {} % correct, "
             "less than desired {}% for predictions with confidence score "
             "of {}".format(low_accuracy, expected_low_confidence_correct,
             low_confidence_threshold)))
        self.assertGreater(high_accuracy, expected_high_confidence_correct,
            ("Scale label reading had poor accuracy with {} % correct, "
             "less than desired {}% for predictions with confidence score "
             "of {}".format(high_accuracy, expected_high_confidence_correct,
             high_confidence_threshold)))


class TestSubfigureDetection(unittest.TestCase):

    def setUp(self):
        """ Instantiates a test search query and FigureSeparator to test """
        nature_json = pathlib.Path(__file__).parent / 'data' / 'nature_test.json'
        with open(nature_json, "r") as f:
            query = json.load(f)
        self.query = query
        self.figure_separator = figure.FigureSeparator(query)

    def test_subfigure_detection_accuracy(self):
        """ Tests the accuracy and validity of identifying subfigures """
        pass

    def test_subfigure_label_reading_accuracy(self):
        """ Tests the accuracy and validity of reading subfigure labels """
        pass

    def test_subfigure_classification_accuracy(self):
        """ Tests the accuracy and validity of classifying subfigures """
        pass


if __name__ == "__main__":
    unittest.main()
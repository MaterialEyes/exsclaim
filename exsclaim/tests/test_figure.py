import unittest
import json
import pathlib
import os
from PIL import Image
import random
import shutil
import torchvision.transforms as T
import torch
import numpy as np

from exsclaim import figure
from exsclaim.figures.scale.dataset import ScaleBarDataset

class TestScaleDetection(unittest.TestCase):

    def setUp(self):
        """ Instantiates a test search query and FigureSeparator to test """
        self.current_file = pathlib.Path(__file__).resolve(strict=True)
        self.current_directory = self.current_file.parent
        self.data = self.current_directory / 'data'
        nature_json = self.data / 'nature_test.json'
        with open(nature_json, "r") as f:
            self.query = json.load(f)
        self.figure_separator = figure.FigureSeparator(self.query)

    def tearDown(self):
        pass
    
    def test_scale_object_detection_validity(self):
        """ Tests the accuracy and validity of scale bar object detection """
        test_image_directory = self.data / 'images' / 'scale_bar_test_images'
        for image_name in os.listdir(test_image_directory):
            with self.subTest(test_name = image_name):
                image_path = test_image_directory / image_name
                image = Image.open(image_path).convert("RGB")
                image = T.ToTensor()(image)
                predicted_scale_bar_info = (
                    self.figure_separator.detect_scale_objects(image))
                self.assertIsInstance(predicted_scale_bar_info,
                    (list, np.ndarray),
                    ("detect_scale_objects() should return a list or "
                     "numpy.ndarray. Returned type {}".format(
                         type(predicted_scale_bar_info))))
                for scale_object in predicted_scale_bar_info:
                    self.assertIsInstance(scale_object, (list, np.ndarray))
                    self.assertEqual(len(scale_object), 6)
                    self.assertIn(scale_object[5], [1, 2])
            
    def test_scale_label_reading(self):
        """ Tests the validity of FigureSeparator.read_scale_bar() results """
        test_images = self.data / 'images' / 'scale_label_test_images'
        # test on each image as a separate subtest case
        for image_name in os.listdir(test_images):
            with self.subTest(test_name = image_name):
                scale_label_image = Image.open(
                    test_images / image_name).convert("RGB")
                magnitude, unit, confidence = (
                    self.figure_separator.read_scale_bar(scale_label_image))
                # Ensure invariants are held
                self.assertIsInstance(magnitude, float,
                    ("read_scale_bar() should return a float, str and a float."
                     " Value one returned type {}".format(type(magnitude))))
                self.assertIsInstance(confidence, (float, np.float32),
                    ("read_scale_bar() should return a float, str, and a float."
                     " Value two returned type {}".format(type(confidence))))
                     

class TestSubfigureDetection(unittest.TestCase):

    def setUp(self):
        """ Instantiates a test search query and FigureSeparator to test """
        nature_json = pathlib.Path(__file__).parent / 'data' / 'nature_test.json'
        with open(nature_json, "r") as f:
            query = json.load(f)
        self.query = query
        self.figure_separator = figure.FigureSeparator(query)

    def test_subfigure_detection(self):
        """ Tests the accuracy and validity of identifying subfigures """
        pass

    def test_subfigure_label_reading(self):
        """ Tests the accuracy and validity of reading subfigure labels """
        pass

    def test_subfigure_classification(self):
        """ Tests the accuracy and validity of classifying subfigures """
        pass


if __name__ == "__main__":
    unittest.main()

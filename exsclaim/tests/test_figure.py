import unittest
import json
import pathlib
import os
from PIL import Image

from .. import figure

class TestScaleDetection(unittest.TestCase):

    def setUp(self):
        """ Instantiates a test search query and JournalFamily to test """
        nature_json = pathlib.Path(__file__).parent / 'data' / 'nature_test.json'
        with open(nature_json, "r") as f:
            query = json.load(f)
        self.query = query
        self.figure_separator = figure.FigureSeparator(query)
    
    def is_number(self, n):
        """ returns true if a string n represents a float """
        try:
            float(n)
        except ValueError:
            return False
        return True

    def is_valid_scale_bar_label(self, text):
        """ returns """
        if self.is_number(text) or "/" in text:
            return False
        if len(text.split(" ")) != 2:
            return False
        if not self.is_number(text.split(" ")[0]):
            return False
        return True

    def test_scale_object_detection(self):
        """ Tests the accuracy and validity of scale bar object detection """
        pass

    def test_scale_label_reading(self):
        """ Tests the accuracy and validity of reading scale bar labels """
        scale_label_data = pathlib.Path(__file__).parent / 'data' / 'scale_label_dataset'
        correct = 0
        incorrect = 0
        for label_dir in os.listdir(scale_label_data):
            label = str(label_dir)
            for image_file in os.listdir(scale_label_data / label):
                scale_label_image = Image.open(scale_label_data / label / image_file).convert("RGB")
                #result = self.figure_separator.read_scale_bar_full(scale_label_image)
                result = self.figure_separator.read_scale_bar_parts(scale_label_image)
                print(result, label)
                self.assertIsInstance(result, str)
                self.assertTrue(self.is_valid_scale_bar_label(result))
                if result == label:
                    correct += 1
                else:
                    incorrect += 1
        accuracy = correct / float(correct + incorrect)
        self.assertGreater(accuracy, 0.7, ("Scale label reading had poor "
                "accuracy with {} correct labels and {} incorrect".format(correct, incorrect)))

                

class TestSubfigureDetection(unittest.TestCase):

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
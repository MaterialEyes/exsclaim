import unittest
import json
import pathlib

from .. import figure

class TestNature(unittest.TestCase):

    def setUp(self):
        """ Instantiates a test search query and JournalFamily to test """
        nature_json = pathlib.Path(__file__).parent / 'nature-test.json'
        with open(nature_json, "r") as f:
            query = json.load(f)
        self.query = query
        self.jfamily = journal.Nature(query)

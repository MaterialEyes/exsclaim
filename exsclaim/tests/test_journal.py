import unittest
import json
import pathlib

import responses
from bs4 import BeautifulSoup
import bs4

from exsclaim import journal

class TestNature(unittest.TestCase):

    def setUp(self):
        """ Instantiates a test search query and JournalFamily to test """
        nature_json = pathlib.Path(__file__).parent / 'data' / 'nature_test.json'
        with open(nature_json, "r") as f:
            query = json.load(f)
        self.query = query
        self.jfamily = journal.Nature(query)


    def test_get_page_info(self):
        """ tests that get_page_info finds correct params in nature article """
        test_html = pathlib.Path(__file__).parent / 'data' / 'nature_search.html'
        with open(test_html, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), 'lxml')
        ## sample soup
        expected_info = (1, 1, 12)
        actual_info = self.jfamily.get_page_info(soup)
        self.assertEqual(expected_info, actual_info)

    def test_get_search_query_urls(self):
        """ tests get_search_query_urls returns appropriate urls

        This does not test the existence of specific url extensions as 
        the order and number of urls is implementation dependent.
        This just ensures we have a list of strings, where the strings
        are appropriate url extensions
        """
        # set up sortby to test that urls extensions contain sorting param
        sortby = self.query['sortby']
        if sortby == 'relevant':
            sbext = self.jfamily.relevant
        elif sortby == 'recent':
            sbext = self.jfamily.recent

        search_extensions = self.jfamily.get_search_query_urls()
        # assure the result is a list, and each item is a string
        self.assertIsInstance(search_extensions, list)
        for url in search_extensions:
            self.assertIsInstance(url, str)
            # assure that each url fits certain constraints
            self.assertIn(self.jfamily.path, url)
            self.assertIn(self.jfamily.pre_sb, url)
            self.assertIn(self.jfamily.post_sb, url)
            self.assertIn(sbext, url)
            domain_length = len(self.jfamily.domain)
            self.assertEqual(url[0:domain_length], self.jfamily.domain)


    def test_get_article_extensions(self):
        article_paths = self.jfamily.get_article_extensions()
        self.assertIsInstance(article_paths, list)
        for article_path in article_paths:
            self.assertIsInstance(article_path, str)


    @responses.activate
    def test_get_soup_from_request(self):
        # set up expected soup from request
        test_html_file = pathlib.Path(__file__).parent / 'data' / 'nature_articles' / 'ncomms1737.html'
        with open(test_html_file, "r", encoding="utf-8") as f:
            test_html = f.read()
        expected_soup = BeautifulSoup(test_html, 'lxml')
        
        # set up and execute mock execution
        mock_url = 'http://www.test_exsclaim.com/article/test_article'
        responses.add(responses.GET, mock_url, body=test_html)
        result_soup = self.jfamily.get_soup_from_request(mock_url)
        self.assertEqual(expected_soup, result_soup)
        

    @responses.activate
    def test_get_figure_list(self):
        """ tests that get_figure_list gets the correct figures from test article """
        # set up expected soup from request
        test_html_file = pathlib.Path(__file__).parent / 'data' / 'nature_articles' / 'ncomms1737.html'
        with open(test_html_file, "r", encoding="utf-8") as f:
            test_html = f.read()
        expected_soup = BeautifulSoup(test_html, 'lxml')

        # set up and execute mock execution
        mock_url = 'http://www.test_exsclaim.com/article/test_article'
        responses.add(responses.GET, mock_url, body=test_html)
        figures = self.jfamily.get_figure_list(mock_url)

        self.assertIsInstance(figures, list)
        for figure in figures:
            self.assertIsInstance(figure, bs4.element.Tag)
            

if __name__ == '__main__':
    unittest.main()
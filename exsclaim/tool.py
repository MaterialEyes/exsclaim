# Copyright 2019 MaterialEyes
# (see accompanying license files for details).

"""Definition of the ExsclaimTool classes.
This module defines the central objects in the EXSCLAIM! 
package. All the model classes are independent of each 
other, but they expose the same interface, so they are 
interchangeable.
"""
import os
import json
import glob
import copy
import time
import logging
import pathlib

from . import journal
from . import caption
from .utilities.logging import Printer
from .utilities import paths

from abc import ABC, abstractmethod

class ExsclaimTool(ABC):
    def __init__(self, search_query):
        self.logger = logging.getLogger(__name__)
        self.initialize_query(search_query)

    def initialize_query(self, search_query):
        """ initializes search query as instance attribute

        Args:
            search_query (a dict or path to dict): The Query JSON
        """
        try:
            with open(search_query) as f:
                # Load query file to dict
                self.search_query = json.load(f)
        except Exception as e:
            self.logger.debug(("Search Query path {} not found. Using it as"
                " dictionary instead"))
            self.search_query = search_query
        # Set up file structure
        base_results_dir = paths.initialize_results_dir(
            self.search_query.get("results_dirs", None)
        )
        self.results_directory = (
            base_results_dir / self.search_query["name"]
        )
        # set up logging / printing
        self.print = "print" in self.search_query.get("logging", [])

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def _update_exsclaim(self):
        pass

    @abstractmethod
    def run(self):
        pass

    def display_info(self, info):
        """ Display information to the user as the specified in the query

        Args:
            info (str): A string to display (either to stdout, a log file)
        """
        if self.print: 
            Printer(info)
        self.logger.info(info)

class JournalScraper(ExsclaimTool):
    """ 
    JournalScraper object.
    Extract scientific figures from journal articles by passing  
    a json-style search query to the run method
    Parameters: 
    None
    """
    journals = {
        'acs':      journal.ACS,
        'nature':   journal.Nature,
        'rsc':      journal.RSC,
    }

    def __init__(self, search_query):
        self.logger = logging.getLogger(__name__ + ".JournalScraper")
        self.initialize_query(search_query)
        self.new_articles_visited = set()

        ## Check if any articles have already been scraped by checking
        ##   results_dir/_articles
        articles_visited = {}
        articles_file = self.results_directory / "_articles"
        if os.path.isfile(articles_file):
            with open(articles_file,'r') as f:
                contents = f.readlines()
            articles_visited ={a.strip() for a in contents}
        self.articles_visited = articles_visited

    def _load_model(self):
        pass

    def _update_exsclaim(self, exsclaim_dict, article_dict):
        """ Update the exsclaim_dict with article_dict contents

        Args:
            exsclaim_dict (dict): An EXSCLAIM JSON
            article_dict (dict): 
        Returns:
            exsclaim_dict (dict): EXSCLAIM JSON with article_dict
                contents added. 
        """
        exsclaim_dict.update(article_dict)
        return exsclaim_dict

    def _appendJSON(self, filename, exsclaim_json):
        """ Commit updates to exsclaim json and update list of scraped articles

        Args:
            filename (string): File in which to store the updated EXSCLAIM JSON
            exsclaim_json (dict): Updated EXSCLAIM JSON
        """
        with open(filename,'w') as f: 
            json.dump(exsclaim_json, f, indent=3)
        articles_file = self.results_directory / "_articles"
        with open(articles_file, "a") as f:
            for article in self.new_articles_visited:
                f.write('%s\n' % article.split("/")[-1])

    def _get_articles(self, j_instance):
        """ Get a list of articles that have not already been scraped

        Args:
            j_instance (journal.JournalFamily): An instance of a
                JounralFamily search. 
        """
        articles = j_instance.get_article_extensions(self.articles_visited)

        return articles

    def run(self, search_query, exsclaim_json={}):
        """ Run the JournalScraper to find relevant article figures

        Args:
            search_query (dict): A Search Query JSON to guide search
            exsclaim_json (dict): An EXSCLAIM JSON to store results in
        Returns:
            exsclaim_json (dict): Updated with results of search
        """
        self.display_info("Running Journal Scraper\n")
        ## Checks that user inputted journal family has been defined and
        ## grabs instantiates an instance of the journal family object
        journal_family = search_query['journal_family']
        if journal_family not in self.journals:
            raise NameError('journal family {0} is not defined'.format(journal_family))
        j_instance = self.journals[journal_family](search_query)

        os.makedirs(self.results_directory, exist_ok=True)
        t0 = time.time()
        counter = 1
        articles = self._get_articles(j_instance)
        ## Extract figures, captions, and metadata from each article
        for article in articles:
            self.display_info(">>> ({0} of {1}) Extracting figures from: ".format(counter, len(articles))+\
                article.split("/")[-1])
            try:
                request = j_instance.get_domain_name() + article
                article_dict = j_instance.get_article_figures(request)
                exsclaim_json = self._update_exsclaim(exsclaim_json, article_dict)
                self.new_articles_visited.add(article)
            except Exception as e:
                exception_string = ("<!> ERROR: An exception occurred in"
                    " JournalScraper on article: {}".format(article))
                if self.print:
                    Printer(exception_string + "\n")
                self.logger.exception(exception_string)
            
            # Save to file every N iterations (to accomodate restart scenarios)
            if counter%1000 == 0:
                self._appendJSON(self.results_directory / "exsclaim.json", exsclaim_json)
            counter += 1

        t1 = time.time()
        self.display_info(">>> Time Elapsed: {0:.2f} sec ({1} articles)\n".format(t1-t0,int(counter-1)))
        self._appendJSON(self.results_directory / "exsclaim.json", exsclaim_json)
        return exsclaim_json


class CaptionDistributor(ExsclaimTool):
    """ 
    CaptionDistributor object.
    Distribute subfigure caption chunks from full figure captions 
    in an exsclaim_dict using custom caption nlp tools
    Parameters:
    model_path: str 
        Absolute path to caption nlp model 
    """
    def __init__(self , search_query={}):
        super().__init__(search_query)
        self.logger = logging.getLogger(__name__ + ".CaptionDistributor")
        self.model_path = ""

    def _load_model(self):
        if "" in self.model_path:
            self.model_path = os.path.dirname(__file__)+'/captions/models/'
        return caption.load_models(self.model_path)

    def _update_exsclaim(self, exsclaim_dict, figure_name, delimiter, caption_dict):
        exsclaim_dict[figure_name]["caption_delimiter"] = delimiter
        for label in caption_dict:
            master_image = {"label": label, "description": caption_dict[label]['description'], "keywords": caption_dict[label]['keywords'], "general": caption_dict[label]['general']}
            exsclaim_dict[figure_name]['unassigned']['captions'].append(master_image)
        return exsclaim_dict

    def _appendJSON(self, exsclaim_json, captions_distributed):
        """ Commit updates to EXSCLAIM JSON and updates list of ed figures

        Args:
            results_directory (string): Path to results directory
            exsclaim_json (dict): Updated EXSCLAIM JSON
            figures_separated (set): Figures which have already been separated
        """
        with open(self.results_directory / "exsclaim.json",'w') as f: 
            json.dump(exsclaim_json, f, indent=3)
        with open(self.results_directory / "_captions", "a+") as f:
            for figure in captions_distributed:
                f.write("%s\n" % figure.split("/")[-1])


    def run(self, search_query, exsclaim_json):
        """ Run the CaptionDistributor to distribute subfigure captions

        Args:
            search_query (dict): A Search Query JSON to guide search
            exsclaim_json (dict): An EXSCLAIM JSON to store results in
        Returns:
            exsclaim_json (dict): Updated with results of search
        """
        self.display_info("Running Caption Distributor\n")
        os.makedirs(self.results_directory, exist_ok=True)
        t0 = time.time()
        model = self._load_model()

        ## List captions that have already been distributed
        captions_file = self.results_directory / "_captions"
        if os.path.isfile(captions_file):
            with open(captions_file, "r") as f:
                contents = f.readlines()
            captions_distributed = {f.strip() for f in contents}
        else:
            captions_distributed = set()
        new_captions_distributed = set()

        figures = ([exsclaim_json[figure]["figure_name"] for figure in exsclaim_json 
                    if exsclaim_json[figure]["figure_name"] not in captions_distributed])
        counter = 1
        for figure_name in figures:
            self.display_info(">>> ({0} of {1}) ".format(counter,+\
                len(figures))+\
                "Parsing captions from: "+figure_name)
            try:
                caption_text  = exsclaim_json[figure_name]['full_caption']
                delimiter = caption.find_subfigure_delimiter(model, caption_text)
                caption_dict  = caption.associate_caption_text(model, caption_text, search_query['query'])
                exsclaim_json = self._update_exsclaim(exsclaim_json, figure_name, delimiter, caption_dict) 
                new_captions_distributed.add(figure_name)
            except Exception as e:
                if self.print:
                    Printer(("<!> ERROR: An exception occurred in"
                        " CaptionDistributor on figue: {}".format(figure_name)))
                self.logger.exception(("<!> ERROR: An exception occurred in"
                    " CaptionDistributor on figue: {}".format(figure_name)))        
            # Save to file every N iterations (to accomodate restart scenarios)
            if counter%1000 == 0:
                self._appendJSON(exsclaim_json, new_captions_distributed)
                new_captions_distributed = set()
            counter += 1

        t1 = time.time()
        self.display_info(">>> Time Elapsed: {0:.2f} sec ({1} captions)\n".format(t1-t0,int(counter-1)))
        self._appendJSON(exsclaim_json, new_captions_distributed)
        return exsclaim_json
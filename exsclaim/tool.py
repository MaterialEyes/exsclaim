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

from . import utils
from . import journal
from . import caption

from abc import ABC, abstractmethod

class ExsclaimTool(ABC):
    def __init__(self , model_path):
        self.model_path = model_path

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def _update_exsclaim(self):
        pass

    @abstractmethod
    def run(self):
        pass


class JournalScraper(ExsclaimTool):
    """ 
    JournalScraper object.
    Extract scientific figures from journal articles by passing  
    a json-style search query to the run method
    Parameters: 
    None
    """
    def __init__(self):
        pass

    def _load_model(self):
        pass

    def _update_exsclaim(self,exsclaim_dict,article_dict):
        exsclaim_dict.update(article_dict)
        return exsclaim_dict

    def _appendJSON(self,filename,json_dict):
        with open(filename,'w') as f: 
            json.dump(json_dict, f, indent=3)

    def _get_articles(self, search_query, j_instance):

        ## Check if any articles have already been scraped by checking
        ##   results_dir/_articles
        articles_visited = []
        if os.path.isfile(search_query['results_dir'] + "_articles"):
            with open(search_query['results_dir']+'_articles','r') as f:
                contents = f.readlines()
            articles_visited = [a.strip() for a in contents]

        ## Collects a new a list of articles, and checks them against
        ##   the articles that have already been visited before writing
        ##   them to the _articles file.
        articles = j_instance.get_article_extensions()
        with open(search_query['results_dir']+'_articles', 'a') as f:
            for article_number, article_path in enumerate(articles):
                if article_path.split("/")[-1] not in articles_visited:
                    f.write('%s\n' % article_path.split("/")[-1])
                if article_number >= search_query['maximum_scraped'] - 1:
                    break

        return articles

    def run(self,search_query,exsclaim_dict={}):
        utils.Printer("Running Journal Scraper\n")
        
        ## Checks that user inputted journal family has been defined and
        ## grabs instantiates an instance of the journal family object
        journal_family = search_query['journal_family']
        if journal_family not in journal.journals:
            raise NameError('journal family {0} is not defined'.format(journal_family))
        j_instance = journal.journals[journal_family](search_query)

        os.makedirs(search_query['results_dir'], exist_ok=True)
        t0 = time.time()
        counter = 1
        articles = self._get_articles(search_query, j_instance)
        for article in articles:
            utils.Printer(">>> ({0} of {1}) Extracting figures from: ".format(counter, len(articles))+\
                article.split("/")[-1])

            try:
                request = j_instance.get_domain_name() + article
                article_dict = j_instance.get_article_figures(request)
                exsclaim_dict = self._update_exsclaim(exsclaim_dict,article_dict)
            except:
                utils.Printer("<!> ERROR: An exception occurred in JournalScraper\n")
            
            # Save to file every N iterations (to accomodate restart scenarios)
            if counter%1000 == 0:
                self._appendJSON(search_query['results_dir']+'_js.json',exsclaim_dict)
            counter += 1

        t1 = time.time()
        utils.Printer(">>> Time Elapsed: {0:.2f} sec ({1} articles)\n".format(t1-t0,int(counter-1)))
        self._appendJSON(search_query['results_dir']+'_js.json',exsclaim_dict)
        return exsclaim_dict


class CaptionSeparator(ExsclaimTool):
    """ 
    CaptionSeparator object.
    Separate subfigure caption chunks from full figure captions 
    in an exsclaim_dict using custom caption nlp tools
    Parameters:
    model_path: str 
        Absolute path to caption nlp model 
    """
    def __init__(self , model_path=""):
        super().__init__(model_path)

    def _load_model(self):
        if "" in self.model_path:
            self.model_path = os.path.dirname(__file__)+'/captions/models/'
        return caption.load_models(self.model_path)

    def _update_exsclaim(self,exsclaim_dict,figure_name,delimiter,caption_dict):
        exsclaim_dict[figure_name]["caption_delimiter"] = delimiter
        for label in caption_dict:
            master_image = {"label": label, "description": caption_dict[label]['description'], "keywords": caption_dict[label]['keywords'], "general": caption_dict[label]['general']}
            exsclaim_dict[figure_name]['unassigned']['captions'].append(master_image)
        return exsclaim_dict

    def _appendJSON(self,filename,json_dict):
        with open(filename,'w') as f: 
            json.dump(json_dict, f, indent=3)

    def run(self,search_query,exsclaim_dict):
        utils.Printer("Running Caption Separator\n")
        os.makedirs(search_query['results_dir'], exist_ok=True)
        t0 = time.time()
        model = self._load_model()
        counter = 1
        for figure_name in exsclaim_dict:
            utils.Printer(">>> ({0} of {1}) ".format(counter,+\
                len(exsclaim_dict))+\
                "Parsing captions from: "+figure_name)
            try:
                caption_text  = exsclaim_dict[figure_name]['full_caption']
                delimiter = caption.find_subfigure_delimiter(model,caption_text)
                caption_dict  = caption.associate_caption_text(model,caption_text,search_query['query'])
                exsclaim_dict = self._update_exsclaim(exsclaim_dict,figure_name,delimiter,caption_dict) 
            except:
                utils.Printer("<!> ERROR: An exception occurred in CaptionSeparator\n")
        
            # Save to file every N iterations (to accomodate restart scenarios)
            if counter%1000 == 0:
                self._appendJSON(search_query['results_dir']+'_cs.json',exsclaim_dict)
            counter += 1

        t1 = time.time()
        utils.Printer(">>> Time Elapsed: {0:.2f} sec ({1} captions)\n".format(t1-t0,int(counter-1)))
        # -------------------------------- #  
        # -- Save current exsclaim_dict -- #
        # -------------------------------- # 
        with open(search_query['results_dir']+'_cs.json', 'w') as f:
            json.dump(exsclaim_dict, f, indent=3)
        # -------------------------------- #  
        # -------------------------------- # 
        # -------------------------------- # 
        return exsclaim_dict
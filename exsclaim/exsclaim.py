# Copyright 2019 MaterialEyes
# (see accompanying license files for details).

"""Definition of the ExsclaimTool classes.

This module defines the central objects in the EXSCLAIM! 
package. All the model classes are independent of each 
other, but they expose the same interface, so they are 
interchangeable.
"""
import json
import copy
import time
import utils
import journal
import caption
import figure
import imagetext


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

    def run(self,search_query,exsclaim_dict={}):
        utils.Printer("Running Journal Scraper\n") 
        counter = 1
        articles = journal.get_article_extensions(search_query)
        for article in articles:
            utils.Printer(">>> ({0} of {1}) ".format(counter,+\
                min(len(articles),search_query["maximum_scraped"]))+\
                "Extracting figures from: "+article.split("/")[-1])
            try:
                request = journal.get_base_url(search_query) + article
                article_dict = journal.get_article_figures(request,search_query['results_dir'])
                exsclaim_dict = self._update_exsclaim(exsclaim_dict,article_dict)
            except:
                utils.Printer("<!> ERROR: An exception occurred\n")
                return
            counter += 1
        utils.Printer(">>> SUCCESS!\n")
        # -------------------------------- #  
        # -- Save current exsclaim_dict -- #
        # -------------------------------- # 
        # with open(search_query['results_dir']+'0_js.json', 'w') as f:
        #     json.dump(exsclaim_dict, f, indent=3)
        # -------------------------------- #  
        # -------------------------------- # 
        # -------------------------------- # 
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
    def __init__(self , model_path):
        super().__init__(model_path)

    def _load_model(self):
        return caption.load_models(self.model_path)

    def _update_exsclaim(self,exsclaim_dict,figure_name,delimiter,caption_dict):
        exsclaim_dict[figure_name]["caption_delimiter"] = delimiter
        for label in caption_dict:
            master_image = {"label": label, "description": caption_dict[label]['description'], "keywords": caption_dict[label]['keywords']}
            exsclaim_dict[figure_name]['unassigned']['captions'].append(master_image)
        return exsclaim_dict

    def run(self,search_query,exsclaim_dict):
        utils.Printer("Running Caption Separator\n")
        model = self._load_model()
        counter = 1
        for figure_name in exsclaim_dict:
            utils.Printer(">>> ({0} of {1}) ".format(counter,+\
                len(exsclaim_dict))+\
                "Parsing captions from: "+figure_name)
            caption_text  = exsclaim_dict[figure_name]['full_caption']
            delimiter = caption.find_subfigure_delimiter(model,caption_text)
            caption_dict  = caption.associate_caption_text(model,caption_text,search_query['query'])
            exsclaim_dict = self._update_exsclaim(exsclaim_dict,figure_name,delimiter,caption_dict)
            counter += 1
        utils.Printer(">>> SUCCESS!\n")
        # -------------------------------- #  
        # -- Save current exsclaim_dict -- #
        # -------------------------------- # 
        # with open(search_query['results_dir']+'1_cs.json', 'w') as f:
        #     json.dump(exsclaim_dict, f, indent=3)
        # -------------------------------- #  
        # -------------------------------- # 
        # -------------------------------- # 
        return exsclaim_dict


class FigureSeparator(ExsclaimTool):
    """ 
    FigureSeparator object.

    Separate subfigure images from full figure image
    using CNN trained on crowdsourced labeled figures

    Parameters:

    None
    """
    def __init__(self , model_path):
        super().__init__(model_path)

    def _load_model(self):
        return figure.load_model(self.model_path)

    def _update_exsclaim(self,exsclaim_dict,figure_name,figure_dict):
        figure_name = figure_name.split("/")[-1]
        exsclaim_dict[figure_name]['unassigned'].update(figure_dict)
        return exsclaim_dict

    def run(self,search_query,exsclaim_dict):
        utils.Printer("Running Figure Separator\n")
        model = self._load_model()
        counter = 1
        figures = figure.get_figure_paths(search_query)
        for figure_name in figures:
            utils.Printer(">>> ({0} of {1}) ".format(counter,+\
                len(figures))+\
                "Extracting images from: "+figure_name.split("/")[-1])
            figure_dict = figure.extract_image_objects(model,figure_name)
            exsclaim_dict = self._update_exsclaim(exsclaim_dict,figure_name,figure_dict)
            counter += 1
        utils.Printer(">>> SUCCESS!\n")
        # -------------------------------- #  
        # -- Save current exsclaim_dict -- #
        # -------------------------------- # 
        # with open(search_query['results_dir']+'2_fs.json', 'w') as f:
        #     json.dump(exsclaim_dict, f, indent=3)
        # -------------------------------- #  
        # -------------------------------- # 
        # -------------------------------- # 
        return exsclaim_dict
        

class TextReader(ExsclaimTool):
    """ 
    TextReader object.

    Read cropped images containing subfigure label and scale bar text 

    Parameters:

    None
    """
    def __init__(self , model_path):
        super().__init__(model_path)

    def _load_model(self):
        return imagetext.load_model(self.model_path)

    def _update_exsclaim(self,exsclaim_dict,figure_name,images_dict):
        for label in exsclaim_dict[figure_name]['unassigned']['subfigure_labels']:
            updated_label = [b['text'] for b in images_dict if b['geometry'] == label['geometry']][0]
            if updated_label != 'low_confidence':
                label['text'] = updated_label
        return exsclaim_dict

    def run(self,search_query,exsclaim_dict):
        utils.Printer("Running Image Text Reader\n")
        model = self._load_model()
        counter = 1
        for figure_name in exsclaim_dict:
            utils.Printer(">>> ({0} of {1}) ".format(counter,+\
                len(exsclaim_dict))+\
                "Reading image text from: "+figure_name)
            path = exsclaim_dict[figure_name]["figure_path"]
            # These are images of unassigned subfigure labels
            images_dict = copy.deepcopy(exsclaim_dict[figure_name]['unassigned']['subfigure_labels'])
            images_dict = imagetext.read_image_text(model,path,images_dict,exsclaim_dict[figure_name]['caption_delimiter'])
            exsclaim_dict = self._update_exsclaim(exsclaim_dict,figure_name,images_dict)
            counter+=1
        utils.Printer(">>> SUCCESS!\n")
        # -------------------------------- #  
        # -- Save current exsclaim_dict -- #
        # # -------------------------------- # 
        # with open(search_query['results_dir']+'3_tr.json', 'w') as fd:
        #     json.dump(exsclaim_dict, fd, indent=3)
        # -------------------------------- #  
        # -------------------------------- # 
        # -------------------------------- # 
        return exsclaim_dict
from exsclaim.utilities.logging import Printer
import os
import csv
import json
from os.path import join
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import textwrap
import pathlib
import logging

from .utilities import boxes, paths
from .figure import FigureSeparator
from .tool import CaptionDistributor, JournalScraper


class Pipeline:

    def __init__(self , query_path):
        """ initialize a Pipeline to run on query path and save to exsclaim path

        Args:
            query_path (dict or path to json): An EXSCLAIM user query JSON
        """
        self.logger = logging.getLogger(__name__)
        # Load Query on which Pipeline will run
        self.current_path = pathlib.Path(__file__).resolve().parent
        if "test" == query_path:
            query_path = self.current_path / 'tests' / 'data' / 'nature_test.json'
        if isinstance(query_path, dict):
            self.query_dict = query_path
            self.query_path = ""
        else:
            assert os.path.isfile(query_path), "query path must be a dict, query path, or 'test', was {}".format(query_path)
            self.query_path = query_path
            with open(self.query_path) as f:
                # Load query file to dict
                self.query_dict = json.load(f)
        # Set up file structure
        base_results_dir = paths.initialize_results_dir(
            self.query_dict.get("results_dirs", None)
        )
        self.results_directory = (
            base_results_dir / self.query_dict["name"]
        )
        os.makedirs(self.results_directory, exist_ok=True)
        # Set up logging
        self.print = False
        for log_output in self.query_dict.get("logging", []):
            if log_output.lower() == "print":
                self.print = True
            else:
                log_output = self.results_directory / log_output
                logging.basicConfig(filename=log_output, filemode="w+", level=logging.INFO, style="{")
        # Check for an existing exsclaim json
        try:
            self.exsclaim_path = self.results_directory / "exsclaim.json"
            with open(self.exsclaim_path, 'r') as f:
                # Load configuration file values
                self.exsclaim_dict = json.load(f)
        except Exception as e:
            self.logger.info("No exsclaim.json file found, starting a new one.")
            # Keep preset values
            self.exsclaim_dict = {}

    def display_info(self, info):
        """ Display information to the user as the specified in the query

        Args:
            info (str): A string to display (either to stdout, a log file)
        """
        if self.print: 
            Printer(info)
        self.logger.info(info)

    def run(self, tools=None, figure_separator=True, caption_distributor=True, journal_scraper=True):
        """ Run EXSCLAIM pipeline on Pipeline instance's query path

        Args:
            tools (list of ExsclaimTools): list of ExsclaimTool objects
                to run on query path in the order they will run. Default
                argument is JournalScraper, CaptionDistributor, 
                FigureSeparator
            journal_scraper (boolean): true if JournalScraper should
                be included in tools list. Overriden by a tools argument
            caption_distributor (boolean): true if CaptionDistributor should
                be included in tools list. Overriden by a tools argument
            figure_separator (boolean): true if FigureSeparator should
                be included in tools list. Overriden by a tools argument
        Returns:
            exsclaim_dict (dict): an exsclaim json
        Modifies:
            self.exsclaim_dict
        """
        exsclaim_art = """
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        @@@@@@@@@@@@@@@@@@@&   /&@@@(   /@@@@@@@@@@@@@@@@@@@
        @@@@@@@@@@@@@@@ %@@@@@@@@@@@@@@@@@@@ *@@@@@@@@@@@@@@
        @@@@@@@@@@@@ @@@@@@@@@@@@@@,  .@@@@@@@@ *@@@@@@@@@@@
        @@@@@@@@@.#@@@@@@@@@@@@@@@@,    @@@@@@@@@@ @@@@@@@@@
        @@@@@@@&,@@@@@@@@@@@@@@@@@@.    @@@@@@@@@@@@ @@@@@@@
        @@@@@@ @@@@@@@@@@@@@@@@@@@@     @@@@@@@@@@@@@ @@@@@@
        @@@@@ @@@@@@@@@@@@@@@@@@@@@    *@@@@@@@@@@@@@@/@@@@@
        @@@@ @@@@@@@@@@@@@@@@@@@@@@    @@@@@@@@@@@@@@@@,@@@@
        @@@ @@@@@@@@@@@@@@@@@@@@@@&    @@@@@@@@@@@@@@@@@ @@@
        @@@,@@@@@@@@@@@@@@@@@@@@@@*   (@@@@@@@@@@@@@@@@@@%@@
        @@.@@@@@@@@@@@@@@@@@@@@@@@    @@@@@@@@@@@@@@@@@@@ @@
        @@ @@@@@@@@@@@@@@@@@@@@@@@    @@@@@@@@@@@@@@@@@@@ @@
        @@ @@@@@@@@@@@@@@@@@@@@@@/   &@@@@@@@@@@@@@@@@@@@ @@
        @@,@@@@@@@@@@@@@@@@@@@@@@    @@@@@@@@@@@@@@@@@@@@ @@
        @@@.@@@@@@@@@@@@@@@@@@@@&   @@@@@@@@@@@@@@@@@@@@@%@@
        @@@ @@@@@@@@@@@@@@@@@@@@@  /@@@@@@@@@@@@@@@@@@@@ @@@
        @@@@ @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@,@@@@
        @@@@@ @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*@@@@@
        @@@@@@ @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ @@@@@@
        @@@@@@@@ @@@@@@@@@@@@    @@@@@@@@@@@@@@@@@@@ @@@@@@@
        @@@@@@@@@.(@@@@@@@@@@     @@@@@@@@@@@@@@@@ @@@@@@@@@
        @@@@@@@@@@@@ @@@@@@@@@#   #@@@@@@@@@@@@ /@@@@@@@@@@@
        @@@@@@@@@@@@@@@ ,@@@@@@@@@@@@@@@@@@@ &@@@@@@@@@@@@@@
        @@@@@@@@@@@@@@@@@@@@   ,%@@&/   (@@@@@@@@@@@@@@@@@@@
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        """
        self.display_info(exsclaim_art)
        # set default values
        if tools is None:
            tools = []
            if journal_scraper:
                tools.append(JournalScraper(self.query_dict))
            if caption_distributor:
                tools.append(CaptionDistributor(self.query_dict))
            if figure_separator:
                tools.append(FigureSeparator(self.query_dict))
        # run each ExsclaimTool on search query
        for tool in tools:
            self.exsclaim_dict = tool.run(self.query_dict,self.exsclaim_dict)
        
        # group unassigned objects
        self.group_objects()

        # Save results as specified
        save_methods = self.query_dict.get("save_format", [])
        if 'save_subfigures' in save_methods:
            self.to_file()

        if 'visualization' in save_methods or 'visualize' in save_methods:
            for figure in self.exsclaim_dict:
                self.make_visualization(figure)

        if 'boxes' in save_methods:
            for figure in self.exsclaim_dict:
                self.draw_bounding_boxes(figure)

        if 'postgres' in save_methods:
            self.to_csv()
            self.to_postgres()
        elif 'csv' in save_methods:
            self.to_csv()

        if 'mongo' in save_methods:
            import pymongo
            db_client = pymongo.MongoClient(self.query_dict["mongo_connection"])
            db = db_client["materialeyes"]
            collection = db[self.query_dict["name"]]
            db_push = list(self.exsclaim_dict.values())
            collection.insert_many(db_push)

        return self.exsclaim_dict

    def assign_captions(self, figure):
        """ Assigns all captions to master_images JSONs for single figure

        Args:
            figure (dict): a Figure JSON
        Returns: 
            masters (list of dicts): list of master_images JSONs
            unassigned (dict): the updated unassigned JSON
        """
        unassigned = figure.get("unassigned", [])
        masters = []

        captions = unassigned.get("captions", {})
        not_assigned = set([a['label'] for a in captions])

        for index, master_image in enumerate(figure.get("master_images", [])):
            label_json = master_image.get("subfigure_label", {})
            subfigure_label = label_json.get("text", index)
            # remove periods or commas from around subfigure label
            processed_label = subfigure_label.replace(")","")
            processed_label = processed_label.replace("(","")
            processed_label = processed_label.replace(".","")
            paired = False
            for caption_label in captions:
                # remove periods or commas from around caption label
                processed_caption_label = caption_label['label'].replace(")","")
                processed_capiton_label = processed_caption_label.replace("(","")
                processed_caption_label = processed_caption_label.replace(".","")
                # check if caption label and subfigure label match and caption label
                # has not already been matched
                if (processed_caption_label.lower() == processed_label.lower()) and \
                (processed_caption_label.lower() in [a.lower() for a in not_assigned]):
                    master_image["caption"]  = caption_label['description']
                    master_image["keywords"] = caption_label['keywords']
                    master_image["general"]  = caption_label['general']
                    masters.append(master_image)
                    not_assigned.remove(caption_label['label'])
                    # break to next master image if a pairing was found
                    paired = True
                    break
            if paired:
                continue
            # no pairing found, create empty fields
            master_image["caption"] = []
            master_image["keywords"]= []
            master_image["general"] = []
            masters.append(master_image)

        # update unassigned captions
        new_unassigned_captions = []
        for caption_label in captions:
            if caption_label['label'] in not_assigned:
                new_unassigned_captions.append(caption_label)

        unassigned["captions"] = new_unassigned_captions
        return masters, unassigned

    def group_objects(self):
        """ Pair captions with subfigures for each figure in exsclaim json """
        search_query = self.query_dict
        self.display_info("Matching Image Objects to Caption Text\n")
        counter = 1
        for figure in self.exsclaim_dict:
            self.display_info(">>> ({0} of {1}) ".format(counter,+\
                len(self.exsclaim_dict))+\
                "Matching objects from figure: "+figure)
    
            figure_json = self.exsclaim_dict[figure]
            masters, unassigned = self.assign_captions(figure_json)

            figure_json["master_images"] = masters
            figure_json["unassigned"] = unassigned

            counter +=1 
        self.display_info(">>> SUCCESS!\n")
        with open(self.results_directory / 'exsclaim.json', 'w') as f:
            json.dump(self.exsclaim_dict, f, indent=3)
        
        return self.exsclaim_dict

    ### Save Methods ###

    def to_file(self):
        """ Saves data to a csv and saves subfigures as individual images
        
        Modifies:
            Creates directories to save each subfigure
        """
        search_query = self.query_dict
        self.display_info(
            ("Printing Master Image Objects to: {}/images\n".format(
                self.results_directory
            ))
        )
        for figure_name in self.exsclaim_dict:
            # figure_name is <figure_root_name>.<figure_extension>
            figure_root_name, figure_extension = os.path.splitext(figure_name)
            try:
                figure = plt.imread(
                    self.results_directory / "figures" / figure_name
                )
            except Exception as e:
                self.logger.exception(("Error printing {0} to file."
                    " It may be damaged!".format(figure_name)))
                figure = np.zeros((256,256))

            ## save each master, inset, and dependent image as their own file
            ## in a directory according to label
            figure_dict = self.exsclaim_dict[figure_name]
            for master_image in figure_dict.get("master_images", []):
                # create a directory for each master image in 
                # <results_dir>/images/<figure_name>/<subfigure_label>
                directory = (
                    self.results_directory / "images" / figure_root_name / 
                    master_image['subfigure_label']['text']
                )
                os.makedirs(directory, exist_ok=True)
                # generate the name of the master_image
                master_class  = ('uas' if master_image['classification'] is None 
                                       else master_image['classification'][0:3].lower())
                master_name = "/" + "_".join([figure_root_name,
                                              master_image['subfigure_label']['text'],
                                              master_class
                                            ]) + figure_extension
                # save master image to file
                master_patch = boxes.crop_from_geometry(master_image['geometry'], figure)
                master_patch = master_patch.copy(order='C')
                try:
                    plt.imsave(directory + master_name, master_patch)  
                except Exception as err:
                    self.logger.exception(("Error in saving cropped master"
                        " image of figure: {}".format(figure_root_name)))
                # Repeat for dependents of the master image to file
                for dependent_id, dependent_image in enumerate(master_image.get("dependent_images", [])):
                    dependent_root_name = "/".join([directory, "dependent"])
                    os.makedirs(dependent_root_name, exist_ok=True)
                    dependent_class  = ('uas' if dependent_image['classification'] is None 
                                              else dependent_image['classification'][0:3].lower())
                    dependent_name = "_".join([master_name.split('par')[0] +
                                              "dep" + str(dependent_id),
                                              dependent_class
                                              ]) + figure_extension
                    # save dependent image to file
                    dpatch = boxes.crop_from_geometry(dependent_image['geometry'], figure)
                    try:
                        plt.imsave(dependent_root_name+dependent_name,dpatch) 
                    except Exception as err:
                        self.logger.exception(("Error in saving cropped master"
                            " image of figure: {}".format(figure_root_name)))
                    # Repeat for insets of dependents of master image to file
                    for inset_id, inset_image in enumerate(dependent_image.get("inset_images", [])):
                        inset_root_name = "/".join([dependent_root_name,"inset"])
                        os.makedirs(inset_root_name, exist_ok=True)
                        inset_classification  = ('uas' if inset_image['classification'] is None
                                                 else inset_image['classification'][0:3].lower())
                        inset_name = "_".join([dependent_name.split(figure_extension)[0][0:-3] +
                                               "ins" + str(inset_id),
                                               inset_classification]) + figure_extension
                        
                        ipatch = boxes.crop_from_geometry(inset_image['geometry'],figure)
                        # save inset image to file
                        try:
                            plt.imsave(inset_root_name+inset_name,ipatch)
                        except Exception as err:
                            self.logger.exception(("Error in saving cropped master"
                                " image of figure: {}".format(figure_root_name)))
                # Write insets of masters to file
                for inset_id, inset_image in enumerate(master_image.get("inset_images", [])):
                    inset_root_name = "/".join([directory,"inset"])
                    os.makedirs(inset_root_name, exist_ok=True)
                    inset_classification  = ('uas' if inset_image['classification'] is None
                                              else inset_image['classification'][0:3].lower())
                    inset_name = "_".join([master_name.split(figure_extension)[0][0:-3] +
                                           "ins" + str(inset_id),
                                           inset_classification]) + figure_extension
                    ipatch = boxes.crop_from_geometry(inset_image['geometry'], figure)
                    # save inset image to file
                    try:
                        plt.imsave(inset_root_name+inset_name,ipatch)
                    except Exception as err:
                        self.logger.exception(("Error in saving cropped master"
                            " image of figure: {}".format(figure_root_name)))
        self.display_info(">>> SUCCESS!\n")

    def make_visualization(self, figure_name):
        """ Save subfigures and their labels as images

        Args:
            figure_path (str): A path to the image (.png, .jpg, or .gif)
                file containing the article figure
        Modifies:
            Creates images and text files in <save_path>/extractions folders
            showing details about each subfigure
        """
        os.makedirs(self.results_directory / "extractions", exist_ok=True)
        figure_json = self.exsclaim_dict[figure_name]
        master_images = figure_json.get("master_images", [])
        # to handle older versions that didn't store height and width
        for master_image in master_images:
            if 'height' not in master_image or 'width' not in master_image:
                geometry = master_image["geometry"]
                x1, y1, x2, y2 = boxes.convert_labelbox_to_coords(geometry)
                master_image['height'] = y2 - y1
                master_image['width'] = x2 - x1
        image_buffer = 150
        height = int(sum([master["height"] + image_buffer for master in master_images]))
        width = int(max([master["width"] for master in master_images]))
        image_width = max(width, 500)
        image_height = height

        ## Make and save images
        labeled_image = Image.new(mode="RGB",size=(image_width, image_height))
        draw = ImageDraw.Draw(labeled_image)
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf")

        figures_path = self.results_directory / 'figures'
        full_figure = Image.open(figures_path / figure_json["figure_name"]).convert("RGB")
        draw_full_figure = ImageDraw.Draw(full_figure)

        image_y = 0
        for subfigure_json in master_images:
            geometry = subfigure_json["geometry"]
            x1, y1, x2, y2 = boxes.convert_labelbox_to_coords(geometry)
            classification = subfigure_json["classification"]
            caption = "\n".join(subfigure_json.get("caption", []))
            caption = "\n".join(textwrap.wrap(caption, width=100))
            subfigure_label = subfigure_json["subfigure_label"]["text"]
            scale_bar_label = subfigure_json.get("scale_label", "None")
            scale_bars = subfigure_json.get("scale_bars", [])
            # Draw bounding boxes on detected objects
            for scale_bar in scale_bars:
                scale_geometry = scale_bar["geometry"]
                coords = boxes.convert_labelbox_to_coords(scale_geometry)
                bounding_box = [int(coord) for coord in coords]
                draw_full_figure.rectangle(bounding_box, width=2, outline="green")
                if scale_bar["label"]:
                    label_geometry = scale_bar["label"]["geometry"]
                    coords = boxes.convert_labelbox_to_coords(label_geometry)
                    bounding_box = [int(coord) for coord in coords]
                    draw_full_figure.rectangle(bounding_box, width=2, outline="green")
            label_geometry = subfigure_json["subfigure_label"]["geometry"]
            if label_geometry != []:
                coords = boxes.convert_labelbox_to_coords(label_geometry)
                bounding_box = [int(coord) for coord in coords]
                draw_full_figure.rectangle(bounding_box, width=1, outline="green")
            # Draw image
            subfigure = full_figure.crop((int(x1), int(y1), int(x2), int(y2)))
            text = ("Subfigure Label: {}\n"
                    "Classification: {}\n"
                    "Scale Bar Label: {}\n"
                    "Caption:\n{}".format(subfigure_label,
                                          classification,
                                          scale_bar_label,
                                          caption))
            text.encode("utf-8")
            labeled_image.paste(subfigure, box=(0,image_y))
            image_y += int(subfigure_json["height"])
            draw.text((0, image_y), text, fill="white", font=font)
            image_y += image_buffer

        del draw
        labeled_image.save(self.results_directory / "extractions" / figure_name)

    def draw_bounding_boxes(self, figure_name,
                            draw_scale=True,
                            draw_labels=False,
                            draw_subfigures=False):
        """ Save figures with bounding boxes drawn

        Args:
            figure_path (str): A path to the image (.png, .jpg, or .gif)
                file containing the article figure
            draw_scale (bool): If True, draws scale object bounding boxes
            draw_labels (bool): If True, draws subfigure label bounding boxes
            draw_subfigures (bool): If True, draws subfigure bounding boxes
        Modifies:
            Creates images and text files in <save_path>/boxes folders
            showing details about each subfigure
        """
        os.makedirs(self.results_directory / "boxes", exist_ok=True)
        figure_json = self.exsclaim_dict[figure_name]
        master_images = figure_json.get("master_images", [])



        figures_path = self.results_directory / 'figures'
        full_figure = Image.open(figures_path / figure_json["figure_name"]).convert("RGB")       
        draw_full_figure = ImageDraw.Draw(full_figure)
        
        scale_objects = []
        subfigure_labels = []
        subfigures = []
        for subfigure_json in master_images:
            # collect subfigures
            geometry = subfigure_json["geometry"]
            x1, y1, x2, y2 = boxes.convert_labelbox_to_coords(geometry)
            subfigures.append((x1, y1, x2, y2))
            # Collect scale bar objects
            scale_bars = subfigure_json.get("scale_bars", [])
            for scale_bar in scale_bars:
                scale_geometry = scale_bar["geometry"]
                coords = boxes.convert_labelbox_to_coords(scale_geometry)
                bounding_box = [int(coord) for coord in coords]
                scale_objects.append(bounding_box)
                if scale_bar["label"]:
                    label_geometry = scale_bar["label"]["geometry"]
                    coords = boxes.convert_labelbox_to_coords(label_geometry)
                    bounding_box = [int(coord) for coord in coords]
                    scale_objects.append(bounding_box)
            # Collect subfigure labels
            label_geometry = subfigure_json["subfigure_label"]["geometry"]
            if label_geometry != []:
                coords = boxes.convert_labelbox_to_coords(label_geometry)
                bounding_box = [int(coord) for coord in coords]
                subfigure_labels.append(bounding_box)
        unassigned_scale = figure_json.get("unassigned", {}).get("scale_bar_objects", [])
        for scale_object in unassigned_scale:
            if "geometry" in scale_object:
                scale_geometry = scale_object["geometry"]
                coords = boxes.convert_labelbox_to_coords(scale_geometry)
                bounding_box = [int(coord) for coord in coords]
                scale_objects.append(bounding_box)
            if scale_object.get("label") is not None:
                label_geometry = scale_object["label"]["geometry"]
                coords = boxes.convert_labelbox_to_coords(label_geometry)
                bounding_box = [int(coord) for coord in coords]
                scale_objects.append(bounding_box)
        # Draw desired bounding boxes
        if draw_scale:
            for bounding_box in scale_objects:
                draw_full_figure.rectangle(bounding_box, width=2, outline="green")
        if draw_labels:
            for bounding_box in subfigure_labels:
                draw_full_figure.rectangle(bounding_box, width=2, outline="green")
        if draw_subfigures:
            for bounding_box in subfigures:
                draw_full_figure.rectangle(bounding_box, width=2, outline="red")
        del draw_full_figure
        full_figure.save(self.results_directory / "boxes" / figure_name)

    def to_csv(self):
        """ Places data in a set of csv's ready for database upload 
        
        Modifies:
            Creates csv/ folder with article, figure, scalebar, scalebarlabel,
            subfigure, and subfigurelabel csvs.
        """
        exsclaim_json = self.exsclaim_dict
        csv_dir = self.results_directory / "csv"
        os.makedirs(csv_dir, exist_ok=True)
        articles = set()
        classification_codes = {
            'microscopy': "MC",
            "diffraction": "DF",
            "graph": "GR", 
            "basic_photo": "PH",
            "illustration": "IL",
            "unclear": "UN",
            "parent": "PT"
        }
        article_rows = []
        figure_rows = []
        subfigure_rows = []
        subfigure_label_rows = []
        scale_label_rows = []
        scale_rows = []
        for figure_name in exsclaim_json:
            figure_json = exsclaim_json[figure_name]
            # create row for unique articles 
            article_id = figure_json["article_name"]
            if article_id not in articles:
                article_row = [
                    article_id,
                    figure_json["title"],
                    figure_json["article_url"],
                    figure_json["license"],
                    figure_json["open"],
                    figure_json.get("authors", ""),
                    figure_json.get("abstract", "")
                ]
                article_rows.append(article_row)
                articles.add(article_id)
            base_name = ".".join(figure_name.split(".")[:-1])
            figure_id = "-fig".join(base_name.split("_fig"))
            # create row for figure.csv
            figure_row = [
                figure_id,
                figure_json["full_caption"],
                figure_json["caption_delimiter"],
                figure_json["image_url"],
                figure_json["figure_path"],
                figure_json["article_name"]
            ]
            figure_rows.append(figure_row)
            # loop through subfigures
            for master_image in figure_json.get("master_images", []):
                subfigure_label = master_image["subfigure_label"]["text"]
                subfigure_coords = boxes.convert_labelbox_to_coords(master_image["geometry"])
                subfigure_id = figure_id + "-" + subfigure_label
                subfigure_row = [
                    subfigure_id,
                    classification_codes[master_image["classification"]],
                    master_image.get("height", None),
                    master_image.get("width", None),
                    master_image.get("nm_height", None),
                    master_image.get("nm_width", None),
                    *subfigure_coords,
                    "\t".join(master_image["caption"]),
                    str(master_image["keywords"]).replace("[", "{").replace("]","}"),
                    str(master_image["general"]).replace("[", "{").replace("]","}"),
                    figure_id
                ]
                subfigure_rows.append(subfigure_row)
                if master_image["subfigure_label"].get("geometry", None):
                    subfigure_label_coords = boxes.convert_labelbox_to_coords(master_image["subfigure_label"]["geometry"])
                    subfigure_label_rows.append([
                        master_image["subfigure_label"]["text"],
                        *subfigure_label_coords,
                        master_image["subfigure_label"].get("label_confidence", None),
                        master_image["subfigure_label"].get("box_confidence", None),
                        subfigure_id
                ])
                for i, scale_bar in enumerate(master_image.get("scale_bars", [])):
                    scale_bar_id = subfigure_id + "-" + str(i)
                    scale_bar_coords = boxes.convert_labelbox_to_coords(scale_bar["geometry"])
                    scale_rows.append([
                        scale_bar_id,
                        *scale_bar_coords,
                        scale_bar.get("length", None),
                        scale_bar.get("line_label_distance", None),
                        scale_bar.get("confidence", None),
                        subfigure_id
                    ])
                    if scale_bar.get("label", None) is None:
                        continue
                    scale_label = scale_bar["label"]
                    scale_label_coords = boxes.convert_labelbox_to_coords(scale_label["geometry"])
                    scale_label_rows.append([
                        scale_label["text"],
                        *scale_label_coords,
                        scale_label.get("label_confidence", None),
                        scale_label.get("box_confidence", None),
                        scale_label.get("nm", None),
                        scale_bar_id
                    ])
        ## Save lists of rows to csvs
        with open(csv_dir / "article.csv", "w", encoding="utf-8", newline="") as article_file:
            article_writer = csv.writer(article_file)
            article_writer.writerows(article_rows)
        with open(csv_dir / "figure.csv", "w", encoding="utf-8", newline="") as figure_file:
            figure_writer = csv.writer(figure_file)
            figure_writer.writerows(figure_rows)
        with open(csv_dir / "subfigure.csv", "w", encoding="utf-8", newline="") as subfigure_file:
            subfigure_writer = csv.writer(subfigure_file)
            subfigure_writer.writerows(subfigure_rows)
        with open(csv_dir / "scalebar.csv", "w", encoding="utf-8", newline="") as scale_bar_file:
            scale_writer = csv.writer(scale_bar_file)
            scale_writer.writerows(scale_rows)
        with open(csv_dir / "scalebarlabel.csv", "w", encoding="utf-8", newline="") as scale_label_file:
            scale_label_writer = csv.writer(scale_label_file)
            scale_label_writer.writerows(scale_label_rows)
        with open(csv_dir / "subfigurelabel.csv", "w", encoding="utf-8", newline="") as subfigure_label_file:
            subfigure_label_writer = csv.writer(subfigure_label_file)
            subfigure_label_writer.writerows(subfigure_label_rows)

    def to_postgres(self):
        """ Send csv files to a postgres database
        
        Modifies:
            Fills an existing postgres database with data from csv/ dir
        """
        from .utilities.postgres import Database
        csv_dir = self.results_directory / "csv"
        db = Database("exsclaim")
        for csv_file in ["article.csv", "figure.csv", "subfigure.csv", "scalebar.csv", "scalebarlabel.csv", "subfigurelabel.csv"]:
            table_name = csv_file.replace(".csv", "")
            table_name = "results_" + table_name
            db.copy_from(csv_dir / csv_file, table_name)
            db.commit()
        db.close()        
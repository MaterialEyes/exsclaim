import os
import csv
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import textwrap
import pathlib

from .utilities import boxes, logging
from .figure import FigureSeparator
from .tool import CaptionDistributor, JournalScraper

class Pipeline:
    def __init__(self , query_path, test=False):
        """ initialize a Pipeline to run on query path and save to exsclaim path

        Args:
            query_path (dict or path to json): An EXSCLAIM user query JSON
            test (boolean): if True, initialize with test query json
        """
        if test:
            current_path = pathlib.Path(__file__).resolve().parent
            self.query_path = current_path / 'tests' / 'data' / 'nature_test.json'
            with open(self.query_path, "r") as f:
                self.query_dict = json.load(f)
        
        else:
            self.query_path = query_path
            try:
                with open(self.query_path) as f:
                    # Load query file to dict
                    self.query_dict = json.load(f)
            except: 
                self.query_dict = query_path
                self.query_path = ""

        
        try:
            self.exsclaim_path = self.query_dict["results_dir"] + "exsclaim.json"
            with open(self.exsclaim_path, 'r') as f:
                # Load configuration file values
                self.exsclaim_dict = json.load(f)
        except:
            # Keep preset values
            self.exsclaim_dict = {}


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
        print("""
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
        """)
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
        if 'csv' in save_methods or 'save_subfigures' in save_methods:
            self.to_file()

        if 'visualization' in save_methods or 'visualize' in save_methods:
            for figure in self.exsclaim_dict:
                self.make_visualization(figure)

        if 'boxes' in save_methods:
            for figure in self.exsclaim_dict:
                self.draw_bounding_boxes(figure)

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
        logging.Printer("Matching Image Objects to Caption Text\n")
        counter = 1
        for figure in self.exsclaim_dict:
            logging.Printer(">>> ({0} of {1}) ".format(counter,+\
                len(self.exsclaim_dict))+\
                "Matching objects from figure: "+figure)
    
            figure_json = self.exsclaim_dict[figure]
            masters, unassigned = self.assign_captions(figure_json)

            figure_json["master_images"] = masters
            figure_json["unassigned"] = unassigned

            counter +=1 
        logging.Printer(">>> SUCCESS!\n")

        with open(search_query['results_dir'] + 'exsclaim.json', 'w') as f:
            json.dump(self.exsclaim_dict, f, indent=3)
        
        return self.exsclaim_dict

    def to_file(self):
        """ Saves data to a csv and saves subfigures as individual images
        
        Note: 
            Will be run if 'save_subfigures' or 'csv' are in query['save_format']
        Modifies:
            Creates directories to save each subfigure and a csv file to save data
        """
        search_query = self.query_dict
        logging.Printer("".join(["Printing Master Image Objects to: ",
                              search_query['results_dir'].strip("/"),"/images","\n"]))
        # Rows for output csv file
        rows = [['article_url', 'figure_path','figure_num', 'image_path',
                 'master_label', 'dependent_id', 'inset_id', 'class',
                 'subclasses', 'caption', 'keywords', 'scale_bar', 'pixel_size']]

        for figure_name in self.exsclaim_dict:
            # figure_name is <figure_root_name>.<figure_extension>
            figure_root_name, figure_extension = os.path.splitext(figure_name)
            
            try:
                figure = plt.imread(search_query['results_dir'] + "figures/" + figure_name)
            except:
                print("Error printing {0} to file. It may be damaged!".format(figure_name))
                figure = np.zeros((256,256))

            ## Populate a CSV for each subfigure and save each master, inset, and
            ## dependent image as their own file in a directory according to label
            figure_dict = self.exsclaim_dict[figure_name]
            for master_image in figure_dict.get("master_images", []):
                # create a directory for each master image in 
                # <results_dir>/images/<figure_name>/<subfigure_label>
                directory = "/".join([search_query['results_dir'].strip("/"),
                                      "images",
                                      figure_root_name,
                                      master_image['subfigure_label']['text']
                                      ])
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
                    print("Error in saving cropped master image: {}".format(err))        
                # append data to rows for csv
                rows.append([figure_dict["article_url"], figure_dict["figure_path"],
                             figure_root_name.split("_fig")[-1], 
                             directory+master_name, 
                             master_image['subfigure_label']['text'], None, None,
                             master_image['classification'], master_image['general'],
                             master_image['caption'], master_image['keywords'],
                             master_image.get('scale_bar', {}).get('label', {}).get('text', None),
                             None])
                
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
                        print("Error in saving cropped dependent image: {}".format(err)) 
                    # append data to rows for csv
                    rows.append([figure_dict["article_url"], figure_dict["figure_path"],
                                figure_root_name.split("_fig")[-1],
                                dependent_root_name + dependent_name,
                                master_image['subfigure_label']['text'],
                                str(dependent_id), None,
                                dependent_image['classification'], None, None,
                                dependent_image.get('scale_bar', {}).get('label', {}).get('text', None),
                                None])
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
                            print("Error in saving cropped inset image: {}".format(err))
                        # append data to rows for csv
                        rows.append([figure_dict["article_url"], figure_dict["figure_path"],
                                     figure_root_name.split("_fig")[-1],
                                     inset_root_name+inset_name,
                                     master_image['subfigure_label']['text'],
                                     str(dependent_id), str(inset_id),
                                     inset_image['classification'], None, None,
                                     inset_image.get('scale_bar', {}).get('label', {}).get('text', None),
                                     None])
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
                        print("Error in saving cropped dependent image: {}".format(err))  
                    # append data to rows for csv
                    rows.append([figure_dict["article_url"], figure_dict["figure_path"],
                                 figure_root_name.split("_fig")[-1],
                                 inset_root_name + inset_name,
                                 master_image['subfigure_label']['text'],
                                 None, str(inset_id),
                                 inset_image['classification'], None, None,
                                 inset_image.get('scale_bar', {}).get('label', {}).get('text', None),
                                 None])
        # write rows to csv file
        with open(search_query['results_dir']+'labels.csv', 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(rows)

        logging.Printer(">>> SUCCESS!\n")

    def make_visualization(self, figure_name):
        """ Save subfigures and their labels as images

        Args:
            figure_path (str): A path to the image (.png, .jpg, or .gif)
                file containing the article figure
        Modifies:
            Creates images and text files in <save_path>/extractions folders
            showing details about each subfigure
        """
        os.makedirs(os.path.join(self.query_dict["results_dir"], "extractions"), exist_ok=True)
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

        full_figure = Image.open(figure_json["figure_path"]).convert("RGB")
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
        labeled_image.save(os.path.join(self.query_dict["results_dir"], "extractions", figure_name))

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
        os.makedirs(os.path.join(self.query_dict["results_dir"], "boxes"), exist_ok=True)
        figure_json = self.exsclaim_dict[figure_name]
        master_images = figure_json.get("master_images", [])


        full_figure = Image.open(figure_json["figure_path"]).convert("RGB")
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
        full_figure.save(os.path.join(self.query_dict["results_dir"], "boxes", figure_name))
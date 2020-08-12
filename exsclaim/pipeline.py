import os
import csv
import json
import time
import numpy as np
import matplotlib.pyplot as plt

from . import utils

def assign_captions(figure):
    """ Assigns all captions to master_images JSONs

    param figure: a Figure JSON

    returns (masters, unassigned): where masters is a list of master_images
        JSONs and unassigned is the updated unassigned JSON
    """
    unassigned = figure.get("unassigned", [])
    masters = []

    captions = unassigned.get("captions", {})
    not_assigned = set([a['label'] for a in captions])

    for index, master_image in enumerate(figure.get("master_images", [])):
        label_json = master_image.get("subfigure_label", {})
        subfigure_label = label_json.get("text", index)            
        processed_label = subfigure_label.replace(")","")
        processed_label = processed_label.replace("(","")
        processed_label = processed_label.replace(".","")
        paired = False
        for caption_label in captions:
            processed_caption_label = caption_label['label'].replace(")","")
            processed_capiton_label = processed_caption_label.replace("(","")
            processed_caption_label = processed_caption_label.replace(".","")
            if (processed_caption_label.lower() == processed_label.lower()) and \
               (processed_caption_label.lower() in [a.lower() for a in not_assigned]):
                master_image["caption"]  = caption_label['description']
                master_image["keywords"] = caption_label['keywords']
                master_image["general"]  = caption_label['general']
                masters.append(master_image)
                not_assigned.remove(caption_label['label'])
                paired = True
                break
        if paired:
            continue

        master_image["caption"] = []
        master_image["keywords"]= []
        master_image["general"] = []
        masters.append(master_image)

    # new_unassigned_captions = {}
    new_unassigned_captions = []
    for caption_label in captions:
        if caption_label['label'] in not_assigned:
            # new_unassigned_captions[caption_label] = captions[caption_label]
            new_unassigned_captions.append(caption_label)

    unassigned["captions"] = new_unassigned_captions
    return masters, unassigned

class Pipeline:
    def __init__(self , query_path, exsclaim_path):
        self.query_path = query_path
        try:
            with open(self.query_path) as f:
                # Load query file to dict
                self.query_dict = json.load(f)
        except: 
            self.query_dict = query_path
            self.query_path = ""

        self.exsclaim_path = exsclaim_path
        try:
            with open(exsclaim_path, 'r') as f:
                # Load configuration file values
                self.exsclaim_dict = json.load(f)
        except FileNotFoundError:
            # Keep preset values
            self.exsclaim_dict = {}

    def run(self, tools):
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
        search_query = self.query_dict
        for tool in tools:
            self.exsclaim_dict = tool.run(search_query,self.exsclaim_dict)
        return self.exsclaim_dict

    def group_objects(self):
        """
        Gather image objects that are part of the "unassigned" exsclaim_dict entry 
        and group together based on their association with a given subfigure label.
        """
        search_query = self.query_dict
        utils.Printer("Matching Image Objects to Caption Text\n")
        counter = 1
        for figure in self.exsclaim_dict:
            utils.Printer(">>> ({0} of {1}) ".format(counter,+\
                len(self.exsclaim_dict))+\
                "Matching objects from figure: "+figure)
    
            figure_json = self.exsclaim_dict[figure]
            masters, unassigned = assign_captions(figure_json)

            figure_json["master_images"] = masters
            figure_json["unassigned"] = unassigned

            counter +=1 
        utils.Printer(">>> SUCCESS!\n")

        with open(search_query['results_dir']+'exsclaim.json', 'w') as f:
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
        utils.Printer("".join(["Printing Master Image Objects to: ",
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
                master_patch = utils.crop_from_geometry(master_image['geometry'], figure)
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
                    dpatch = utils.crop_from_geometry(dependent_image['geometry'], figure)
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
                        
                        ipatch = utils.crop_from_geometry(inset_image['geometry'],figure)
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
                    ipatch = utils.crop_from_geometry(inset_image['geometry'], figure)
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

        utils.Printer(">>> SUCCESS!\n")
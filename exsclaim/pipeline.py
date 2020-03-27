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

    # not_assigned = set(captions.keys())
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

            # Shouldn't need this - its old. Remove eventually.
            # if masters == [] and len(unassigned['captions']) == 1:
            #     masters = [{'classification':'figure', 'confidence':None, 'geometry':figure_json["figure_path"],  \
            #                "caption":unassigned['captions'][0]['description'],\
            #                "keywords":unassigned['captions'][0]['keywords'],\
            #                "general":unassigned['captions'][0]['general']}]
            #     unassigned = {'master_images': [], 'dependent_images': [], 'inset_images': [], 'subfigure_labels': [], 'scale_bar_labels': [], 'scale_bar_lines': [], 'captions': []}

            figure_json["master_images"] = masters
            figure_json["unassigned"] = unassigned

            counter +=1 
        utils.Printer(">>> SUCCESS!\n")

        with open(search_query['results_dir']+'exsclaim.json', 'w') as f:
            json.dump(self.exsclaim_dict, f, indent=3)
        return self.exsclaim_dict

    def to_file(self):
        """



        """
        search_query = self.query_dict
        utils.Printer("".join(["Printing Master Image Objects to: ",search_query['results_dir'].strip("/"),"/images","\n"]))
        rows = [['article_url', 'figure_path','figure_num', 'image_path', \
                 'master_label', 'dependent_id', 'inset_id', 'class', 'subclasses', 'caption', 'keywords', 'scale_bar', 'pixel_size']]

        for figure_name in self.exsclaim_dict:
            fig_base, fig_ext = os.path.splitext(figure_name)
            
            try:
                figure = plt.imread(search_query['results_dir'] + "figures/" + figure_name)
            except:
                print("Error printing {0} to file. It may be damaged!".format(figure_name))
                figure = np.zeros((256,256))

            figure_dict = self.exsclaim_dict[figure_name]
            # Write masters to file
            if figure_dict.get("master_images") == []:
                pass
            # I think I can delete this...
            # elif not isinstance(figure_dict.get("master_images")[0]["geometry"], list):
            #     image_dir  = figure_dict.get("master_images")[0]["geometry"].split("/")[-1].split(".jpg")[0]+"/0"
            #     mname = figure_dict.get("master_images")[0]["geometry"].split("/")[-1].split(".jpg")[0]+"_0_fig.jpg"
            #     mbase = "/".join([search_query['results_dir'].strip("/"),"images",image_dir])+"/"
            #     os.makedirs(mbase, exist_ok=True)
            #     try:
            #         img = plt.imread(figure_dict.get("master_images")[0]['geometry'])
            #         plt.imsave(mbase+mname,img) 
            #     except:
            #         pass 
            else:
                for midx, mimage in enumerate(figure_dict.get("master_images", [])):
                    mbase = "/".join([search_query['results_dir'].strip("/"),"images",fig_base,mimage['subfigure_label']['text']])
                    mcls  = 'uas' if mimage['classification'] is None else mimage['classification'][0:3].lower()
                    mname = "/"+"_".join([figure_dict['figure_name'].split(fig_ext)[0],mimage['subfigure_label']['text'],mcls])+fig_ext
                    mpatch = utils.labelbox_to_patch_v2(mimage['geometry'],figure)
                    os.makedirs(mbase, exist_ok=True)
                    mpatch =mpatch.copy(order='C')
                    try:
                        plt.imsave(mbase+mname,mpatch)  
                    except:
                        pass        
                    rows.append([figure_dict["article_url"],figure_dict["figure_path"],fig_base.split("_fig")[-1],mbase+mname,\
                           mimage['subfigure_label']['text'], None, None,\
                           mimage['classification'],mimage['general'],mimage['caption'],mimage['keywords'],\
                           mimage.get('scale_bar', {}).get('label', {}).get('text', None),None])
                    # Write dependents of masters to file
                    for didx, dimage in enumerate(mimage.get("dependent_images", [])):
                        dbase = "/".join([mbase,"dependent"])
                        dcls  = 'uas' if dimage['classification'] is None else dimage['classification'][0:3].lower()
                        dname = "_".join([mname.split('par')[0]+"dep"+str(didx),dcls])+fig_ext
                        dpatch = utils.labelbox_to_patch_v2(dimage['geometry'],figure)
                        os.makedirs(dbase, exist_ok=True)

                        try:
                            plt.imsave(dbase+dname,dpatch) 
                        except:
                            pass 
        
                        rows.append([figure_dict["article_url"],figure_dict["figure_path"],fig_base.split("_fig")[-1],dbase+dname,\
                            mimage['subfigure_label']['text'], str(didx), None,\
                            dimage['classification'],None, None,\
                            dimage.get('scale_bar', {}).get('label', {}).get('text', None),None])
                        # Write insets of dependents to file
                        for iidx, iimage in enumerate(dimage.get("inset_images", [])):
                            ibase = "/".join([dbase,"inset"])
                            icls  = 'uas' if iimage['classification'] is None else iimage['classification'][0:3].lower()
                            iname = "_".join([dname.split(fig_ext)[0][0:-3]+"ins"+str(iidx),icls])+fig_ext
                            
                            ipatch = utils.labelbox_to_patch_v2(iimage['geometry'],figure)
                            os.makedirs(ibase, exist_ok=True)

                            try:
                                plt.imsave(ibase+iname,ipatch)
                            except:
                                pass 

                            rows.append([figure_dict["article_url"],figure_dict["figure_path"],fig_base.split("_fig")[-1],ibase+iname,\
                                mimage['subfigure_label']['text'], str(didx), str(iidx),\
                                iimage['classification'],None, None,\
                                iimage.get('scale_bar', {}).get('label', {}).get('text', None),None])
                    # Write insets of masters to file
                    for iidx, iimage in enumerate(mimage.get("inset_images", [])):
                        ibase = "/".join([mbase,"inset"])
                        icls  = 'uas' if iimage['classification'] is None else iimage['classification'][0:3].lower()
                        iname = "_".join([mname.split(fig_ext)[0][0:-3]+"ins"+str(iidx),icls])+fig_ext
                        ipatch = utils.labelbox_to_patch_v2(iimage['geometry'],figure)
                        os.makedirs(ibase, exist_ok=True)
                        
                        try:
                            plt.imsave(ibase+iname,ipatch)
                        except:
                            pass 
        
                        rows.append([figure_dict["article_url"],figure_dict["figure_path"],fig_base.split("_fig")[-1],ibase+iname,\
                            mimage['subfigure_label']['text'], None, str(iidx),\
                            iimage['classification'],None, None,\
                            iimage.get('scale_bar', {}).get('label', {}).get('text', None),None])

        with open(search_query['results_dir']+'labels.csv', 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(rows)
        csvFile.close()

        utils.Printer(">>> SUCCESS!\n")


import os
import time
import glob
import argparse
from tools.parse import *
from models.caption_nlp import *

def parse_command_line_arguments():
    """ reads command line arguments and returns them in dictionary """
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str)
    parser.add_argument('--image_extend', type=str, default="gif",
                         help='file extension for images (jpg, png, gif, etc.)')
    parser.add_argument('--csv_path', type=str, default='test/csv/key.csv')
    parser.add_argument('--stok', type=str, default='config/sentence_search_patterns.yml', 
                         help='path to the sentence token configuration file')
    parser.add_argument('--keyword', type=str, default="",
                         help='primary keyword (class) to associate with images')
    parser.add_argument('--synonymns', type=str, default="",
                         help='synonymns for the keyword')
    args = parser.parse_args()
    return vars(args)

def load_model():
    """ 
    Create the caption_nlp_model which is a custom spaCy nlp pipeline 
    for caption text with a caption matcher object. This rule-based model is
    re-generated for each call to load_model. For an ML model, train and 
    then load a model which has been written to disk.
    """    
    return caption_nlp_model()

def run_model(model,keyword_query = {}, images_dir = "./test/images", extension = "gif", csv_file = "./test/csv/key.csv", config_file = "./config/sentence_search_patterns.yml"):
    """ runs model on entries of CDE .csv output

    param model: spaCy nlp model (custom nlp pipeline + matcher object)
    param keyword_query: structured keyword quer
    param images_dir: path to input image directory. 
    param extension: file extension for images (ex: 'png', 'jpg', or 'gif')
    param csv_file: path to input csv file (result from CDE extraction)
    param config_file: path to the sentence token configuration file (.yaml)

    returns figure_to_result: dictionary mapping relevant figure information to tuples 
    """
    nlp, matcher = model

    figure_to_result = {}  
    for index, row in pd.read_csv(csv_file).iterrows():
        
        print("<*>"*4+"\n"+"Figure ",index,"\n")
        try:
            print("Caption: ",row["caption"][0:15]+"...")
            image_count, dt, de, dk  = parse_caption(row["caption"],[],nlp,matcher,config_file,keyword_query)
        except:
            print("Caption Parse Failure!")
            image_count, dt, de, dk  = -99,{},{},{}
        
        figure_to_result[str(row["article"].split(".")[0]+"_"+row["fig id"])] = (row["article"],row["fig id"],row["url"],row["caption"],image_count,de,dk)

    return figure_to_result

def generate_figure_json(figure_to_result):
    """ generates figure-level MaterialEyes JSON for whole input dataset """
    results_json = {}
    for figure_name in figure_to_result:

        explicit_caption_text = figure_to_result[figure_name][5]
        keyword_caption_text = figure_to_result[figure_name][6]
        
        # Populate with global info
        results_json[figure_name] = {"article" : figure_to_result[figure_name][0],\
                                     "fig id"  : figure_to_result[figure_name][1],\
                                     "url"     : figure_to_result[figure_name][2],\
                                     "caption" : figure_to_result[figure_name][3],\
                                     "num subfigs" : figure_to_result[figure_name][4]}
        
 
        results_json[figure_name]["subfigs"]={}

        # Create place holder for any image_level information that has not been assigned
        results_json[figure_name]["subfigs"] = {"unassigned":{}}
        
        # Populate with caption and keyword info relevant to each subfigure
        for sflb in explicit_caption_text:
            results_json[figure_name]["subfigs"][str(sflb)] = \
                {"type"     : "", 
                 "location" : "",\
                 "caption"  : explicit_caption_text[sflb],\
                 "keywords" : keyword_caption_text[sflb],\
                 "label"    : {},
                 "scalebar" : {},
                 "children" : {}}

    return results_json

def load_and_run_model(keyword_query = {}, images_dir = "./test/images", extension = "gif", csv_file = "./test/csv/key.csv", config_file = "./config/sentence_search_patterns.yml"):
    """ loads and runs model on input images and outputs data as MaterialEyes JSON
 
    param keyword_query: structured keyword quer
    param images_dir: path to input image directory. 
    param extension: file extension for images (ex: 'png', 'jpg', or 'gif')
    param csv_file: path to input csv file (result from CDE extraction)
    param config_file: path to the sentence token configuration file (.yaml)

    returns MaterialEyes JSON of all figures 
    """
    model = load_model()
    figure_to_result = run_model(model, keyword_query = keyword_query, images_dir = images_dir, extension = extension, csv_file = csv_file, config_file = config_file)
    return generate_figure_json(figure_to_result)

def main():
    """
    Visualize the detection result for the given image and the pre-trained model.
    """
    ## get arguments passed in through command line
    args = parse_command_line_arguments()
    query = {args['keyword']:args['synonymns'].split(",")}
    load_and_run_model(keyword_query = query, images_dir = args['images_dir'], extension = args['image_extend'], csv_file = args['csv_path'], config_file = args['stok'])
    

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("time elapsed = ",time.time()-start_time)

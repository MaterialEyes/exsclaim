import os
import json
import time
import glob
import argparse
from captions.tools.parse import *
from captions.models.caption_nlp import *

def js_r(filename):
    with open(filename) as f_in:
        return(json.load(f_in))

def js_add_subfigs(exsclaim_json,figure,de,dk):    
    """ 
    Add subfigure information to figure_level entry of an exsclaim_json
    """
    # Create place holder for any image_level information that has not been assigned
    exsclaim_json[figure]["unassigned"] = {}

    if "subfigs" not in exsclaim_json[figure]:
        exsclaim_json[figure]["subfigs"]={}

    for subfig_label in de:
        exsclaim_json[figure]["subfigs"][str(subfig_label)] = \
            {"type"     : "", 
             "location" : "",\
             "caption"  : de[subfig_label],\
             "keywords" : dk[subfig_label],\
             "label"    : {},
             "scalebar" : {},
             "children" : {}}

    return exsclaim_json

def load_model():
    """ 
    Create the caption_nlp_model which is a custom spaCy nlp pipeline 
    for caption text with a caption matcher object. This rule-based model is
    re-generated for each call to load_model. For an ML model, train and 
    then load a model which has been written to disk.
    """        
    return caption_nlp_model()

def run_model(model, query_json, exsclaim_json, config_file):
    """ 
    Runs model on figure entries in exsclaim_json
    returns exsclaim_json: with captions and keywords properly assigned
    """
    nlp, matcher = model

    keyword_query = {query_json["query"][a]["term"]:query_json["query"][a]["synonyms"] for a in query_json["query"] if query_json["query"][a]["term"] not in ""}

    for figure in exsclaim_json:
        try:
            print("Reading caption: ",exsclaim_json[figure]['caption'][0:min(50,int(0.25*len(exsclaim_json[figure]['caption'])))]+"...")
            image_count, dt, de, dk  = parse_caption(exsclaim_json[figure]['caption'],[],nlp,matcher,config_file,keyword_query)
        except:
            print("Caption Parse Failure!")
            image_count, dt, de, dk  = -99,{},{},{}

        exsclaim_json = js_add_subfigs(exsclaim_json,figure,de,dk)

    print("\n")
    return exsclaim_json

def load_and_run_model(query_json, exsclaim_json, config_file):
    """ 
    Loads and runs model on input exsclaim_json and populates subfigure entries 
 
    param query_json: json used as the query for the scraper
    param exsclaim_json: json output of scrapter
    param config_file: path to the sentence token configuration file (.yaml)

    returns exsclaim_json: with captions and keywords properly assigned
    """
    model = load_model()
    return run_model(model, query_json, exsclaim_json, config_file)

def main():
    
    exsclaim_json = js_r("scraper/exsclaim.json")
    query_json = js_r("scraper/query.json")
    
    exsclaim_json = load_and_run_model(query_json = query_json, exsclaim_json = exsclaim_json, config_file = "captions/config/sentence_search_patterns.yml")
    print(exsclaim_json)
    # with open("exsclaim.json", "w") as f:
    #     json.dump(exsclaim_json, f)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("time elapsed = ",time.time()-start_time)
import argparse
import json

from PIL import Image

import objectdetector as od
import textdetector as td
import webscraper as ws
import captionparser as cp
import cluster


def parse_command_line_arguments():
    """ parses arguments input throug command line as dictionary """
    ap = argparse.ArgumentParser()
    
    ap.add_argument('-i', '--input_dir', type=str, default='images',
                    help='Path to directory containing input images')

    ap.add_argument('-q', '--query', type=str, default='query.json',
                    help='Path to JSON containing an exsclaim-style query')

    ap.add_argument('-s', '--sentence_patterns', type=str, default='captions/config/sentence_search_patterns.yml',
                    help='Path to YAML containing a dictionary of sentence-level regexp patterns')

    args = vars(ap.parse_args())
    
    return args


if __name__ == '__main__':
    # parse command line args
    args = parse_command_line_arguments()

    # collect query info from json
    query_json = ws.js_r(args['query'])

    # create HTTP/1.1 request based on query 
    request = ws.create_request(query_json) #[GET_URL_BASE,SEARCH_EXTENSION]  

    # get url extensions for articles related to query
    article_extensions = ws.gather_searched_urls(query_json, request)

    # get figures from html files corresponding to article url extensions
    exsclaim_json = ws.gather_article_figures(query_json, request, article_extensions)    

    # parse full figure caption into chunks of subfigure text and populate exsclaim_json with subfigure info
    exsclaim_json, expected_captions = cp.load_and_run_model(query_json,exsclaim_json,args['sentence_patterns'])

    ## Run ObjectDetector on input images
    data = od.load_and_run_model(args['input_dir'])
     
    # add data to the exsclaim_json
    for image_name in exsclaim_json:
        image_json = exsclaim_json[image_name]
        
        unassigned = image_json["unassigned"]
        unassigned.update(data[image_name]["unassigned"])
        image_json["unassigned"] = unassigned
        exsclaim_json[image_name] = image_json

    ## Run TextDetector on subfigure and scalebar labels
    model, transform = td.load_model()
    i = 0
    for image_name in exsclaim_json:
        image_json = exsclaim_json[image_name]
        path = image_json["figure_path"]
        image = Image.open(path)
        image = image.convert("RGB")
        for label in image_json["unassigned"]["Subfigure Label"]:
            x = [label["geometry"][i]["x"] for i in range(len(label["geometry"]))]
            y = [label["geometry"][i]["y"] for i in range(len(label["geometry"]))]
            top, bottom = min(y), max(y)
            left, right = min(x), max(x)
           
            cropped = image.crop((left, top, right, bottom))
            text = td.run_model(cropped, transform, model)
            label["text"] = text
            i += 1

    ## Cluster objects in unassigned
    for figure in exsclaim_json:
        figure_json = exsclaim_json[figure]
        figure_json = cluster.cluster_figure(figure_json)
        exsclaim_json[figure] = figure_json

    with open("exsclaim.json", "w") as f:
        json.dump(exsclaim_json, f)
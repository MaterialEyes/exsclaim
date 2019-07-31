import argparse
import json

from PIL import Image

import objectdetector as od
import textdetector as td
import webscraper as ws


def parse_command_line_arguments():
    """ parses arguments input throug command line as dictionary """
    ap = argparse.ArgumentParser()
    
    ap.add_argument('-i', '--input-dir', type=str, default='images',
                    help='Path to directory containing input images')
    args = vars(ap.parse_args())
    
    return args










if __name__ == '__main__':
    ## parse command line args
    args = parse_command_line_arguments()

   
    ## Scrape web for keywords and initialize an exsclaim style json 
    dict_json = ws.js_r("scraper/query.json")
    request = ws.create_request(dict_json) #[GET_URL_BASE,SEARCH_EXTENSION]  
    # Session #1: Get article url extensions for articles related to query
    article_extensions = ws.session_1(dict_json, request)
    # Session #2: Request and save html files from article url extensions
    exsclaim_json = ws.session_2(dict_json, request, article_extensions)    


    ## Run ObjectDetector on input images
    data = od.load_and_run_model(args['input_dir'])
     
    # add data to the exsclaim_json
    for image_name in exsclaim_json:
        image_json = exsclaim_json[image_name]
        image_json.update(data[image_name])
        exsclaim_json[image_name] = image_json


    ## Run TextDetector on subfigure and scalebar labels
    model, transform = td.load_model()
    for image_name in exsclaim_json:
        image_json = exsclaim_json[image_name]
        path = image_json["figure_path"]
        image = Image.open(path)
        image = image.convert("RGB")
        for label in image_json["unassigned"]["subfigure_label"]:
            x = [label["geometry"][i]["x"] for i in range(len(label["geometry"]))]
            y = [label["geometry"][i]["y"] for i in range(len(label["geometry"]))]
            top, bottom = max(y), min(y)
            left, right = min(x), max(x)
            cropped = image.crop((left, top, bottom, right))
            text = td.run_model(cropped, transform, model)
            label["text"] = text

    print(exsclaim_json)


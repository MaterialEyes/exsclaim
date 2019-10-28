import json
import argparse
import os
import shutil
import requests
import random

from PIL import Image


ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input-json", type=str, 
                help="path to labelbox json data")
ap.add_argument("-s", "--split", type=float, 
                help="percentage of training images")

args = vars(ap.parse_args())


input_json = args["input_json"]
split = args["split"]

# load json into python dict
with open(input_json, "r") as f:
    data = json.load(f)

# create output directories
train_path = "data/train"
test_path = "data/test"
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)
os.makedirs("images", exist_ok=True)

def save_image(url, name):
    """ downloads image at url to 'images/name' """
    response = requests.get(url, stream=True)
    with open(os.path.join("images", name), "wb") as image_file:
        shutil.copyfileobj(response.raw, image_file)

# format for PyTorch-YOLOv3
for image in data:

    url = image["Labeled Data"]
    name = image["External ID"]
    raw_name = name.split(".")[0]
    save_image(url, name)
    
    full_image = Image.open(os.path.join("images", name))
    i = 0
    for label in image["Label"].get("Subfigure Label", []):
        text = label["text"]
        cropped_name = raw_name + "_" + str(i) + "." + name.split(".")[-1]
        if random.random() <= split:
            dataset = "train"
        else: 
            dataset = "test"
        
        geo = label["geometry"]
        xs = [geo[i]["x"] for i in range(len(geo))]
        ys = [geo[i]["y"] for i in range(len(geo))]
        left = min(xs)
        right = max(xs)
        top = min(ys)
        bottom = max(ys)
        
        os.makedirs(os.path.join(dataset, text), exist_ok=True)
        cropped_image = full_image.crop((left, top, right, bottom))
        cropped_image.save(os.path.join(dataset, text, cropped_name))          
        i += 1



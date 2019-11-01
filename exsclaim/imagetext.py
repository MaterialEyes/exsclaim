import os
import json
import yaml
import time
import torch
import torchvision
import numpy as np
from PIL import Image
# used to be called arch
from .imagetexts.architecture import *

file_loc = os.path.dirname(os.path.realpath(__file__))
## get class id's
with open(file_loc+"/imagetexts/classes.json", "r") as f:
    index_to_text = json.load(f)

with open(file_loc+'/captions/models/reference.yml', 'r') as f:
    ref = yaml.safe_load(f)


def load_model(model_path=str) -> "figure_separator_model":
    """ Creates model with specified architecture and weights 

    param architecture: string. Name of NN architecture in architectures.py
        used for model 
    param model_name: string. Name of file containing model weights (model
        state_dict). Usually ends in '.pt'

    returns: PyTorch model, and data_transformation function
    """

    architecture="CNN1"
    model_name="read_sflabel_5_CNN150_adam.pt"

    # assign devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # assign architecture
    if architecture.upper() == "CNN1":
        model = CNN1()
    elif architecture.upper() == "CNN2":
        model = CNN2()

    ## ADD NEW ARCHITECTURES HERE ##
    # elif architecture.upper() == "NAME_OF_ARCHITECTURE":
    #    model = LOAD ARCHITECTURE

    else:
        print("invalid architecture given, using CNN1")
        model = CNN1()
    
    model.load_state_dict(torch.load(model_path + model_name))
    model.to(device)
    model.eval()

    data_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((64,64)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
 
    return (model, data_transform)


def run_model(image, transform, model):
    """ Runs the model on the image """
    # transform image to be 64x64
    image_tensor = transform(image)#.float()
    image_tensor = image_tensor.unsqueeze_(0)
    #input_image = Variable(image_tensor)
    #input_image = input_image.to(device)
    output = model(image_tensor)
    # print("argsort: ",output.data.cpu().numpy().argsort()[0][-10:])
    # print(type(output.data.cpu().numpy().argsort()[0][-10:]))
    best_idxs = list(output.data.cpu().numpy().argsort()[0][-5:])
    best_idxs.reverse()
    score_list = list(output.data.cpu().numpy()[0])
    best_idx = output.data.cpu().numpy().argmax()

    key = "low_confidence"
    top_five = [index_to_text[str(i)] for i in best_idxs]

    for idx in best_idxs:
        if score_list[idx] > 6:
            key = str(index_to_text[str(idx)]).lower()
            break
    return key


def read_image_text(text_reader_model=tuple, figure_path=str, images_dict=dict, char_delim=str):
    
    model, transform = text_reader_model

    image = Image.open(figure_path)
    image = image.convert("RGB")

    # Set of unassigned labels
    if char_delim == 'alpha' or char_delim == 'ALPHA':
        candidate_labels = set([a.lower() for a in ref['alphabet']])
    elif char_delim == 'roman':
        candidate_labels = set([a.lower() for a in ref['roman numerals']])
    elif char_delim == 'position':
        candidate_labels = set([a.lower() for a in ref['positions']])
    else:
        candidate_labels = set([0])
    
    for label in images_dict:
        x = [label["geometry"][i]["x"] for i in range(len(label["geometry"]))]
        y = [label["geometry"][i]["y"] for i in range(len(label["geometry"]))]
        top, bottom = min(y), max(y)
        left, right = min(x), max(x)
        cropped = image.crop((left, top, right, bottom))
        
        # Text recognition prediciton
        text = run_model(cropped, transform, model)
        
        if text != "low_confidence":
            # Only assign if it is assignable
            if text.strip("(").strip(")").strip(".").lower() in candidate_labels:
                candidate_labels.remove(text.strip("(").strip(")").strip(".").lower())
            else:
                text = "low_confidence"

        label["text"] = text

    return images_dict

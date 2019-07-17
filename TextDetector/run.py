import time
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import architectures as a
from PIL import Image
import argparse

def parse_command_line_arguments():
    """ reads arguments input at command line and outputs a dictionary """
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--architecture", type=str, help="Name of nn.Module subclass to use")
    ap.add_argument("-m", "--model_name", type=str, help="Path to pytorch model state_dict")
    ap.add_argument("-i", "--input_directory", type=str, help="Path to input images")

    args = vars(ap.parse_args())
    
    return args

def imshow(img):
    """ displays image """
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def load_model(architecture, model_name):
    """ Creates model with specified architecture and weights 

    param architecture: string. Name of NN architecture in architectures.py
        used for model 
    param model_name: string. Name of file containing model weights (model
        state_dict). Usually ends in '.pt'

    returns: PyTorch model, and data_transformation function
    """
    # assign devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # assign architecture
    if architecture.upper() == "CNN1":
        model = a.CNN1()
    elif architecture.upper() == "CNN2":
        model = a.CNN2()
    else:
        print("invalid architecture given, using CNN1")
        model = a.CNN1()
    
    model.load_state_dict(torch.load(os.getcwd() + "/models/" + model_name))
    model.to(device)
    model.eval()

    data_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((64,64)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
 
    return model, data_transform


def run_model(image, transform, model):
    """ Runs the model on the image """
    # transform image to be 64x64
    image_tensor = transform(image)#.float()
    image_tensor = image_tensor.unsqueeze_(0)
    #input_image = Variable(image_tensor)
    #input_image = input_image.to(device)
    output = model(image_tensor)
    index = output.data.cpu().numpy().argmax()
    return index


if __name__ == '__main__':
    start_time = time.time()
    
    args = parse_command_line_arguments()
    model, transform = load_model(args["architecture"], args["model_name"])
    
    # loop through directory
    directory = os.fsencode(args["input_directory"])
    results = {}
    for file in os.listdir(directory):
        file_name = os.fsdecode(file)
        path = args["input_directory"] + "/" + file_name
        image = Image.open(path)
        image = image.convert("RGB")
        results[file_name] = run_model(image, transform, model)

    print(results)
    print("Completed text recognition in {} seconds".format(time.time() - start_time))

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from .dataset import ScaleBarDataset
from .engine import train_one_epoch, evaluate
from PIL import Image, ImageDraw
from . import utils
import torch
from torch import optim
import torchvision.transforms as T
import skimage as io
import numpy as np
import os
import pathlib
import random
import cv2
import argparse



def random_gaussian_blur(image):
    image = np.array(image)
    random_value = random.randint(0,4)
    if random_value == 2:
        image_blur = cv2.GaussianBlur(image,(15,15),10)
        new_image = image_blur
        return new_image
    return image

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.Lambda(random_gaussian_blur))
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

def get_model(train_status):
    num_classes = 3 #background, scale bar, scale_label
    train_statuses = {"p" : True, "s": False}
    faster = train_statuses[train_status[0]]
    resnet = train_statuses[train_status[1]]
    
    current_model = "scale_bar_model_{}".format(train_status)
    
    # load an object detection model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=faster, pretrained_backbone=resnet)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new on
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
 
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    learning_rate = 0.001
    optimizer = optim.Adam(params, lr=learning_rate)
    # and a learning rate scheduler
    #lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    #lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60])
    lr_scheduler = None
    # find if there is a previous checkpoint
    current_path = pathlib.Path(__file__).resolve(strict=True)
    checkpoints = current_path.parent / 'checkpoints'
    largest = -1
    best_checkpoint = None
    for checkpoint in os.listdir(checkpoints):
        filename = checkpoint.split(".")[0]
        if not os.path.isfile(os.path.join(checkpoints, checkpoint)):
            continue
        model_name, number = filename.split("-")
        if model_name != current_model:
            continue
        number = int(number)
        if number > largest:
            best_checkpoint = checkpoint
            largest = number
    if best_checkpoint == None:
        return model, lr_scheduler, optimizer, 0

    best_checkpoint = checkpoints / best_checkpoint
    cuda = torch.cuda.is_available() and (gpu_id >= 0)
    if cuda:
        checkpoint = torch.load(best_checkpoint)
        model = model.cuda()  
    else:
        checkpoint = torch.load(best_checkpoint, map_location='cpu')
   
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    #lr_scheduler.load_state_dict(checkpoint["lr_state_dict"])
    epoch = checkpoint['epoch']   

    return model, lr_scheduler, optimizer, epoch

def train_object_detector(train_status):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    current_path = pathlib.Path(__file__).resolve(strict=True)
    checkpoints = current_path.parent / "checkpoints"
    root_directory = current_path.parent.parent.parent / 'tests' / 'data'
    # use our dataset and defined transformations
    dataset_train = ScaleBarDataset(root_directory, get_transform(train=True), False)
    dataset_test = ScaleBarDataset(root_directory, get_transform(train=False), True)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=32, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=32, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model, lr_scheduler, optimizer,  start_epoch = get_model(train_status)
    # move model to the right device
    model.to(device)

    model_name = "scale_bar_model_{}".format(train_status)
    num_epochs = 200
    for epoch in range(start_epoch + 1, num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device,
                        epoch, print_freq=5, lr_scheduler=lr_scheduler,  model_name=model_name)
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device, model_name=model_name)

        if epoch % 1 == 0:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        #'lr_state_dict': lr_scheduler.state_dict()
                       },
                       checkpoints / "scale_bar_model_{}-{}.pt".format(train_status, epoch))


def run():
    ## Load text recognition model
    scale_bar_checkpoint = 'scale_bar_model.pt'
    scale_bar_model = get_model(3)
    cuda = torch.cuda.is_available() and (gpu_id >= 0)
    if cuda:
        scale_bar_model.load_state_dict(torch.load(scale_bar_checkpoint)["model_state_dict"])
        scale_bar_model = scale_bar_model.cuda()  
    else:
        scale_bar_model.load_state_dict(torch.load(scale_bar_checkpoint, map_location='cpu')["model_state_dict"])
    
    scale_bar_model.eval()

    # use our dataset and defined transformations
    dataset = ScaleBarDataset('', get_transform(train=True))
    dataset_test = ScaleBarDataset('', get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-100])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-100:])
    image, _ = dataset_test[0]
    label_boxes = np.array(dataset_test[0][1]["boxes"])


    with torch.no_grad():
        prediction = scale_bar_model([image])

    image = Image.fromarray(image.mul(255).permute(1, 2,0).byte().numpy())
    image.save("test2.png")
    draw = ImageDraw.Draw(image)# draw groundtruth
    for elem in range(len(label_boxes)):
        draw.rectangle([(label_boxes[elem][0], label_boxes[elem][1]), (label_boxes[elem][2], label_boxes[elem][3])], outline ="green", width =3)
    for element in range(len(prediction[0]["boxes"])):
        boxes = prediction[0]["boxes"][element].cpu().numpy()
        score = np.round(prediction[0]["scores"][element].cpu().numpy(),
                        decimals= 4)
        if score > 0.8:
            draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])], 
            outline ="red", width =3)
            draw.text((boxes[0], boxes[1]), text = str(score))
    
    image.save("test.png")

if __name__ == "__main__":
    # for command line usage
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--train_status", type=str, default="pp",
                help="whether faster rcc and resent backbone are pretrained")

    args = vars(ap.parse_args())

    train_status = args["train_status"]

    train_object_detector(train_status)

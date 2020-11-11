import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataset import ScaleBarDataset
from engine import train_one_epoch, evaluate
from PIL import Image, ImageDraw
import utils
import torch
import torchvision.transforms as T
import skimage as io
import numpy as np
import os
import pathlib

current_model = "scale_bar_model"

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_model(num_classes):
    # load an object detection model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)# get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new on
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
 
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    # find if there is a previous checkpoint
    largest = -1
    best_checkpoint = None
    for checkpoint in os.listdir('checkpoints'):
        filename = checkpoint.split(".")[0]
        if not os.path.isfile(os.path.join('checkpoints', checkpoint)):
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

    best_checkpoint = 'checkpoints/' + best_checkpoint
    cuda = torch.cuda.is_available() and (gpu_id >= 0)
    if cuda:
        checkpoint = torch.load(best_checkpoint)["model_state_dict"]
        model = model.cuda()  
    else:
        model.load_state_dict(torch.load(best_checkpoint, map_location='cpu')["model_state_dict"])
   
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    lr_scheduler.load_state_dict(checkpoint["lr_state_dict"])
    epoch = checkpoint['epoch']   

    return model, lr_scheduler, optimizer, epoch

def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 3

    current_path = pathlib.Path(__file__).resolve(strict=True)
    root_directory = current_path.parent.parent.parent.parent / 'dataset' / 'dataset_generation' 
    print(root_directory)
    # use our dataset and defined transformations
    dataset_train = ScaleBarDataset(root_directory, get_transform(train=True), False)
    dataset_test = ScaleBarDataset(root_directory, get_transform(train=False), True)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=64, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=64, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model, lr_scheduler, optimizer,  start_epoch = get_model(num_classes)
    # move model to the right device
    model.to(device)


    num_epochs = 200

    for epoch in range(start_epoch + 1, num_epochs):
        print("EPOCH: ", epoch)
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=5)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

        if epoch % 1 == 0:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_state_dict': lr_scheduler.state_dict()
                       },
                       current_model + "-{}.pt".format(epoch))


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
    main()

## Adapted from https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse
import os

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def load_split_train_test(datadir):
    train_transforms = transforms.Compose([transforms.Resize(256), 
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),])
    test_transforms =  transforms.Compose([transforms.Resize(256), 
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),])
    train_data = datasets.ImageFolder(datadir + "/train", transform=train_transforms)
    test_data = datasets.ImageFolder(datadir + "/test", transform=test_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True)
   return trainloader, testloader

def get_model(size, dataset_name, classes):
    model = None
    if size == 50:
        model = models.resnet50(pretrained=True)
    elif size == 152:
        model = models.resnet152(pretrained=True)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    for param in model.parameters():
        param.requires_grad = False
      
    model.fc = nn.Sequential(nn.Linear(2048, 512),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(512, classes),
                                    nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
    model.to(device)
   
    # find previous model to resume training
    largest = -1
    best_checkpoint = None
    for checkpoint in os.listdir('checkpoints'):
        filename = checkpoint.split(".")[0]
        model_name, number = filename.split("-")
        if model_name != dataset_name + "_" + str(size):
            continue
        number = int(number)
        if number > largest:
            best_checkpoint = checkpoint
            largest = number
    # training hasn't started
    if best_checkpoint == None:
        return model, 0, criterion, optimizer

    # Load saved information from checkpoint
    best_checkpoint = 'checkpoints/' + best_checkpoint
    cuda = torch.cuda.is_available() and (gpu_id >= 0)
    if cuda:
        model.load_state_dict(torch.load(best_checkpoint)["model_state_dict"])
        optimizer.load_state_dict(torch.load(best_checkpoint)['optimizer_state_dict'])
        model = model.cuda()
    else:
        model.load_state_dict(torch.load(best_checkpoint, map_location='cpu')["model_state_dict"])
        optimizer.load_state_dict(torch.load(best_checkpoint, map_location='cpu')['optimizer_state_dict'])

    ckpt = torch.load(best_checkpoint)
    epoch = ckpt['epoch']

    return model, epoch, criterion, optimizer


def main(dataset_name, classes, model_size, save_frequency):
    trainloader, testloader = load_split_train_test("~/exsclaim/dataset/dataset_generation/{}".format(dataset_name))
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model, current_epoch, criterion, optimizer = get_model(model_size, dataset_name, classes)

    epochs = 1000
    running_loss = 0
    recent_train_losses, recent_test_losses = [], []
    accuracies, unsaved_epochs = [], []
    for epoch in range(current_epoch, epochs):
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        ## Test model performance on testing data
        test_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)
                test_loss += batch_loss.item()
                
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        ## Save results until time to write to file
        recent_train_losses.append(running_loss/len(trainloader))
        recent_test_losses.append(test_loss/len(testloader))     
        accuracies.append(float(accuracy/len(testloader)))
        unsaved_epochs.append(epoch)

        ## Prepare to resume training
        running_loss = 0
        model.train()

        if epoch % save_frequency == 0:
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                    }, "checkpoints/{}_{}-{}.pt".format(dataset_name, model_size, epoch))
        
            with open("results/{}_{}.txt".format(dataset_name, model_size), "a") as f:
                for i in range(len(unsaved_epochs)): 
                    f.write((f"Epoch {unsaved_epochs[i]}/{epochs}.. "
                        f"Train loss: {recent_train_losses[i]:.3f}.. "
                        f"Test loss: {recent_test_losses[i]:.3f}.. "
                        f"Test accuracy: {accuracies[i]:.3f}\n"))
            unsaved_epochs = []
            recent_test_losses, recent_train_losses = [], []
            accuracies = []

if __name__ == "__main__":
    # for command line usage
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_name", type=str, default="no_text",
                help="path to directory of images to train on")
    ap.add_argument("-c", "--classes", type=int,
                help="number of classes")
    ap.add_argument("-f", "--save_frequency", type=int, default=10,
                help="save model every n epochs")
    ap.add_argument("-s", "--model_size", type=int, default=50,
                help="size of resnet model")
    ap.add_argument("-b", "--batch_size", type=int, default=128)
    args = vars(ap.parse_args())

    dataset_name = args["dataset_name"]
    classes = args["classes"]
    save_frequency = args["save_frequency"]
    model_size = args["model_size"]
    main(dataset_name, classes, model_size, save_frequency)

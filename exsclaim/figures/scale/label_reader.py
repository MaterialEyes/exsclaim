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

def load_split_train_test(datadir, valid_size = .2):
    train_transforms = transforms.Compose([transforms.Resize(256), 
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),])
    test_transforms =  transforms.Compose([transforms.Resize(256), 
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),])
    train_data = datasets.ImageFolder(datadir, transform=train_transforms)
    test_data = datasets.ImageFolder(datadir, transform=test_transforms)
    print(train_data.class_to_idx)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split+1:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=128)
    testloader = torch.utils.data.DataLoader(test_data,
                   sampler=test_sampler, batch_size=128)
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

    best_checkpoint = 'checkpoints/' + best_checkpoint
    cuda = torch.cuda.is_available() and (gpu_id >= 0)
    if cuda:
        model.load_state_dict(torch.load(best_checkpoint)["model_state_dict"])
        model = model.cuda()
    else:
        model.load_state_dict(torch.load(best_checkpoint, map_location='cpu')["model_state_dict"])

    ckpt = torch.load(best_checkpoint)
    epoch = ckpt['epoch']

    return model, epoch, criterion, optimizer


def main(dataset_name, classes, model_size, save_frequency):
    trainloader, testloader = load_split_train_test("~/exsclaim/dataset/dataset_generation/{}".format(dataset_name), .2)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model, current_epoch, criterion, optimizer = get_model(model_size, dataset_name, classes)

    epochs = 1000
    steps = 0
    running_loss = 0
    print_every = 10
    train_losses, test_losses = [], []
    for epoch in range(current_epoch + 1, epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
            if steps % print_every == 0:
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
                train_losses.append(running_loss/len(trainloader))
                test_losses.append(test_loss/len(testloader))                    
                with open("results/{}_{}.txt".format(dataset_name, model_size), "a") as f: 
                    f.write((f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(testloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(testloader):.3f}\n"))
                running_loss = 0
                model.train()
        if epoch % save_frequency == 0:
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                    }, "checkpoints/{}_{}-{}.pt".format(dataset_name, model_size, epoch))

if __name__ == "__main__":
    #load_split_train_test("scale_label_dataset", 0.2)
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
    args = vars(ap.parse_args())

    dataset_name = args["dataset_name"]
    classes = args["classes"]
    save_frequency = args["save_frequency"]
    model_size = args["model_size"]
    main(dataset_name, classes, model_size, save_frequency)
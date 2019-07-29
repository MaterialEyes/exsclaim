import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import architectures as a

import argparse


## Parse command line arguments

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=int, default=100)
ap.add_argument("-a", "--architecture", type=str, help="Name of nn.Module subclass to use")
ap.add_argument("-b", "--batch-size", type=int, default=50)
ap.add_argument("-g", "--gd_alg", type=str, default="adam")
ap.add_argument("-l", "--learn-rate", type=float, default=0.0001) 
args = vars(ap.parse_args())

epochs = args["epochs"]
architecture = args["architecture"]
batch_size = args["batch_size"]
optimization_alg = args["gd_alg"]
learning_rate = args["learn_rate"]
outfile = "models/" + architecture + "_" + optimization_alg + "_" + str(batch_size) + "_" + str(learning_rate) + ".txt"

## Utility Functions 

def imshow(img):
    """ displays image """
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


## Set up training environment

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
model.to(device)

# assign gradient descent optimizer
if optimization_alg == "adam":
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
elif optimization_alg == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
else:
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Prepare Data

# Transform images to all be 64x64
data_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((64,64)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

# Create a training and testing set from directories such that images are contained in 
#   directories named their expected label
trainset= torchvision.datasets.ImageFolder(root='train',
                                           transform=data_transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size, shuffle=True,num_workers=4)

testset= torchvision.datasets.ImageFolder(root='test',
                                           transform=data_transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size, shuffle=True,num_workers=4)

# get some random training images
dataiter = iter(trainloader)
testiter = iter(testloader)
images, labels = dataiter.next()


criterion = nn.CrossEntropyLoss()

best = 1.0
for epoch in range(epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            with open(outfile, "a") as f:
                f.write("epoch: {}\tbatches: {}\ttraining_loss: {}\n".format(epoch+1, i, running_loss/200))
            running_loss = 0.0
    
    running_loss = 0.0
    for i, data in enumerate(testloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
    with open(outfile, "a") as f:
        f.write("epoch: {}\ttest_lost: {}\n".format(epoch+1, running_loss/i))
    if (running_loss/i) <= best:
        torch.save(model.state_dict(), os.getcwd()+"/models/read_sflabel_{}.pt".format(str(epoch)+"_" + version + str(batch_size) + "_" + optimization_alg))

print('Finished Training')

#torch.save(model.state_dict(), os.getcwd()+"/models/read_sflabel_100.pt")

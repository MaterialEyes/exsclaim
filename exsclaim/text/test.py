import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets, utils
import architectures as a
import argparse

## Parse command line arguments

ap = argparse.ArgumentParser()
ap.add_argument("-a", "--architecture", type=str, help="Name of nn.Module subclass to use")
ap.add_argument("-b", "--batch-size", type=int, default=50)
ap.add_argument("-m", "--model", type=str, help="path to pytorch model")
args = vars(ap.parse_args())

architecture = args["architecture"]
batch_size = args["batch_size"]
model_path = args["model"]

## Utility Functions 

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if architecture == "CNN1":
    model = a.CNN1()
elif architecture == "CNN2":
    model = a.CNN2()
else:
    model = a.CNN1()

model.load_state_dict(torch.load(os.getcwd()+model_path))

data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


testset= datasets.ImageFolder(root='test',transform=data_transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size,
                                         shuffle=True,num_workers=4)

dataiter = iter(testloader)
images, labels = dataiter.next()


correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the modelwork on the %d test images: %d %%' % (len(testset),
    100 * correct / total))

"""
class_correct = list(0. for i in range(16))
class_total = list(0. for i in range(16))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(5):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(16):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

"""

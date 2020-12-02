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
import time
import cv2

def random_guassian_blur(image):
    image = np.array(image)
    if True:
        image_blur = cv2.GaussianBlur(image,(15,15),10)
        new_image = image_blur
        return new_image
    return image

def load_split_train_test(datadir):
    normalize_transform = transforms.Compose([transforms.Resize((224, 224)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),])
    blur_transform = transforms.Compose([transforms.Resize((224, 224)),
                                            transforms.Lambda(random_guassian_blur),
                                           transforms.ToTensor()])
    resize_transform = transforms.Compose([transforms.Resize((224, 224)),
                                           transforms.ToTensor()])       
    train_data = datasets.ImageFolder(datadir + "/train", transform=blur_transform)
    test_data = datasets.ImageFolder(datadir + "/test", transform=resize_transform)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True)
    return trainloader, testloader

def get_model(size, dataset_name, classes, pretrained):
    model = None
    trained = True if pretrained == "pretrained" else False
    if size == 50:
        model = models.resnet50(pretrained=trained)
    elif size == 18:
        model = models.resnet18(pretrained = trained)
    elif size == 152:
        model = models.resnet152(pretrained=trained)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    if size == 18:  
        model.fc = nn.Sequential(   nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(512, classes),
                                    nn.LogSoftmax(dim=1))
    else:
        model.fc = nn.Sequential(   nn.Linear(2048, 512),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(512, classes),
                                    nn.LogSoftmax(dim=1))

    model.to(device)
   
    # find previous model to resume training
    largest = -1
    best_checkpoint = None
    for checkpoint in os.listdir('checkpoints/{}'.format(pretrained)):
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
        return model, ""

    return model, best_checkpoint


def main(dataset_name, classes, model_size, save_frequency, pretrained, learning_rate):
    ## Start Tracking Time
    t0 = time.process_time()

    ## Load datasets
    trainloader, testloader = load_split_train_test("~/exsclaim/dataset/dataset_generation/{}".format(dataset_name))
    
    ## Get model checkpoint
    model, checkpoint_path = get_model(model_size, dataset_name, classes, pretrained)
    criterion = nn.NLLLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=learning_rate)
    # Dynamically reduce learning rate as testing loss stops improving
    learning_rate_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    if checkpoint_path != "":
        # Load saved information from checkpoint
        checkpoint_path = 'checkpoints/{}/'.format(pretrained) + checkpoint_path
        cuda = torch.cuda.is_available() and (gpu_id >= 0)
        if cuda:
            checkpoint = torch.load(checkpoint_path)
            model = model.cuda()
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

        current_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        learning_rate_scheduler.load_state_dict(checkpoint["learning_rate_state_dict"])
        best_accuracy = checkpoint["accuracy"]
    else:
        current_epoch = 0
        best_accuracy = 0
 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ## Train model
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
        # update learning rate based on test loss
        learning_rate_scheduler.step(test_loss)      
        
        ## Save results until time to write to file
        recent_train_losses.append(running_loss/len(trainloader))
        recent_test_losses.append(test_loss/len(testloader))     
        accuracies.append(float(accuracy/len(testloader)))
        unsaved_epochs.append(epoch)

        ## Prepare to resume training
        running_loss = 0
        model.train()

        ## Check if we have enough time to continue
        t1 = time.process_time()
        time_per_epoch = (t1 - t0) / float(epoch - current_epoch + 1)
        print(time_per_epoch)
        save = (21600 - (t1 - t0) < time_per_epoch)

        learning_rate = optimizer.param_groups[0]['lr']
        if learning_rate < 0.00000001:
            save = True

        if epoch % (save_frequency) == 0 or accuracy > best_accuracy or save:
            best_accuracy = accuracy
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'learning_rate_state_dict': learning_rate_scheduler.state_dict(),
                    'accuracy': accuracy
                    }, "checkpoints/{}/{}_{}-{}.pt".format(pretrained, dataset_name, model_size, epoch))
        
            with open("results/{}/{}_{}.txt".format(pretrained, dataset_name, model_size), "a") as f:
                for i in range(len(unsaved_epochs)): 
                    f.write(("Epoch {}/{}.. ".format(unsaved_epochs[i], epochs) +
                        "Train loss: {}.. ".format(recent_train_losses[i]) +
                        "Test loss: {}.. ".format(recent_test_losses[i]) +
                        "Test accuracy: {}.. ".format(accuracies[i]) +
                        "Learning Rate: {}\n".format(learning_rate)))

                if learning_rate < 0.00000001:
                    f.write("Training complete!")
                    return
            unsaved_epochs = []
            recent_test_losses, recent_train_losses = [], []
            accuracies = []

if __name__ == "__main__":
    dataset_to_classes = {"unit_data":  4,
                          "some":       69,
                          "all":        117,
                          "scale_all":  39,
                          "scale_some": 23
                         }
     
    dataset_to_save_frequency = {"unit_data":  50,
                                  "some":       40,
                                  "all":        30,
                                  "scale_all":  30,
                                  "scale_some": 40
                                 }
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
    ap.add_argument("-p", "--pretrained", default=True, action='store_false')
    ap.add_argument("-l", "--learning_rate", default = 0.01, type=int)   
 
    args = vars(ap.parse_args())

    dataset_name = args["dataset_name"]
    classes = args["classes"]
    classes = dataset_to_classes[dataset_name]
    save_frequency = args["save_frequency"]
    save_frequency = dataset_to_save_frequency[dataset_name]
    model_size = args["model_size"]
    pretrained = args["pretrained"]
    learning_rate = args["learning_rate"]

    if pretrained:
        pretrained = "pretrained"
    else:
        pretrained = "scratch"
    main(dataset_name, classes, model_size, save_frequency, pretrained, learning_rate)

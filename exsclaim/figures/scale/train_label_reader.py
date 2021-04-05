import argparse
import json
import os
import pathlib
from operator import itemgetter

import numpy as np
import torch
from PIL import Image
from pytorch_model_summary import summary
from torch import nn, optim
from torchvision import transforms

from ..models.crnn import CRNN
from .ctc import ctcBeamSearch
from .dataset import ScaleLabelDataset
from .lm import LanguageModel


def convert_to_rgb(image):
    return image.convert("RGB")

def load_data(batch_size, input_height, input_width, text="random_separate"):
    normalize_transform = transforms.Compose([
        transforms.GaussianBlur((3,3), sigma=(0.1, 2.0)),
        transforms.Resize((input_height, input_width)),
        transforms.Lambda(convert_to_rgb),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])
    resize_transform = transforms.Compose([transforms.Resize((32, 128)),
                                           transforms.Lambda(convert_to_rgb),
                                           transforms.ToTensor()])       
    train_data = ScaleLabelDataset(transforms=normalize_transform, text=text)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return trainloader, trainloader

def get_model(checkpoint_directory, model_name):
    best_model = None
    highest_epoch = 0
    for checkpoint in os.listdir(checkpoint_directory):
        checkpoint_name, epoch = checkpoint.split(".")[0].split("-")
        epoch = int(epoch)
        if checkpoint_name == model_name and epoch > highest_epoch:
            best_model = checkpoint
            highest_epoch = epoch
    if best_model is None:
        return None
    return checkpoint_directory / best_model

def train_one_epoch(model,
                    epoch,
                    criterion, 
                    optimizer,
                    lr_scheduler, 
                    trainloader,
                    testloader,
                    results_file,
                    checkpoint_directory,
                    model_name,
                    save_every,
                    print_every,
                    best_loss):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batches = 0
    for inputs, labels in trainloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        target_lengths = []
        for label in labels:
            label_list = label.tolist()
            target_lengths.append(label_list.index(21))
        target_lengths = torch.tensor(target_lengths, dtype=torch.int8, device=device)
        input_lengths = torch.tensor([32]*len(target_lengths), device=device)
        predictions = model(inputs)
        predictions = predictions.permute(1, 0, 2)
        loss = criterion(predictions, labels, input_lengths, target_lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step(loss)
        batches += 1
        if batches % print_every == 0:
            with open(results_file, "a+") as f:
                f.write("\tBatch: {}, Loss: {}, Learning Rate: {}\n".format(batches, loss, optimizer.param_groups[0]["lr"]))
    # test results
    running_loss = 0
    i = 0
    for inputs, labels in testloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        if i > 1000:
            break
        target_lengths = []
        for label in labels:
            label_list = label.tolist()
            target_lengths.append(label_list.index(21))
        target_lengths = torch.tensor(target_lengths, dtype=torch.int8, device=device)
        input_lengths = torch.tensor([32]*len(target_lengths), device=device)
        predictions = model(inputs)
        predictions = predictions.permute(1, 0, 2)
        loss = criterion(predictions, labels, input_lengths, target_lengths)
        running_loss += float(loss)
    if running_loss < best_loss:
        best_loss = running_loss
    with open(results_file, "a+") as f:
        f.write("Epoch: {}, Running Loss: {}, Learning Rate: {}\n".format(epoch, running_loss, optimizer.param_groups[0]["lr"]))
    # save checkpoint
    if epoch % save_every == 0:
        torch.save({
                "epoch":    epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_state_dict": lr_scheduler.state_dict(),
                "best_loss": best_loss
            }, checkpoint_directory / (model_name + "-{}.pt".format(epoch)))
    return best_loss

def train_crnn(batch_size=32,
          epochs=2000,
          learning_rate=0.001,
          optimizer="SGD",
          lr_scheduler="plateau",
          loss_function="CTC",
          print_every=50,
          save_every=5,
          input_channels=3,
          output_classes=22,
          cnn_to_rnn=0,
          model_name="test",
          input_height=128,
          input_width=512,
          sequence_length=32,
          recurrent_type="bi-lstm",
          cnn_kernel_size=(3,3),
          convolution_layers=4,
          hard_set_lr = None,
          configuration = None,
          text="random_separate"):
    """ trains model """
    current_file = pathlib.Path(__file__).resolve(strict=True)
    exsclaim_root = current_file.parent.parent.parent.parent
    checkpoint_directory = exsclaim_root / "training" / "checkpoints"
    results_file = exsclaim_root / "training" / "results" / (model_name + ".txt")
    # Load CRNN model and assign optimizer, lr_scheduler
    model = CRNN(input_channels=input_channels,
                 output_classes=output_classes,
                 convolution_layers=convolution_layers,
                 cnn_kernel_size=cnn_kernel_size,
                 recurrent_type=recurrent_type,
                 input_height=input_height,
                 input_width=input_width,
                 sequence_length=sequence_length,
                 configuration=configuration)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with open(results_file, "a+") as f:
        f.write(summary(
            model, 
            torch.zeros((1, 3, input_height, input_width)).to(device),
            show_input=False
        ))
        f.write("\n")
    # set up training hyper parameters
    optimizers = {
        "adam" : optim.Adam(model.parameters(), lr=learning_rate),
        "SGD" : optim.SGD(model.parameters(), lr=learning_rate)
    }
    optimizer = optimizers[optimizer]
    best_checkpoint = get_model(checkpoint_directory, model_name)
    if best_checkpoint is not None:
        cuda = torch.cuda.is_available()
        if cuda:
            checkpoint = torch.load(best_checkpoint)
            model = model.cuda()
        else:
            checkpoint = torch.load(best_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        current_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
    else:
        current_epoch = 0
        best_loss = 999999999999

    # hard set learning rate
    if hard_set_lr is not None:
        optimizer.param_groups[0]["lr"] = hard_set_lr
    # dict of lr_schedulers after optimizer state dict is loaded since
    # lr_schedulers take optimizer as a parameter
    lr_schedulers = {
        "plateau":  optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=2000, threshold=0.000001, factor=0.5
        )
    }
    lr_scheduler = lr_schedulers[lr_scheduler]
    if best_checkpoint is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_state_dict"])
    # Set loss function, load data and begin training
    loss_functions = {
        "CTC":      nn.CTCLoss(blank=21, zero_infinity=True),
        "NLL":      nn.NLLLoss()
    }
    criterion = loss_functions[loss_function]
    label_trainloader, label_testloader = load_data(batch_size, input_height, input_width, text)
    character_trainloader, character_testloader = load_data(batch_size, input_height, input_width, text)
    # Set rnn to untrainable
    for name, param in model.named_parameters():
        if "Recurrent" not in name:
            param.requires_grad = False
    for epoch in range(current_epoch+1, int(cnn_to_rnn*epochs)):
        best_loss = train_one_epoch(model, epoch, criterion, optimizer, lr_scheduler,
                        character_trainloader, character_testloader,
                        results_file, checkpoint_directory, model_name, save_every, print_every,
                        best_loss)
    try:
        current_epoch = epoch
    except:
        pass     
    # train full model
    print("\nStarting full model training\n")
    for param in model.parameters():
        param.requires_grad = True
    for epoch in range(current_epoch+1, epochs):
        best_loss = train_one_epoch(model, epoch, criterion, optimizer, lr_scheduler,
                        label_trainloader, label_testloader,
                        results_file, checkpoint_directory, model_name, save_every, print_every,
                        best_loss)


### DECODER FUNCTIONS ###

def ctc_decoders(beamWidth, constrict_search, lm, postprocess):
    classes = "0123456789mMcCuUnN .A"
    idx_to_class = classes + "-"
    def ctc_search(matrix):
        ctc_inputs = torch.exp(matrix)
        ctc_inputs = ctc_inputs.squeeze(0)
        results = ctcBeamSearch(ctc_inputs,
                             classes=classes,
                             lm=lm,
                             beamWidth=beamWidth,
                             constrict_search=constrict_search)
        if postprocess:
            word = postprocess_ctc(results)
        else:
            word = ""
            for step in results[0]:
                word += idx_to_class[step]
        return word
    return ctc_search

def is_number(n):
    try:
        float(n)
        return True
    except:
        return False

def path_to_word(path, idx_to_class):
    word = ""
    for step in path[1:]:
        word += idx_to_class[step[0]]
    word = "".join(word.split("-"))
    word = "".join(word.split(" "))
    for i in range(len(word)):
        if not is_number(word[:i+1]):
            break
    return str(float(word[:i])) + " " + word[i:].lower()

def postprocess_ctc(results):
    classes = "0123456789mMcCuUnN .A"
    idx_to_class = classes + "-"
    for result in results:
        word = ""
        for step in result:
            word += idx_to_class[step]
        word = word.strip()
        word = "".join(word.split("-"))
        try:
            number, unit = word.split(" ")
            number = float(number)
            if unit.lower() not in ["nm", "mm", "cm", "um", "a"]:
                continue
            return word
        except:
            continue
    return None

def decode(logits, beam_width):
    legal_next_steps = create_rules()
    start_idx = 22 
    potential_paths = [[(start_idx, 0)]]
    for timestep in logits:
        new_potential_paths = []
        for path in potential_paths:
            current_idx = path[-1][0]
            legal_idxs = valid_next_char(path, 8)
            next_steps = find_n_best_legal_moves(legal_idxs, timestep, beam_width)        
            for next_step in next_steps:
                new_potential_paths.append(path + [next_step])
        #new_potential_paths += potential_paths
        sorted_potential_paths = sorted(new_potential_paths, key=lambda x : score_candidate(x), reverse=True)
        potential_paths = sorted_potential_paths[:beam_width]
    final_rankings = sorted(potential_paths, key=lambda x : score_candidate(x, True), reverse=True)
    return final_rankings

def find_n_best_legal_moves(legal_idxs, next_timestep, n):
    moves = []
    for legal_idx in legal_idxs:
        moves.append((legal_idx, next_timestep[legal_idx]))
    ranked_moves = sorted(moves, key=itemgetter(1), reverse=True)
    return ranked_moves[:n]

def score_candidate(path, is_final=False):
    """ path is list of (label, logp) tuples """
    units = 0
    nonzero_digits = 0
    decimals = 0
    score = 0
    for label, logp in path:
        if label in [1,2,3,4,5,6,7,8,9]:
            nonzero_digits += 1
        elif label in [10,11,12,13,14,15,16,17]:
            units += 1
        # Angstrom is standalone unit
        elif label == 20:
            units += 2
        elif label == 19:
            decimals += 1
        # score for ordering
        if label in [0,1,2,3,4,5,6,7,8,9] and units > 0:
            return -100
        score = logp
    if (units > 2
            or (units > 0 and nonzero_digits == 0)):
        score = -100 
    if not is_final:
        return score
    # change score to 0 for rule violations
    if (units != 2 
            or nonzero_digits not in [1,2,3,4,5]
            or decimals > 1):
        score = -100
    return score


def valid_next_char(path, sequence_length):
    path_length = len(path)
    spots_left = sequence_length - path_length + 1
    if path_length == 1:
        return [1,2,3,4,5,6,7,8,9]
    prefix = False
    base_unit = False
    nonzero_digits = 0
    digits = 0
    decimals = 0

    for label, logp in path: 
        if label in [1,2,3,4,5,6,7,8,9]:
            nonzero_digits += 1
            digits += 1
        elif label == 0:
            digits += 1
        elif label == 19:
            decimals += 1
        elif label in [10,11,12,13,14,15,16,17] and not prefix:
            prefix = True
        elif label == 20:
            prefix = True
            base_unit = True
        elif label in [10, 11] and prefix:
            base_unit = True
        elif label in [18, 21, 22]:
            continue
        else:
            print("How did I get here?\nThe path is: ", path)
    
    # unit has been started, no digits or decimals allowed
    if prefix:
        # unit has not been finished, no prefixes allowed
        if not base_unit:
            # only one spot left, must finish unit
            if spots_left == 1:
                return [10, 11]
            else:
                return [10, 11, 18, 21] 
        # unit has been finished, only blanks left
        else:
            return [18, 21]
    # unit has not been started
    # decimal must be followed by a digit
    if label == 19:
        return [0,1,2,3,4,5,6,7,8,9,18,21]
    # if unit hasn't started and only one spot left, must be A
    if spots_left == 1:
        return [20]
    elif spots_left == 2:
        # current label is a space, can go right into unit
        if label in [18,21]:
            return [10,11,12,13,14,15,16,17,18,20,21]
        else:
            return [18, 21]
    # more than 2 spots left
    # if last spot is not a blank, must be followed by more numbers or spaces
    if label not in [18, 21]:
        if decimals == 1:
            return [0,1,2,3,4,5,6,7,8,9,18,21]
        else:
            return [0,1,2,3,4,5,6,7,8,9,19,18,21]
    # last spot is blank, can be followed by anything
    if decimals == 1:
        return [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21]
    else:
        return [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
        
def create_rules(): 
    classes = "0123456789mMcCuUnN .A-"
    # digits can be followed by other digits, decimal points, spaces, and blanks
    legal_next_chars = {i : [0,1,2,3,4,5,6,7,8,9,18,19,21] for i in range(0,10)}
    # a space can be followed by unit characters and blanks
    legal_next_chars[18] = [10,11,12,13,14,15,16,17,18,20,21]
    # a decimal point can be followed by digits and blanks
    legal_next_chars[19] = [0,1,2,3,4,5,6,8,9,18,21]
    # A can only be followed by a blank
    legal_next_chars[20] = [18,21]
    # non-A units can only be followed by a blank or an m
    for i in range(10, 18):
        legal_next_chars[i] = [10, 11,18, 21]
    # a blanks can be followed by all
    legal_next_chars[21] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    legal_next_chars[22] = [0,1,2,3,4,5,6,7,8,9,18,21]
    return legal_next_chars

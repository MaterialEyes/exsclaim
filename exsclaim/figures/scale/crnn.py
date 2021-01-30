import json
import os
from torch import optim, nn, utils
from torchvision import datasets, transforms, models
import numpy as np
import torch
import pathlib
from PIL import Image
from pytorch_model_summary import summary
import argparse
from operator import itemgetter
from .ctc import ctcBeamSearch
from .lm import LanguageModel

class ScaleLabelDataset():

    def make_encoding(self, label):
        max_length = 8
        char_to_int = {
            "0":    0,
            "1":    1,
            "2":    2,
            "3":    3,
            "4":    4,
            "5":    5,
            "6":    6,
            "7":    7,
            "8":    8,
            "9":    9,
            "m":    10,
            "M":    11,
            "c":    12,
            "C":    13,
            "u":    14,
            "U":    15,
            "n":    16,
            "N":    17,
            " ":    18,
            ".":    19,
            "A":    20,
            "empty": 21
        }
        target = torch.zeros(max_length)
        for i in range(max_length):
            try:
                character = label[i]
                number = char_to_int[character]
            except:
                number = 21
            target[i] = number
        return target            

    def __init__(self, root, transforms, test=True):
        self.root = root
        self.transforms = transforms
        if test:
            scale_bar_dataset = os.path.join(root, "test")
        else:
            scale_bar_dataset = os.path.join(root, "train")

        self.image_paths = []
        for label in os.listdir(scale_bar_dataset):
            label_folder = os.path.join(scale_bar_dataset, label)
            for image in os.listdir(label_folder):
                image_path = os.path.join(label_folder, image)
                self.image_paths.append(image_path)
  
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image_transformed = self.transforms(image)
        image.close()
        label = image_path.split("/")[-2]
        target = self.make_encoding(label)
        return image_transformed, target

    def __len__(self):
        return len(self.image_paths)

class CRNN(nn.Module):

    def __init__(self, 
                 input_channels=3,
                 output_classes=22,
                 convolution_layers=4,
                 cnn_kernel_size=(3,3),
                 in_channels=None,
                 max_pooling=None,
                 batch_normalization=None,
                 dropout=None,
                 activation_type="relu",
                 recurrent_layers=2,
                 recurrent_type="bi-lstm",
                 input_height=32,
                 input_width=128,
                 sequence_length=8):
        """ Initialize a Convolutional Recurrent Neural Network

        Args:
            input (int): Number of input channels. Default: 3 (for RGB images)
            output (int): Number of output channels/classes. Default: 22
            convolution_layers (int): Number of Conv2d layers in architecture
            cnn_kernal_size (tuple): (h, w) of kernel applied in each cnn layer
            in_channels (list): Number on input channels for each respective
                convolutional layer. Must be None or a list of length of
                convolution_layers. If None, defaults to in_channels defined
                below.
            batch_normalization (list): list of ints, where each int is a
                convultional layer after which batch normalization should
                occur. Each element of the list should be 
                <= convolution_layers. If False, no batch normalization will
                occur.
            max_pooling (list): list of ints, where each int is a
                convultional layer after which max pooling should occur. Each
                element of the list should be <= convolution_layers. If False,
                no max_pooling will occur.
            dropout (list): list of ints, where each int is a
                convultional layer after which batch normalization should
                occur. Each element of the list should be 
                <= convolution_layers. If False, no dropout will occur.
            activation_type (string): Activation function to be applied after
                each convolutional layer. Default: relu
            recurrent_type (string): Type of RNN to be used. Default: bi-lstm
            recurrent_layers (int): Number of recurrent layers
            input_height (int): expected height of input images. Default: 16
            input_width (int): expected width of input images. Default: 128
            sequence_length (int): Output width of CNN and input sequence
                length to RNN (time steps). Default: 8.
        """
        super(CRNN, self).__init__()
        activation_functions = {
            "relu":         nn.ReLU(),
            "leaky_relu":   nn.LeakyReLU(),
            "tanh":         nn.Tanh()
        }
        max_pooling_kernel = (3, 3)
        max_pooling_stride = (2, 2)
        max_pooling_padding = (max_pooling_kernel[0] // 2, max_pooling_kernel[1] // 2)
        current_dims = (input_height, input_width)
        activation_function = activation_functions[activation_type.lower()]
        if in_channels is None:
            if convolution_layers < 6:
                in_channels = [64, 128, 256, 256, 512, 512, 512]
            else:
                in_channels = [max(2**(i // 3 + 5), 2014)
                               for i in range(convolution_layers)]
        if batch_normalization is None:
            batch_normalization = [i for i in range(0, convolution_layers, 2)]
        if dropout is None:
            dropout = []
        if max_pooling is None:
            max_pooling = list(np.linspace(0, convolution_layers - 1,
                                3).astype(int))
        in_channels = [input_channels] + in_channels
        # Build the CNN layers
        cnn = nn.Sequential()
        for layer in range(convolution_layers):
            in_channel = in_channels[layer]
            out_channel = in_channels[layer + 1]
            padding = (cnn_kernel_size[0] // 2, cnn_kernel_size[1] // 2)
            convolution_layer = nn.Conv2d(in_channel, out_channel, cnn_kernel_size, padding=padding)
            cnn.add_module("Convolution {}".format(layer), convolution_layer)
            if layer in max_pooling:
                cnn.add_module("Max Pooling {}".format(layer),
                               nn.MaxPool2d(max_pooling_kernel, max_pooling_stride, max_pooling_padding))
                current_dims = max_pooling_output_dim(current_dims,
                                                       max_pooling_kernel,
                                                       max_pooling_padding,
                                                       max_pooling_stride)
            if layer in batch_normalization:
                cnn.add_module("Batch Nomralization {}".format(layer),
                               nn.BatchNorm2d(out_channel))
            cnn.add_module("Activation {}".format(layer), activation_function)
            if layer in dropout:
                cnn.add_module("Dropout {}".format(layer), nn.Dropout2d(0.5))
        # Add a Max Pooling layer to get the height of the image to 1 and
        # the width to the desired sequence length
        divisor = int(current_dims[1] / sequence_length)
        kernel = (current_dims[0] + 1, max(divisor + 1, 3))
        padding = (current_dims[0] // 2, max(divisor // 2, 1))
        stride = (current_dims[0], divisor)
        cnn.add_module("Max Pooling Last {}".format(divisor),
            nn.MaxPool2d(kernel, stride, padding))
        current_dims = max_pooling_output_dim(current_dims, kernel, padding, stride)
        hidden = 256
        recurrent_types = {
            "bi-lstm": nn.LSTM(out_channel, hidden, bidirectional=True),
            "lstm": nn.LSTM(out_channel, hidden, bidirectional=False)
        }
        # Build the RNN layers
        rnn = nn.Sequential()
        rnn.add_module("Recurrent Layer {}".format(layer+1),
                        recurrent_types[recurrent_type])
        if recurrent_type == "bi-lstm":
            hidden_out = hidden*2
        else:
            hidden_out = hidden
        self.fc = nn.Sequential()
        self.fc.add_module("Fully Connected", nn.Linear(hidden_out, output_classes))
        self.rnn = rnn
        self.cnn = cnn

    def forward(self, input):
        cnn_output = self.cnn(input)
        batch_size, channels, height, width = cnn_output.size()
        assert height == 1
        # eliminate height dimension
        cnn_output = cnn_output.squeeze(2) # batch, channels, width
        # Format input to rnn to sequence, batch, channels because
        # that is expected input for RNN (unless batch_first=True)
        # In PyTorch LSTM nomenclature, width is sequence length and 
        # channels is input size
        cnn_output = cnn_output.permute(2, 0, 1) # width, batch, channels
        rnn_output, _ = self.rnn(cnn_output) # width, batch, channels
        # Reformat shape for fully connected layer
        rnn_output = rnn_output.permute(1, 0, 2) # batch, width/seq, channels
        output = self.fc(rnn_output) # batch, width/seq, classes
        output = nn.LogSoftmax(dim=2)(output) # batch, width/seq, classes

        return output

def max_pooling_output_dim(dimension, kernel, padding, stride, dilation=(1,1)):
    output_dims = []
    for i in range(0, 2):
        numerator = dimension[i] + 2*padding[i] - dilation[i] * (kernel[i] -1) - 1
        output = int(float(numerator) / stride[i] + 1)
        output_dims.append(output)
    return output_dims

def convert_to_rgb(image):
    return image.convert("RGB")

def load_data(datadir, batch_size):
    normalize_transform = transforms.Compose([transforms.Resize((32, 128)),
                                           transforms.Lambda(convert_to_rgb),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),])
    resize_transform = transforms.Compose([transforms.Resize((32, 128)),
                                           transforms.Lambda(convert_to_rgb),
                                           transforms.ToTensor()])       
    train_data = ScaleLabelDataset(datadir, transforms=resize_transform, test=False)
    test_data = ScaleLabelDataset(datadir, transforms=resize_transform, test=True)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return trainloader, testloader

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
    batches = 0
    for inputs, labels in trainloader:
        target_lengths = []
        for label in labels:
            label_list = label.tolist()
            target_lengths.append(label_list.index(21))
        target_lengths = torch.tensor(target_lengths, dtype=torch.int8)
        input_lengths = target_lengths 
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
                f.write("\tBatch {}: Loss {}\n".format(batches, loss))
    # test results
    running_loss = 0
    for inputs, labels in testloader:
        target_lengths = []
        for label in labels:
            label_list = label.tolist()
            target_lengths.append(label_list.index(21))
        target_lengths = torch.tensor(target_lengths, dtype=torch.int8)
        input_lengths = target_lengths 
        predictions = model(inputs)
        predictions = predictions.permute(1, 0, 2)
        loss = criterion(predictions, labels, input_lengths, target_lengths)
        running_loss += loss
    if running_loss < best_loss:
        best_loss = running_loss
    with open(results_file, "a+") as f:
        f.write("Epoch {}: Running Loss: {}\n".format(epoch, running_loss))
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

def train(batch_size=32,
          epochs=600,
          learning_rate=0.001,
          optimizer="adam",
          lr_scheduler="plateau",
          loss_function="CTC",
          print_every=250,
          save_every=2,
          input_channels=3,
          output_classes=22,
          cnn_to_rnn=0,
          model_name="test",
          input_height=32,
          input_width=128,
          sequence_length=8,
          recurrent_type="bi-lstm",
          cnn_kernel_size=(3,3),
          convolution_layers=4):
    """ trains model """
    current_file = pathlib.Path(__file__).resolve(strict=True)
    exsclaim_root = current_file.parent.parent.parent.parent
    label_directory = exsclaim_root / 'dataset' / 'dataset_generation' / 'samples'
    character_directory = exsclaim_root / 'dataset' / 'dataset_generation' / 'number_samples'
    checkpoint_directory = current_file.parent / "checkpoints" / "label"
    results_file = current_file.parent / "results" / "label" / (model_name + ".txt")
    # Load CRNN model and assign optimizer, lr_scheduler
    model = CRNN(input_channels=input_channels,
                 output_classes=output_classes,
                 convolution_layers=convolution_layers,
                 cnn_kernel_size=cnn_kernel_size,
                 recurrent_type=recurrent_type,
                 input_height=input_height,
                 input_width=input_width,
                 sequence_length=sequence_length)
    print(summary(model, torch.zeros((1, 3, 32, 128)), show_input=False))
    optimizers = {
        "adam" : optim.Adam(model.parameters(), lr=learning_rate)
    }
    optimizer = optimizers[optimizer]
    lr_schedulers = {
        "plateau":  optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    }
    lr_scheduler = lr_schedulers[lr_scheduler]
    best_checkpoint = get_model(checkpoint_directory, model_name)
    if best_checkpoint is not None:
        cuda = torch.cuda.is_available() and (gpu_id >= 0)
        if cuda:
            checkpoint = torch.load(best_checkpoint)
            model = model.cuda()
        else:
            checkpoint = torch.load(best_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_state_dict"])
        current_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
    else:
        current_epoch = 0
        best_loss = 999999999999
    # Set loss function, load data and begin training
    loss_functions = {
        "CTC":      nn.CTCLoss(blank=21, zero_infinity=True),
        "NLL":      nn.NLLLoss()
    }
    criterion = loss_functions[loss_function]
    label_trainloader, label_testloader = load_data(label_directory, batch_size)
    character_trainloader, character_testloader = load_data(character_directory, batch_size)
    # Set rnn to untrainable
    for name, param in model.named_parameters():
        if "Recurrent" in name:
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


def run_model(model_name="test",
              language_model_file=None,
              decoder=0,
              search_width=100):
    model = CRNN(3, 22)
    current_file = pathlib.Path(__file__).resolve(strict=True)
    exsclaim_root = current_file.parent.parent.parent.parent
    data_directory = exsclaim_root / 'dataset' / 'dataset_generation' / 'samples'
    checkpoint_directory = current_file.parent / "checkpoints" / "label"
    best_checkpoint = get_model(checkpoint_directory, model_name)
    if best_checkpoint is not None:
        cuda = torch.cuda.is_available() and (gpu_id >= 0)
        if cuda:
            checkpoint = torch.load(best_checkpoint)
            model = model.cuda()
        else:
            checkpoint = torch.load(best_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    resize_transform = transforms.Compose([transforms.Resize((32, 128)),
                                           transforms.Lambda(convert_to_rgb),
                                           transforms.ToTensor()])       
    
    classes = "0123456789mMcCuUnN .A"
    idx_to_class = classes + "-"
    test_image_directory = current_file.parent.parent.parent.parent / 'dataset' / 'dataset_generation' / 'scale_label_dataset'
    correct = 0
    incorrect = 0
    all_results = {}
    all_results["correct"] = []
    for image_label in os.listdir(test_image_directory):
        for image_name in os.listdir(test_image_directory / image_label):
            all_results["correct"].append(image_label)
            # open image
            image_path = test_image_directory / image_label / image_name
            image = Image.open(image_path).convert("RGB")
            image = resize_transform(image)
            image = image.unsqueeze(0)
            # run image on model
            logps = model(image)

            # run ctcs
            for language_model_file in [ None, "corpus.txt", "realistic.txt"]:
                if language_model_file is not None:
                    lm = LanguageModel(current_file.parent / language_model_file, classes)
                else:
                    lm = None
                for beamwidth in [25, 50]:
                    for constrict_search in [True,  False]:
                        for postprocess in [True, False]:
                            if constrict_search and language_model_file:
                                continue
                            decode_function = ctc_decoders(beamwidth, constrict_search, lm, postprocess)
                            word = decode_function(logps)
                            decoder_results = all_results.get((language_model_file, beamwidth, constrict_search, postprocess), [])
                            decoder_results.append(word)
                            all_results[(language_model_file, beamwidth, constrict_search, postprocess)] = decoder_results
            print("finished one")                
            if False:
                outputs = logps.squeeze(0)
                outputs = outputs.tolist()
                results = decode(outputs, 5000)
                words_dict = {}
                for result in results[:50]:
                    word = path_to_word(result, idx_to_class)
                    word_instances = words_dict.get(word, 0)
                    words_dict[word] = word_instances + torch.exp(torch.Tensor([score_candidate(result, True)]))
                
                    if word == image_label:
                        correct += 1
                    else:
                        incorrect += 1
            break
    print(all_results)

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
 
if __name__ == "__main__":
    current_file = pathlib.Path(__file__).resolve(strict=True)
    parent_directory = current_file.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--test", default=True, action="store_false")
    ap.add_argument("-m", "--model_name", type=str)
    ap.add_argument("-d", "--decoder", type=str, default="")
    args = ap.parse_args()
    if not args.test:
        if args.decoder != "":
            with open(parent_directory / "decoder_configurations.json", "r") as f:
                all_configs = json.load(f)
            config = all_configs[args.model_name]
            run_model(model_name=args.model_name,
                  language_model_file = config["language_model_file"],
                  decoder=config["decoder"],
                  search_width=config["search_width"])
        else:
            run_model(model_name=args.model_name)
    else:
        with open(parent_directory / "label_reader_configurations.json", "r") as f:
            all_configs = json.load(f)
            config = all_configs[args.model_name]
        train(batch_size = config["batch_size"],
              learning_rate = config["learning_rate"],
              cnn_to_rnn = config["cnn_to_rnn"],
              model_name = args.model_name,
              input_height = config["input_height"],
              input_width = config["input_width"],
              sequence_length = config["sequence_length"],
              recurrent_type = config["recurrent_type"],
              cnn_kernel_size = config["cnn_kernel_size"],
              convolution_layers = config["convolution_layers"])


    #run_model()






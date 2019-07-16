# TextDetector

This module reads the text of images using neural networks trained in Pytorch. The models are specifically trained for reading subfigure and scale bar labels. 

### Setup

Run setup.sh to initialize a virtualenv for this module, install its requirements, and download the default model. 

### Training

To train a model using PyTorch 
'''
python train.py [--batch-size (-b) | --architecture (-a) | --learn-rate (-l) | --gd_alg (-g)]
    batch-size: an integer, size of the batch to use during training. Defaults to 50.
    architecture: a string, name of the neural net architecture to use from architectures.py. Defaults to CNN1
    learn-rate: a float, learning rate of gradient descent algorighm. Defaults to 0.0001
    gd-alg: a string, gradient descent optimization algorithm. Defaults to SGD
'''

### Running Model

To run the model use:
'''
python test.py --model (-m) [--batch-size (-b) | --architecture (-a)]
    batch-size: an integer, size of the batch to use during training. Defaults to 50.
    architecture: a string, name of the neural net architecture to use from architectures.py. Defaults to CNN1
    model: a string, path to the PyTorch model. 

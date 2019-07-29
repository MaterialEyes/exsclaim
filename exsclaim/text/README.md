# TextDetector

This module reads the text of images using neural networks trained in Pytorch. The models are specifically trained for reading subfigure and scale bar labels. 

### Setup

Run setup.sh in the repo's root directory in interactive mode to start conda environment with the required packages and to download the default model. 
```
bash -i setup.sh
conda activate exsclaim_env
```

### Training

To train a model using PyTorch 
```
python train.py [--batch-size (-b) | --architecture (-a) | --learn-rate (-l) | --gd_alg (-g)]
    batch-size: an integer, size of the batch to use during training. Defaults to 50.
    architecture: a string, name of the neural net architecture to use from architectures.py. Defaults to CNN1
    learn-rate: a float, learning rate of gradient descent algorighm. Defaults to 0.0001
    gd-alg: a string, gradient descent optimization algorithm. Defaults to SGD
```

### Testing Model

To test the model use:
```
python test.py --model (-m) [--batch-size (-b) | --architecture (-a)]
    batch-size: an integer, size of the batch to use during training. Defaults to 50.
    architecture: a string, name of the neural net architecture to use from architectures.py. Defaults to CNN1
    model: a string, path to the PyTorch model. 
```
### Running Model
To run the model on a directory of images (note: this model only performs well on cropped scale bar and subfigure labels), run:
```
python run.py --model (-m) --architecture (-a) --input-directory (-i) 
    architecture: a string, name of the neural net architecture to use from architectures.py. Defaults to CNN1
    model_name: a string, name of the PyTorch model which should be located in models/
    input-directory: a string, path to the directory of images you wich to have transcribed
```


# EXSCLAIM!



## Description

EXSCLAIM! is a Python3 library for the **EX**traction, **S**eparation, **CL**eaning, and further **A**nnotation of **IM**ages from scientific figures. At the highest level, this library allows users to organize and utilize the large quantity of data from open, peer-reviewed scientific literature. Taking in keywords as a search query, EXSCLAIM! returns a JSON containng relevant images paired with their captions and scale bar data. 

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
    * [All Together](#alltogether)
    * [Individually](#individually)
    * [Training](#training)
- [How It Works](#howitworks)
    * [Definitions](#definitions)
    * [JSON Format](#jsonformat)
    * [Overview](#overview)
- [Credits](#credits)
- [License](#license)

## Installation <a name="installation"></a>

<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> 7b1c8c4... merged setup.sh
### How do I get EXSCLAIM! working on macOS?
[0] If conda is not installed, download an installer and follow the prompts on the installer screens:
* [Anaconda installer for macOS](https://www.anaconda.com/distribution/)
* [Miniconda installer for macOS](https://docs.conda.io/en/latest/miniconda.html)

[1] Download or clone a copy of this repository from GitLab
```sh
$ git clone https://gitlab.com/MaterialEyes/exsclaim
```
[2] Create the environment from the `env_min_osx.yaml` file and activate
```sh
$ conda env create -f env_min_osx.yaml
$ conda activate exsclaim_min
```
[3] Install select NLP and webscraping dependencies with pip
```sh
$ pip install lxml
$ pip install -U spacy
```
[4] Install `gdown` for Google Drive direct download of big files.
```sh
$ pip install gdown
```
[5] Download a small English language model to use in spaCy (NLP)
```sh
$ python -m spacy download en
```
[6] Create directories for object detection and text detection models and download current models.
```sh
$ mkdir exsclaim/text/models
$ gdown  -O exsclaim/text/models/read_sflabel_5_CNN150_adam.pt https://drive.google.com/uc?id=1pkBWn0Ss0c9TeAgsOBFqFFJZ-5ntxzpb
$ mkdir exsclaim/objects/checkpoints
$ gdown -O exsclaim/objects/checkpoints/snapshot930.ckpt.zip https://drive.google.com/uc?id=1xWxqQGDH_szfCe8eWDBwTcjzCmq7Bnf1
$ unzip exsclaim/objects/checkpoints/snapshot930.ckpt.zip -d exsclaim/objects/checkpoints
$ rm exsclaim/objects/checkpoints/snapshot930.ckpt.zip
```
<<<<<<< HEAD
>>>>>>> f4b0c75... Update README.md
=======
>>>>>>> 7b1c8c4... merged setup.sh

## Usage <a name="usage"></a> 

### All Together <a name="alltogether"></a>


### Individually <a name="individually"></a>


### Training <a name="training"></a>


## How It Works <a name="howitworks"></a>


### Definitions <a name="definitions"></a>


### JSON Format <a name="jsonformat"></a>


### Overview <a name="overview"></a>


## Credits <a name="credits"></a>


## License <a name="license"></a>

![image](https://drive.google.com/uc?export=view&id=142XkACsDxT9r9VgVg0RUsVvjJhaBqRIs)


EXSCLAIM! is a library for the automatic **EX**traction, **S**eparation, and **C**aption-based natural **L**anguage **A**nnotation of **IM**ages from scientific figures.

## Getting started

### Requirements
You need a working python 3.x installation to be able to use exsclaim, and gdown to directly download text detection and figure separation models. We highly recommend installing Anaconda, which takes care of installing Python and managing additional packages. In the following it will be assumed that you use Anaconda. Download and install it from here.

### Installing exsclaim from GitHub
- Clone this repo:
```sh
git clone https://github.com/eschwenk/exsclaim
```
- Create a new conda environment with Python 3.7 and gdown:
```sh
conda create -n exsclaim -c conda-forge python=3.7 gdown
```
- Activate this environment and navigate to the root directory:
```sh
conda activate exsclaim
cd exsclaim
```
- Download text detection and figure separation models:
```sh
./bin/download_models.sh
```
- Install with pip:
```sh
pip install .
```
- Finally, download the best-matching model for the spaCy installation:
```sh
python -m spacy download en_core_web_sm
```

Check out the project [wiki](https://gitlab.com/MaterialEyes/exsclaim/wikis/home) for more details!

## Citation
If you find this code useful, please consider citing our [paper](#paper)
```sh
@article{,
  title={},
  author={},
  journal={},
  volume={},
  number={},
  pages={},
  year={},
  publisher={}
}
```

## Acknowledgements <a name="credits"></a>

## License <a name="license"></a>

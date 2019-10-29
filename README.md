![image](https://drive.google.com/uc?export=view&id=1RC81zqSoIirVwQcPfvi0X5V-nMyD32aJ)
## __EXSCLAIM!__    [[wiki](https://gitlab.com/MaterialEyes/exsclaim/wikis/home)]  [[paper](#paper)]
EXSCLAIM! is a Python3 library for the **EX**traction, **S**eparation, **CL**eaning, and further **A**nnotation of **IM**ages from scientific figures. This library allows users to organize and utilize the large quantity of imaging data from open, peer-reviewed scientific literature. Taking in keywords as a search query, EXSCLAIM! returns a JSON containing relevant images paired with their associated caption text and scale bar data. 

## Getting started
### Installation
- If conda is not installed, download an installer and follow the prompts on the installer screens:
    * [Anaconda installer for macOS](https://www.anaconda.com/distribution/)
    * [Miniconda installer for macOS](https://docs.conda.io/en/latest/miniconda.html)
<br/><br/>
- Clone this repo:
```sh
git clone https://github.com/eschwenk/exsclaim
```
- Use the _source_ command to run [setup.sh](https://github.com/eschwenk/exsclaim-prerelease/blob/master/setup.sh)
```sh
# Current stable release
source setup.sh

# Developer mode (enables additional model training/testing tools)
source setup.sh dev
```
- A successful execution of the [setup.sh](https://github.com/eschwenk/exsclaim-prerelease/blob/master/setup.sh) script will create and _activate_ a conda environment, as well as download all relevant models. For future sessions, it is only necessary to _activate_ the conda environment before using the __EXSCLAIM!__ tools.
```sh
# To activate this environment, use
conda activate exsclaim

# To deactivate an active environment, use
conda deactivate
```

Check out the project [wiki](https://gitlab.com/MaterialEyes/exsclaim/wikis/home) for more details!

# Citation
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

# Acknowledgements <a name="credits"></a>

# License <a name="license"></a>

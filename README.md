![image](https://drive.google.com/uc?export=view&id=142XkACsDxT9r9VgVg0RUsVvjJhaBqRIs)

Automatic **EX**traction, **S**eparation, and **C**aption-based natural **L**anguage **A**nnotation of **IM**ages from scientific figures 
[[wiki](https://github.com/MaterialEyes/exsclaim/wiki)] [[paper](https://arxiv.org/abs/2103.10631)]


## 🤔 Consider Collaboration

If you find this tool or any of its derived capabilities useful, please consider registering as a user of Center for Nanoscale Materials. We will keep you posted of latest developments, as well as opportunities for computational resources, relevant data, and collaboration. Please contact Maria Chan (mchan@anl.gov) for details.

## Introduction

EXSCLAIM! is a Python package that can be used for the automatic generation of datasets of labelled images from published papers. It in three main steps: 
1. [JournalScraper](https://github.com/MaterialEyes/exsclaim/wiki/JournalScraper): scrap journal websites, acquiring figures, captions, and metadata
2. [CaptionDistributor](https://github.com/MaterialEyes/exsclaim/wiki/JournalScraper): separate figure captions into the component chunks that refer to the figure's subfigures
3. [FigureSeparator](https://github.com/MaterialEyes/exsclaim/wiki/JournalScraper): separate figures into subfigures, detect scale information, label, and type of image

You can use exsclaim as a:
- direct python import
```
from exsclaim.pipeline import Pipeline
test_pipeline = Pipeline(query)
results = test_pipeline.run()
```
- [command line interface](https://github.com/MaterialEyes/exsclaim/wiki/Command-Line-Interface)
```
$ exsclaim run /path/to/query.json
```
- [user interface](https://github.com/MaterialEyes/exsclaim/wiki/User-Interface)
![Screenshot of EXSCLAIM user interface. Search form on left of screen and grid of image results on the right.](https://drive.google.com/uc?export=view&id=1OGPrMwld_9fYPlYh50PV7JowIzP52nha)

## Using EXSCLAIM

### Requirements
EXSCLAIM works with Python 3.6+. We recommend using a conda or python environment to install dependencies. To use the pipeline, you need a Query on which to run the pipeline. The query can be a JSON or Python dictionary (depending on how you are accessing the pipeline) and must have the parameters(/keys/attributes) defined in the [Query JSON schema](https://github.com/MaterialEyes/exsclaim/wiki/JSON-Schema#query-json-) and examples can be found [in the query directory](https://github.com/MaterialEyes/exsclaim/tree/master/query).

### Installation
There are multiple ways to use EXSCLAIM. If you wish to develop or modify the source code, see [Git Clone](#gitclone) installation instructions. If you simply wish to utilize the package, see the [Pip](#pip) installation instructions. For utilizing EXSCLAIM's User Interface (which is useful if you want to avoid writing any code or want an easy to way to view results), see [UI](#ui) instructions. 

#### Pip
To install the latest stable release:
```
pip install exsclaim
python -m spacy download en_core_web_sm
```

#### Docker
To start container:
`docker-compose up`
To re-build:
`docker-compose up --build`


To run in python shell
`docker exec -it <service_name> bash`

#### Git Clone
To install directly from github, run the following commands (it is recommended to run in a conda or python virtual environment):
```
git clone https://github.com/MaterialEyes/exsclaim.git
cd exsclaim
pip setup.py install
python -m spacy download en_core_web_sm
```

If you run into errors, please check [Troubleshooting](https://github.com/MaterialEyes/exsclaim/wiki/Troubleshooting). If they persist, please open an issue.

## Acknowledgements <a name="credits"></a>
This material is based upon work supported by Laboratory Directed Research and Development (LDRD) funding from Argonne National Laboratory, provided by the Director, Office of Science, of the U.S. Department of Energy under Contract No. DE-AC02-06CH11357

This work was performed at the Center for Nanoscale Materials, a U.S. Department of Energy Office of Science User Facility, and supported by the U.S. Department of Energy, Office of Science, under Contract No. DE-AC02-06CH11357.

We gratefully acknowledge the computing resources provided on Bebop, a high-performance computing cluster operated by the Laboratory Computing Resource Center at Argonne National Laboratory.

## Citation
If you find EXSCLAIM! useful, please encourage its development by citing the following [paper](https://arxiv.org/abs/2103.10631) in your research:
```
Schwenker, E., Jiang, W. Spreadbury, T., Ferrier N., Cossairt, O., Chan M.K.Y., EXSCLAIM! - An automated pipeline for the construction and
labeling of materials imaging datasets from scientific literature. arXiv e-prints (2021): arXiv-2103
```

#### Bibtex
```
@article{schwenker2021exsclaim,
  title={EXSCLAIM! - An automated pipeline for the construction of labeled materials imaging datasets from literature},
  author={Schwenker, Eric and Jiang, Weixin and Spreadbury, Trevor and Ferrier, Nicola and Cossairt, Oliver and Chan, Maria KY},
  journal={arXiv e-prints},
  pages={arXiv--2103},
  year={2021}
}
```
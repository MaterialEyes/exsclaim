![image](https://drive.google.com/uc?export=view&id=142XkACsDxT9r9VgVg0RUsVvjJhaBqRIs)

Automatic **EX**traction, **S**eparation, and **C**aption-based natural **L**anguage **A**nnotation of **IM**ages from scientific figures 
[[wiki](https://github.com/MaterialEyes/exsclaim/wiki)] [[paper](https://arxiv.org/abs/2103.10631)]



#### 🤔 Consider Collaboration ...
If you find this tool or any of its derived capabilities useful, please consider registering as a user of Center for Nanoscale Materials. We will keep you posted of latest developments, as well as opportunities for computational resources, relevant data, and collaboration. Please contact Maria Chan (mchan@anl.gov) for details.

## Getting Started

### Requirements
You need a working python 3.x installation to be able to use EXSCLAIM! We recommend using a conda or virtualenv environment to install dependencies. 

### Installation

#### Pip
To install test version:
```
pip install --extra-index-url https://test.pypi.org/simple/ exsclaim-materialeyes
python -m spacy download en_core_web_sm
```
To check that it installed correctly, run the following python code (it may take a few minutes the first time as it will download model checkpoints):
```
from exsclaim.pipeline import Pipeline
test_pipeline = Pipeline("test")
results = test_pipeline.run()
```
You should see something like this, and then results in the extracted/nature-test/ directory
```
Running Journal Scraper
GET request: https://www.nature.com/.....
>>>> (1 of 2) ....
```

If you run into errors, please check [Troubleshooting](#troubleshooting)

#### Git Clone
To install for development, run the following commands (it is recommended to run in a conda or python virtual environment):
```
git clone https://github.com/MaterialEyes/exsclaim.git
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
To test that it has been installed correctly, you can run the following code from the root exsclaim directory:
```
python -m unittest discover
```
This will run a series of unit tests that should take a few minutes. If successful, when complete the terminal should print <code>OK</code>.

## Usage

### REQUIRED: Formulate a JSON search query
A search query JSON is the singular point-of-entry for using the EXSCLAIM! search and retrieval tools.

Here we query open access [Nature](https://www.nature.com) journals to find figures related to HAADF-STEM images of exfoliated MoS<sub>2</sub> flakes. Limiting the results to the top 5 most relevant hits, the query might look something like:

> [nature-exfoliated-MoS2-flakes.json](https://github.com/MaterialEyes/exsclaim/blob/master/query/nature-exfoliated-MoS2-flakes.json) 
```
{   
    "name": "nature-exfoliated-MoS2-flakes",
    "journal_family": "nature",
    "maximum_scraped": 5,
    "sortby": "relevant",
    "query":
    {
        "search_field_1":
        {
            "term":"MoS2 flake",
            "synonyms":["MoS2 nanostructures", "MoS2 layers", "MoS2"]
        },
        "search_field_2": 
        {
            "term":"HAADF-STEM",
            "synonyms":["HAADF-STEM","HAADF STEM","HAADF images","HAADF-STEM imaging","HAADF-STEM image"]
        },
        "search_field_3": 
        {
            "term":"exfoliated edge",
            "synonyms":["edge of an exfoliated"]
        }
    },
    "results_dir": "extracted/nature-exfoliated-MoS2-flakes/",
    "open": true,
    "save_format": [""]
}
```
Saving the query avoids having to completely reformulate the structure with each new search entry and establishes provenance for the extraction results. Additional JSON search query examples can be found in the [query](https://github.com/MaterialEyes/exsclaim/blob/master/query) folder in the root directory. A full specification of the Query JSON schema can be found [here](https://github.com/MaterialEyes/exsclaim/wiki/JSON-Schema#query-json-).

### REQUIRED: Use the Pipeline class to conduct a search based on the desired Query JSON

There are several ways to access the Pipeline class. 

#### Option One: Direct Python Import
With the [nature-exfoliated-MoS2-flakes.json](https://github.com/MaterialEyes/exsclaim/blob/master/query/nature-exfoliated-MoS2-flakes.json) search query from above, extract relevant figures by running a <code>JournalScraper</code> through an EXSCLAIM! <code>Pipeline</code>:

```python
from exsclaim.pipeline import Pipeline # will always use

# Set query path
query_path = "query/nature-exfoliated-MoS2-flakes.json"

# Initialize EXSCLAIM! pipeline with a Query JSON (this can either be a
# path to a json file or a Python dictionary).
exsclaim_pipeline = Pipeline(query_path) 

# Run the tools through the pipeline, writiing results to the 'results_dir'
# specified in the Query JSON. 
# using run() with no arguments runs all three phases of the pipeline.
exsclaim_pipeline.run(journal_scraper=True,      # Runs JournalScraper module
                      caption_distributor=True,    # Runs CaptionDistributor module    
                      figure_separator=True)     # Runs FigureSeparator module

```
Successful execution of the code will result in the creation of a directory populated with figures extracted from journals returned as search hits from the main [Nature](https://www.nature.com) homepage.

#### Option Two: Using pre-built run.py
The run.py file provides an easy method to quickly access the Pipeline class with limited Python knowledge. You only have to supply two parameters: the tools you wish to run and the query you wish to run them on. You can use the command line: 
```
python run.py --query nature-exfoliated-MoS2-flakes --tools jcf
```
Or edit the QUERY and TOOLS variables at the top of the run.py file and running:
```
python run.py
```
In either case, if the query JSON you supply is in the query/ folder, you only need to supply the name (without the .json). Otherwise, supply the whole path. For tools, supply a string with the first letter of each tool you wish to run:
 - j: JournalScraper
 - c: CaptionDistributor
 - f: FigureSeparator

### Results

After successful completion of the pipeline, results will be saved in the results directory written in the query JSON 'results_dir' field. This will include:
 - exsclaim.json file: This is a json that maps each extracted figure name to its respective Figure JSON. The Figure JSON contains information about the article the figure appears in and, if CaptionDistributor and FigureSeparator were run, information on each of the figure's subfigures. The whole exsclaim JSON schema is described [here](https://github.com/MaterialEyes/exsclaim/wiki/JSON-Schema#exsclaim-json-).
 - _articles, _captions, _figures files: Used to keep track of the articles, figure captions and figures that have already been processed by JournalScraper, CaptionDistributor, and FigureSeparator, resepectively. 
 - figures/: stores all figures downloaded by JournalScraper
 - html/: stores full html of each article scraped by JournalScraper
 - extractions/ (optional): present if "visualize" present in Query JSON "save_format" list. Contains .png files for each figure, displaying FigureSeparator and CaptionDistributor results.
 - boxes/ (optional): present if "boxes" present in Query JSON "save_format" list. Contains .png files for each figure, with bounding boxes drawn on each figure.
 - images/ (optional): present if "save_subfigures" present in Query JSON "save_format" list. A directory contianing each subfigure extracted as a separate file. 

## Example Datasets
Checkout EXSCLAIM!-generated datasets published in the [Materials Data Facility](https://materialsdatafacility.org/):
 - EXSCLAIM! Exploratory Dataset - Nanostructure Images from Nature Journals ([10.18126/v7bl-lj1n](https://doi.org/10.18126/v7bl-lj1n))
 - EXSCLAIM! Validation Dataset - Selections from Amazon Mechanical Turk Benchmark ([10.18126/a6jr-yfoq](https://doi.org/10.18126/a6jr-yfoq))

 ## Uninstall
 ```
 pip uninstall exsclaim-materialeyes
 ```
 The screen should then show something like:
 ```
 Uninstalling exsclaim-materiealeyes-0.0.10:
  Would remove:
    /path/to/site-packages/exsclaim/*
    /path/to/site-packages/exsclaim_materialeyes-0.0.10.dist-info/*
  Would not remove (might be manually added):
    /path/to/site-packages/exsclaim/figures/checkpoints/classifier_model.pt
    /path/to/site-packages/exsclaim/figures/checkpoints/object_detection_model.pt
    /path/to/site-packages/exsclaim/figures/checkpoints/scale_bar_detection_model.pt
    /path/to/site-packages/exsclaim/figures/checkpoints/text_recognition_model.pt
```
To completely remove the installation, run:
Linux/OSX:
```
rm /path/to/site-packages/exsclaim/figures/checkpoints/*pt
```
Windows:
```
del "\path\to\site-packages\exsclaim\figures/checkpoints\*pt"
```

## Troubleshooting

### Downloading models
If you run into trouble downloading the models programmatically, just download them from [here](https://anl.box.com/s/b8snw0uk242velopy1t6zm56i4rvy7ct) and place them in the /path/to/exsclaim/figures/checkpoints/ folder.

### Installing opencv
There are some issues installing opencv-python with wheels. You can try:
```
pip install --upgrade pip setuptools wheel
```
and then trying to install again. Or try installing without pip, with:
```
sudo apt-get install python-opencv
```
or:
```
conda install -c menpo opencv
```

## Citation
If you find EXSCLAIM! useful, please encourage its development by citing the following paper in your research:
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

## Acknowledgements <a name="credits"></a>
This material is based upon work supported by Laboratory Directed Research and Development (LDRD) funding from Argonne National Laboratory, provided by the Director, Office of Science, of the U.S. Department of Energy under Contract No. DE-AC02-06CH11357

This work was performed at the Center for Nanoscale Materials, a U.S. Department of Energy Office of Science User Facility, and supported by the U.S. Department of Energy, Office of Science, under Contract No. DE-AC02-06CH11357.

We gratefully acknowledge the computing resources provided on Bebop, a high-performance computing cluster operated by the Laboratory Computing Resource Center at Argonne National Laboratory.


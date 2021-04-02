![image](https://drive.google.com/uc?export=view&id=142XkACsDxT9r9VgVg0RUsVvjJhaBqRIs)

Automatic **EX**traction, **S**eparation, and **C**aption-based natural **L**anguage **A**nnotation of **IM**ages from scientific figures 
[[wiki](https://github.com/MaterialEyes/exsclaim/wiki)] [[paper](https://)]


## Getting started

There are multiple ways to use EXSCLAIM. If you wish to develop or modify the source code, see [Git Clone](#gitclone) installation instructions. If you simply wish to utilize the package, see the [Pip](#pip) installation instructions. For utilizing EXSCLAIM's User Interface (which is useful if you want to avoid writing any code or want an easy to way to view results), see [UI](#ui) instructions. 

If you run into errors, please check [Troubleshooting](#troubleshooting)

## Installing EXSCLAIM

### Requirements
You need a working python 3.6+ installation to be able to use EXSCLAIM! We recommend using a conda or virtualenv environment to install dependencies. 

### Methods

#### Pip
To install the latest stable release:
```
pip install --extra-index-url https://test.pypi.org/simple/ exsclaim-materialeyes
python -m spacy download en_core_web_sm
```

#### Git Clone
To install for development, run the following commands (it is recommended to run in a conda or python virtual environment):
```
git clone https://github.com/MaterialEyes/exsclaim.git
cd exsclaim
pip setup.py install
python -m spacy download en_core_web_sm
```

### Testing the installation

To test the installation was successful, run the following command:
```
$ exsclaim test
```
You should see something like this, and then results in the extracted/nature-test/ directory
```
Running Journal Scraper
GET request: https://www.nature.com/.....
>>>> (1 of 2) ....
```
The first time you run the pipeline may take a long time, as you must download model checkpoints.

## Using EXSCLAIM

Using EXSCLAIM requires a user-generated query. These tell the pipeline how to run and what to look for. In a query, you specify which keywords to look for, which journals to look for them in, how many to articles to look at, and how to log and store results. The full query schema is available [in the wiki](https://github.com/MaterialEyes/exsclaim/wiki/JSON-Schema#query-json-) and examples can be found [in the query directory](https://github.com/MaterialEyes/exsclaim/tree/master/query).

Once you have a query, you can choose what tools to run. The options are JournalScraper, CaptionDistributor, and FigureSeparator. For most cases you will want to run all three, which is the default behavoir. 

The result of running the exsclaim pipeline is a dataset of images from published journal articles labeled with their captions and other extracted metadata. For more information, see [Results](#results).

Depending on your use case and experience with Python, you can use EXSCLAIM as a Python import, a command-line tool, or (easiest) use its user interface.

### Importing EXSCLAIM

You can import EXSCLAIM to run in Python scripts (or modules):
```
from exsclaim.pipeline import Pipeline
test_pipeline = Pipeline(query)
results = test_pipeline.run()
```
<code>query</code> can either be a Python dictionary or the path to a JSON file. Either must have the parameters(/keys/attributes) defined in the [Query JSON schema](https://github.com/MaterialEyes/exsclaim/wiki/JSON-Schema#query-json-) and examples can be found [in the query directory](https://github.com/MaterialEyes/exsclaim/tree/master/query).

If you wish to run only a subset of tools, you can use the keyword arguments like this:
```
results = test_pipeline.run(figure_separator=True, caption_distributor=True, journal_scraper=True)
```
settting those you wish not to use to False. 

### Command-Line Tool

You can utilize EXSCLAIM from the command line:
```
$ exsclaim /path/to/query.json
```
To specify which tools to run, use the <code>--tools</code> flag. The default is to run all tools. For example:
```
$ exsclaim --tools jc /path/to/query.json
```
After the <code>--tools</code> flag, provide the first letter of each tool you wish to run. The above command will run the JournalScraper and CaptionDistributor.


### User Interface

To use the UI, you must have PostgreSQL installed. To download, check [the official instructions](https://www.postgresql.org/download/).

Then in the command-line, type:
```
$ exsclaim_ui
```
Do not close your command line window while using the UI. Navigate to http://127.0.0.1:8000/ to use the UI. From here you can naviaget to the query page to submit a query using a simple web form, or to the results page to explore and filter results.


To test that it has been installed correctly, you can run the following code from the root exsclaim directory:
```
python -m unittest discover
```
This will run a series of unit tests that should take a few minutes. If successful, when complete the terminal should print <code>OK</code>.



### Walkthrough

#### Making a query
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


There are several ways to access the Pipeline class. 

#### Direct Python Import
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
```sh
Schwenker, E., Jiang, W., Spreadbury, T., Ferrier N., Cossairt, O., Chan M.K.Y., EXSCLAIM! – An automated pipeline for the construction and
labeling of materials imaging datasets from scientific literature. **in preparation** (2021)
```

#### Bibtex
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
This material is based upon work supported by Laboratory Directed Research and Development (LDRD) funding from Argonne National Laboratory, provided by the Director, Office of Science, of the U.S. Department of Energy under Contract No. DE-AC02-06CH11357

This work was performed at the Center for Nanoscale Materials, a U.S. Department of Energy Office of Science User Facility, and supported by the U.S. Department of Energy, Office of Science, under Contract No. DE-AC02-06CH11357.

We gratefully acknowledge the computing resources provided on Bebop, a high-performance computing cluster operated by the Laboratory Computing Resource Center at Argonne National Laboratory.

## License <a name="license"></a>

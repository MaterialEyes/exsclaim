![image](https://drive.google.com/uc?export=view&id=142XkACsDxT9r9VgVg0RUsVvjJhaBqRIs)

Automatic **EX**traction, **S**eparation, and **C**aption-based natural **L**anguage **A**nnotation of **IM**ages from scientific figures 
[[wiki](https://gitlab.com/MaterialEyes/exsclaim/wikis/home)] [[paper](https://)]


## Getting started

### Requirements
You need a working python 3.x installation to be able to use EXSCLAIM! We recommend using a conda or virtualenv environment to install dependencies. 
### Installation

To install test version:
```
pip install -i https://test.pypi.org/simple/ exsclaim-tspread
```
To test that it has been installed correctly, run the following python code:
```
from exsclaim.pipeline import Pipeline
test_pipeline = Pipeline("", test=True)
results = test_pipeline.run()
```
You should then see something like:
```
Running Journal Scraper
GET request: https://www.nature.com/.....
>>>> (1 of 10) ....
```
When complete the results should be in extracted/nature-test/ and the json will be stored in the 'results' variable.

### Usage

### REQUIRED: Formulate a JSON search query
A JSON search query is the singular point-of-entry for using the EXSCLAIM! search and retrieval tools.

Here we query [Nature](https://www.nature.com) journals to find figures related to HAADF-STEM images of exfoliated MoS<sub>2</sub> flakes. Limiting the results to the top 5 most relevant hits, the query might look something like:

> [nature-exfoliated-MoS2-flakes.json](https://github.com/eschwenk/exsclaim-prerelease/tree/master/test/query/nature-exfoliated-MoS2-flakes.json) 
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
    "results_dir": "extracted/nature-exfoliated-MoS2-flakes/"
}
```

Saving the query avoids having to completely reformulate the structure with each new search entry and establishes provenance for the extraction results. Additional JSON search query examples can be found in the [test](https://github.com/eschwenk/exsclaim-prerelease/tree/master/test) folder in the root directory. 

### OPTION 1: Query a journal source to extract relevant figures
With the [nature-exfoliated-MoS2-flakes.json](https://github.com/eschwenk/exsclaim-prerelease/tree/master/test/query/nature-exfoliated-MoS2-flakes.json) search query from above, extract relevant figures by running a <code>JournalScraper</code> through an EXSCLAIM! <code>Pipeline</code>:

```python
from exsclaim.pipeline import Pipeline # will always use
from exsclaim.tool import JournalScraper

# Set query path
query_path = "query/nature-exfoliated-MoS2-flakes.json"

# Set path to initial exsclaim_dict JSON (if applicable)
exsclaim_path = ""

# Initialize EXSCLAIM! tool(s) and define run order in a tools list
js = JournalScraper()
tools = [js] 

# Initialize EXSCLAIM! pipeline
exsclaim_pipeline = Pipeline(query_path=query_path , exsclaim_path=exsclaim_path)

# Run the tools through the pipeline
exsclaim_pipeline.run(tools) # figures written to 'results_dir' specified in the query

```

Successful execution of the code will result in the creation of a directory populated with figures extracted from journals returned as search hits from the main [Nature](https://www.nature.com) homepage.

### OPTION 2: Create an annotated materials imaging dataset from extracted figures
To extend the search to create an annotated imaging dataset, import a <code>CaptionSeparator</code> and <code>FigureSeparator</code> tool (in addition to the <code>JournalScraper</code>) to run through the EXSCLAIM! <code>Pipeline</code>:

> [nature-exfoliated-MoS2-flakes.py](https://github.com/eschwenk/exsclaim-prerelease/tree/master/test/nature-exfoliated-MoS2-flakes.py)
```python
from exsclaim.pipeline import Pipeline # will always use
from exsclaim.tool import JournalScraper, CaptionSeparator, FigureSeparator

# Set query path
query_path = "query/nature-exfoliated-MoS2-flakes.json"

# Set path to initial exsclaim_dict JSON (if applicable)
exsclaim_path = ""

# Initialize EXSCLAIM! tool(s) and define run order in a tools list
js = JournalScraper()
cs = CaptionSeparator()
fs = FigureSeparator()
tools = [js,cs,fs] 

# Initialize EXSCLAIM! pipeline
exsclaim_pipeline = Pipeline(query_path=query_path , exsclaim_path=exsclaim_path)

# Run the tools through the pipeline
exsclaim_pipeline.run(tools) # figures written to 'results_dir' specified in the query

# Save image and label (.csv) results to file
exsclaim_pipeline.to_file()
```

For this example, extracted images are written into separate subfigure folders within a folder for the figure itself. The root directory will also contain a 'labels.csv' with annotations, and for a more concise record of the search results, an 'exsclaim.json', which records urls to each extracted figure, bounding box information for the detected images, and the associated caption text for each image.  

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
This material is based upon work supported by Laboratory Directed Research and Development (LDRD) funding from Argonne National Laboratory, provided by the Director, Office of Science, of the U.S. Department of Energy under Contract No. DE-AC02-06CH11357

This work was performed at the Center for Nanoscale Materials, a U.S. Department of Energy Office of Science User Facility, and supported by the U.S. Department of Energy, Office of Science, under Contract No. DE-AC02-06CH11357.

We gratefully acknowledge the computing resources provided on Bebop, a high-performance computing cluster operated by the Laboratory Computing Resource Center at Argonne National Laboratory.

## License <a name="license"></a>

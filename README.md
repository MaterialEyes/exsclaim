![image](https://drive.google.com/uc?export=view&id=142XkACsDxT9r9VgVg0RUsVvjJhaBqRIs)


Automatic **EX**traction, **S**eparation, and **C**aption-based natural **L**anguage **A**nnotation of **IM**ages from scientific figures.
[[wiki](https://gitlab.com/MaterialEyes/exsclaim/wikis/home)] [[paper](https://)]

## Getting started

### Requirements
You need a working python 3.x installation to be able to use exsclaim, and [gdown](https://github.com/wkentaro/gdown) to directly download text detection and figure separation models. We highly recommend installing [Anaconda](https://anaconda.org/), to manage the installation environment.

### Installation (Recommended)
- Clone this repo and create a new conda environment with Python 3.7 and gdown:
```sh
git clone https://github.com/eschwenk/exsclaim
conda create -n exsclaim -c conda-forge python=3.7 gdown
```
- Activate this environment, navigate to the root directory and download the models:
```sh
conda activate exsclaim
cd exsclaim
./bin/download_models.sh
```
- Install with pip:
```sh
pip install .
```
- Finally, download required model for the spaCy installation:
```python
python -m spacy download en_core_web_sm
```

## Usage
A json search query is the singular point of entry for using the search and retrieval tools in the EXSCLAIM! pipeline.

Here, we will query [Nature]() journals to find figures related to HAADF-STEM images of exfoliated MoS<sub>2</sub> flakes (for demo purposes, we will limit our results to the top 5 most relevant hits). For this task, the query might look something like:

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
We recommend saving this query as a .json file to avoid having to completely reformulate a query with each new search entry, and for cases where tracing the provenance of the extraction results is important. Example of json search queries can be found in the [test]() folder in the root directory. 

### Query a journal source to extract relevant figures
With the [nature-exfoliated-MoS2-flakes.json]() search query from above, extract relevant figures by running a <code>JournalScraper</code> through an EXSCLAIM! <code>Pipeline</code>:

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
Upon successful execution of the code in this example, the following directory structure with the extracted figures will appear:

<p align="center">
<img src="https://drive.google.com/uc?export=view&id=14sAooOZn0ARwU0YrGiNOr7BjhD5J6--c" width="300" />
</p>

### Create an annotated materials imaging dataset from literature
To extend the [nature-exfoliated-MoS2-flakes.json]() search query to create an annotated imaging dataset from the extracted figures, import a <code>CaptionSeparator</code> and <code>FigureSeparator</code> tool (in addition to the <code>JournalScraper</code>) to run through the EXSCLAIM! <code>Pipeline</code>:
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
For this example, the resulting directory structure will organize the extracted images into subfigure folders within an image folder for each governing figure. The root diretory will also contain a 'labels.csv' with annotations, and for a more concise record of the search results, the "exsclaim.json" records urls to each extracted figure, bounding box information for the detected images, as well as associated caption text.  

<p align="center">
<img src="https://drive.google.com/uc?export=view&id=1cz1DERwrhO3P-1D3J9VBrVwK6FGRORb9" width="300" />
</p>

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

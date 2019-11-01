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
```python
python -m spacy download en_core_web_sm
```

Check out the project [wiki](https://gitlab.com/MaterialEyes/exsclaim/wikis/home) for more details!

## Usage
A simple example of EXSCLAIM! figure-extraction capabilities:

(1) Import the <code>Pipeline</code> and <code>JournalScraper</code> objects and construct a query to 
extract figures from the top 10 most-relevant *Nature* journal articles 
returned in a search for **HRTEM** and **Au nanoparticles**.
```python
>>> from exsclaim.pipeline import Pipeline
>>> from exsclaim.tool import JournalScraper
>>> query = \
... {   
...     "name": "nature-hrtem-au-nanoparticles",
...     "journal_family": "nature",
...     "maximum_scraped": 10,
...     "sortby": "relevant",
...     "results_dir": "extracted/",
...     "query":
...     {
...         "search_field_1":
...         {
...             "term":"HRTEM",
...             "synonyms":["High-resolution transmission electron microscopy"]
...         },
...         "search_field_2":
...         {
...             "term":"Au nanoparticles",
...             "synonyms":["Gold nanoparticles"]
...         }
...     }
... }
```
(2)  Initialize a <code>Pipeline</code> and <code>JournalScraper</code> from scratch and put the tool(s) in a list that specifies run-time order.
```python
>>> js = JournalScraper()
>>> exsclaim_pipeline = Pipeline(query,exsclaim_path="")
>>> tools = [js] 
```
(3)  Run the tools through the <code>Pipeline</code> and save results to the directory specified in the query
```python
>>> exsclaim_pipeline.run(tools)
```
```sh       
Running Journal Scraper
(1 of 10) Extracting figures from: s41467-019-12853-8
(2 of 10) Extracting figures from: srep11949
⋮
>>> SUCCESS
```
```python
>>> exsclaim_pipeline.to_file()
```

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

import argparse
import pathlib
from exsclaim.pipeline import Pipeline

# Path to/ Name of Query JSON, used if no command line argument is supplied
# If no file extension (.json) is supplied, path will be assumed to be:
# path/to/exsclaim/query/<QUERY>.json
QUERY = "nature-test"

# First letter of tools to be run, if no command line argument is supplied
TOOLS = "jcf"

# Parse Command Line arguments, if present
parser = argparse.ArgumentParser(description='Run the EXSCLAIM! Pipeline')
parser.add_argument('--query', '-q', type=str, default=None,
                    help=('Name of EXSCLAIM Query JSON, defined here: '
                          'https://github.com/MaterialEyes/exsclaim/wiki/JSON-Schema#query-json-'
                          '. Samples are in the query folder. If a file '
                          'extension is included (.json) the variable will '
                          'be treated as a full path, otherwise the full path '
                          'will be assumed as /path/to/exsclaim/query/<query>.json'
                          '. If no value is supplied, QUERY variable in '
                          'run.py will be used.'))
parser.add_argument('--tools', '-t', type=str, default=None,
                    help=('String containing the first letter of each tool '
                          'to be run on input query.\nJ\tJournalScraper\nC\t'
                          'CaptionDistributor\nF\FigureSeparator. Order and '
                          'case insensitive. If no value is supplied, TOOLS '
                          'variable in run.py will be used.'))
args = parser.parse_args()
if args.query is not None:
    QUERY = args.query
if args.tools is not None:
    TOOLS = args.tools
# Format args to run enter into Pipeline
TOOLS = TOOLS.lower()
if ".json" not in QUERY:
    current_file = pathlib.Path(__file__).resolve(strict=True)
    queries = current_file.parent / "query"
    QUERY = queries / (QUERY + ".json")
f = "f" in TOOLS
j = "j" in TOOLS
c = "c" in TOOLS

# Run the pipeline
pipeline = Pipeline(QUERY)
pipeline.run(journal_scraper=j, caption_distributor=c, figure_separator=f)
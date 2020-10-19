from exsclaim.pipeline import Pipeline
from exsclaim.tool import JournalScraper, CaptionSeparator
from exsclaim.figure import FigureSeparator

# Set query paths
query_path = "query/nature-twins-Fe3O4-thinfilms.json"

# Initialize EXSCLAIM! pipeline
exsclaim_pipeline = Pipeline(query_path=query_path)

# Run the pipeline. Using run() with no list of tools runs the pipeline on
# the JournalScraper, CaptionSeparator, FigureSeparator
exsclaim_pipeline.run(tools)
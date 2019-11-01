from exsclaim.pipeline import Pipeline
from exsclaim.tool import JournalScraper, CaptionSeparator, FigureSeparator, TextReader

# Set query paths
query_path = "query/nature-haadf-ag-np.json"

# Set path to initial exsclaim_dict JSON (if applicable)
exsclaim_path = ""

# Initialize EXSCLAIM! tools
js = JournalScraper()
cs = CaptionSeparator()
fs = FigureSeparator()
tr = TextReader()

tools = [js,cs,fs,tr] # define run order

# Initialize EXSCLAIM! pipeline
exsclaim_pipeline = Pipeline(query_path=query_path , exsclaim_path=exsclaim_path)

# Run the tools through the pipeline
exsclaim_pipeline.run(tools)

# Group related image objects into master images
exsclaim_pipeline.group_objects()

# Save results to file
exsclaim_pipeline.to_file()

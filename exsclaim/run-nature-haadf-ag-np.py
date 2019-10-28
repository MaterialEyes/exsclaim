import pipeline
import exsclaim

# Set query paths
query_path = "data/query/nature-haadf-ag-np.json"

# Set path to initial exsclaim_dict JSON (if applicable)
exsclaim_path = ""

# Set model paths
caption_models    = "captions/models/"
figure_models     = "figures/models/"
imagetext_models  = "imagetexts/models/"

# Initialize EXSCLAIM! tools
js = exsclaim.JournalScraper()
cs = exsclaim.CaptionSeparator(caption_models)
fs = exsclaim.FigureSeparator(figure_models)
tr = exsclaim.TextReader(imagetext_models)

tools = [js,cs,fs,tr] # define run order

# Initialize EXSCLAIM! pipeline
exsclaim_pipeline = pipeline.Pipeline(query_path=query_path , exsclaim_path=exsclaim_path)

# Run the tools through the pipeline
exsclaim_pipeline.run(tools)

# Group related image objects into master images
exsclaim_pipeline.group_objects()

# Save results to file
exsclaim_pipeline.to_file()
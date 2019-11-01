from exsclaim.pipeline import Pipeline

# Set query paths
query_path = "query/nature-nmat3669-test.json"

# Set path to exisiting exsclaim_dict JSON (output from running all tools through the pipeline)
exsclaim_path = "extracted/nature-nmat3669-test/uas.json"

# Initialize EXSCLAIM! pipeline with exisiting exsclaim_dict
exsclaim_pipeline = Pipeline(query_path=query_path , exsclaim_path=exsclaim_path)

# Group related image objects into master images
exsclaim_pipeline.group_objects()

# Save results to file
exsclaim_pipeline.to_file()
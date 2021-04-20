from django.shortcuts import render
import os
import sys
import multiprocessing as mp
import pathlib
try:
    from exsclaim.pipeline import Pipeline
except:
    current_file = pathlib.Path(__file__).resolve(strict=True)
    base_dir = current_file.parent.parent.parent.parent
    sys.path.append(str(base_dir))
    from exsclaim.pipeline import Pipeline

def run_pipeline(search_query):
    pipeline = Pipeline(search_query)
    pipeline.run()

def query_view(request):
    template_name = "exsclaim/query.html"
    # formulate exsclaim's query json

    if request.method == "POST":
        query = request.POST.dict()
        journal_families = request.POST.getlist("journalfamily[]")
        query_name = "-".join(journal_families + [query["keyword"]])
        search_query = {
            "name": query_name,
            "journal_family": journal_families[0],
            "maximum_scraped": int(query["max_scraped"]),
            "sortby": "relevant",
            "query":
            {
                "search_field_1":
                {
                    "term": query["keyword"],
                    "synonyms": query["synonyms"].split(",")
                }
            },
            "open": query.get("open", False) == "true",
            "save_format": ["postgres"],
            "logging": ["exsclaim.log"],
            "results_dir": query.get("results_base_dir", None),
        }

        p = mp.Process(target=run_pipeline, args=(search_query,))
        p.start()
        minutes = int(query["max_scraped"]) * (150/60)
        results_log = os.path.join(search_query["name"], "exsclaim.log") 
        return render(request,
            "exsclaim/submission.html",
            context={
                "minutes": minutes,
                "log": results_log
            }
        )
    
    else:
        return render(request, template_name)
from exsclaim.ui.results.models import Figure
from django.db.models.query import QuerySet
import json
from pathlib import Path
from shutil import copyfile
import requests
from exsclaim.ui.exsclaim_gui.settings import STATICFILES_DIRS

def move_image(figure, destination):
    """ move image represented by figure to destination """
    figure_path = None
    for directory in STATICFILES_DIRS:
        potential_path = Path(directory) / figure.path
        if Path.is_file(potential_path):
            figure_path = potential_path
    if not figure_path:
        image_data = requests.get(figure.url).content
        with open(destination / Path(figure.path).name, 'wb') as f:
            f.write(image_data)
    else:
        copyfile(figure_path, destination / figure_path.name)


def extract_queryset(queryset):
    """ save a queryset as an exsclaim.json with figures zip """
    exsclaim_json = {}
    figure_directory = Path.cwd() / "queryset" / "figures"
    Path.mkdir(figure_directory, parents=True, exist_ok=True)
    for subfigure in queryset:
        figure_path = subfigure.figure.path
        figure_name = Path(figure_path).name
        figure = subfigure.figure
        article = figure.article
        if figure_name not in exsclaim_json:
            exsclaim_json[figure_name] = {
                "title": article.title,
                "article_url": article.url,
                "article_name": article.doi,
                "authors": article.authors,
                "abstract": article.abstract,
                "full_caption": figure.caption,
                "caption_delimiter": figure.caption_delimiter,
                "image_url": figure.url,
                "license": article.license,
                "open": article.open,
                "figure_path": figure_path,
                "master_images": []
            }
        exsclaim_json[figure_name]["master_images"].append(
            {
                "classification": subfigure.classification,
                "height": subfigure.height,
                "width": subfigure.width,
                "geometry": [
                    {"x": subfigure.x1, "y": subfigure.y1},
                    {"x": subfigure.x1, "y": subfigure.y2},
                    {"x": subfigure.x2, "y": subfigure.y2},
                    {"x": subfigure.x2, "y": subfigure.y1},                                        
                ],
                "caption": subfigure.caption,
                "keywords": subfigure.keywords,
                "general": subfigure.general,
                "nm_width": subfigure.nm_width,
                "nm_height": subfigure.nm_height,
            }
        )
        move_image(figure, figure_directory)
    with open(figure_directory.parent / "exsclaim.json", "w") as f:
        json.dump(exsclaim_json, f)
    

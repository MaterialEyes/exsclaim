from exsclaim.ui.results.models import Figure
from django.db.models.query import QuerySet
import json
from pathlib import Path
from shutil import copyfile
import requests
from exsclaim.ui.exsclaim_gui.settings import STATICFILES_DIRS
from exsclaim.ui.results import models

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


def extract_queryset(queryset, output_directory):
    """ save a queryset as an exsclaim.json with figures zip """
    exsclaim_json = {}
    figure_directory = Path(output_directory).resolve() / "figures"
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
        subfigure_json = {
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
        scale_bars = models.ScaleBar.objects.filter(subfigure__exact=subfigure.subfigure_id)
        scale_bar_jsons = []
        for scale_bar in scale_bars:
            scale_bar_labels = models.ScaleBarLabel.objects.filter(scale_bar__exact=scale_bar.scale_bar_id)
            if scale_bar_labels:
                scale_bar_label = scale_bar_labels[0]
                scale_bar_label_json = {
                    "text": scale_bar_label.text,
                    "label_confidence": scale_bar_label.label_confidence,
                    "box_confidence": scale_bar_label.box_confidence,
                    "nm": scale_bar_label.nm,
                    "geometry": [
                        {"x": scale_bar_label.x1, "y": scale_bar_label.y1},
                        {"x": scale_bar_label.x1, "y": scale_bar_label.y2},
                        {"x": scale_bar_label.x2, "y": scale_bar_label.y2},
                        {"x": scale_bar_label.x2, "y": scale_bar_label.y1},                                        
                    ],
                }
            else:
                scale_bar_label_json = None
            scale_bar_json = {
                "label": scale_bar_label_json,
                "length": scale_bar.length,
                "line_label_distance": scale_bar.line_label_distance,
                "confidence": scale_bar.confidence,
                "geometry": [
                    {"x": scale_bar.x1, "y": scale_bar.y1},
                    {"x": scale_bar.x1, "y": scale_bar.y2},
                    {"x": scale_bar.x2, "y": scale_bar.y2},
                    {"x": scale_bar.x2, "y": scale_bar.y1},                                        
                ],
            }
            scale_bar_jsons.append(scale_bar_json)
        subfigure_json["scale_bars"] = scale_bar_jsons
        try:
            label = models.SubfigureLabel.objects.get(subfigure__exact=subfigure.subfigure_id)
            subfigure_json["subfigure_label"] = {
                "text": label.text,
                "geometry": [
                    {"x": label.x1, "y": label.y1},
                    {"x": label.x1, "y": label.y2},
                    {"x": label.x2, "y": label.y2},
                    {"x": label.x2, "y": label.y1},                                        
                ],
            }
        except:
            pass
        exsclaim_json[figure_name]["master_images"].append(subfigure_json)

        move_image(figure, figure_directory)
    with open(figure_directory.parent / "exsclaim.json", "w") as f:
        json.dump(exsclaim_json, f)
    

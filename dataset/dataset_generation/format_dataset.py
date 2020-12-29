import json
import os
from PIL import Image
import random
import requests

def match_scale_bar_to_subfigure(master_images, scale_bars):
    """ Matches scale bars labels and lines to master images """
    unmatched = len(scale_bars)
    i = 0
    unmatched_scale_bars = scale_bars
    new_master_images = []
    while unmatched > 0:
        new_unmatched_scale_bars = []
        for scale_bar in unmatched_scale_bars:
            matched = False
            for subfigure in master_images:
                if is_contained(scale_bar["geometry"], subfigure["geometry"], i):
                    matched = True
                    unmatched -= 1
                    ## Add to json
                    subfigure.setdefault("scale_bar", {})
                    subfigure["scale_bar"].setdefault("geometry", scale_bar["geometry"])
                    #subfigure["scale_bar"]["geometry"] = scale_bar
                    new_master_images.append(subfigure)

            if not matched:
                new_unmatched_scale_bars.append(scale_bar)
        unmatched_scale_bars = new_unmatched_scale_bars
        if i > 25:
            print("uh oh... no match")
            break
        i += 1
    
    return master_images

def match_scale_label_to_subfigure(master_images, scale_labels):
    """ Matches scale bars labels and lines to master images """
    unmatched = len(scale_labels)
    i = 0
    unmatched_scale_labels = scale_labels
    new_master_images = []
    while unmatched > 0:
        new_unmatched_scale_labels = []
        for scale_label in unmatched_scale_labels:
            matched = False
            for subfigure in master_images:
                if is_contained(scale_label["geometry"], subfigure["geometry"], i):
                    matched = True
                    unmatched -= 1
                    ## Add to json
                    subfigure.setdefault("scale_bar", {})
                    subfigure["scale_bar"].setdefault("label", scale_label)
                    new_master_images.append(subfigure)

            if not matched:
                new_unmatched_scale_labels.append(scale_label)
        unmatched_scale_labels = new_unmatched_scale_labels
        if i > 50:
            print("uh oh... no match")
            break
        i += 5
    
    return master_images

def is_contained(inner, outer, padding=0):
    """ tests whether one bounding box is within another """
    inner_x1, inner_y1, inner_x2, inner_y2 = convert_box_format(inner)
    outer_x1, outer_y1, outer_x2, outer_y2 = convert_box_format(outer)
    outer_x1, outer_y1 = outer_x1 - padding, outer_y1 - padding
    outer_x2, outer_y2 = outer_x2 + padding, outer_y2 + padding

    if (inner_x1 > outer_x1 and inner_x2 < outer_x2 and
        inner_y1 > outer_y1 and inner_y2 < outer_y2):
        return True
    else:
        return False

def convert_box_format(geometry):
    """ Converts from [{"x": x1, "y": y1}, ...] to (x1, y1, ...) """
    #print(geometry)
    x1 = min([point["x"] for point in geometry])
    y1 = min([point["y"] for point in geometry])
    x2 = max([point["x"] for point in geometry])
    y2 = max([point["y"] for point in geometry])

    return x1, y1, x2, y2

def make_scale_detection_dataset(train_test_ratio=4,
                                 labelbox_json="labelbox.json"):
    """ creates a json containing all figures with scale bars """
    with open(labelbox_json, "r") as f:
        labelbox_dict = json.load(f)

    test_data = {}
    train_data = {}
    test_articles = set()
    train_articles = set()
    labels = 0
    for figure in labelbox_dict:
        download_labelbox_images(figure)
        figure_name = figure["External ID"]
        article_name = "_".join(figure_name.split(".")[0].split("_")[:-1])
        labelbox_name = figure["ID"]
        masters = figure["Label"].get("Master Image", [])
        scale_bars = figure["Label"].get("Scale Bar Line", [])
        scale_labels = figure["Label"].get("Scale Bar Label", [])
        if scale_bars == []:
            continue
        labels += len(scale_labels)
        figure_path = figure_name

        if (article_name in train_articles 
            or (random.randint(1, train_test_ratio) != 1
                and article_name not in test_articles)):
            # The image belongs in the training set
            train_data[figure_name] = {
                "figure_name": figure_name,
                "figure_path": figure_path,
                "master_images": masters,
                "scale_bars": scale_bars,
                "scale_labels": scale_labels
            }
            train_articles.add(article_name)
        else:
            test_data[figure_name] = {
                "figure_name": figure_name,
                "figure_path": figure_path,
                "master_images": masters,
                "scale_bars": scale_bars,
                "scale_labels": scale_labels
            }
            test_articles.add(article_name)

    print("Test Figures {}, Training Figures: {}".format(len(test_data), len(train_data)))
    with open("scale_bars_dataset_test.json", "w+") as f:
        json.dump(test_data, f)
    with open("scale_bars_dataset_train.json", "w+") as f:
        json.dump(train_data, f)


def is_number(n):
    try:
        float(n)
    except ValueError:
        return False
    return True


def clean_text(text):
    """ adds spaces to text, returns false if improper """
    if is_number(text) or "/" in text:
        return False
    if len(text.split(" ")) != 2:
        i = 1 if text[0] != "." else 2
        while is_number(text[:i]):
            i +=1
        text = text[:i-1] + " " + text[i-1:]
    
    if not is_number(text.split(" ")[0]):
        return False
    return text
    

def make_scale_reader_dataset():
    with open("labelbox.json", "r") as f:
        labelbox_dict = json.load(f)

    exsclaim_dict = {}
    labels = 0
    for figure in labelbox_dict:
        figure_name = figure["External ID"]
        figure_path = os.path.join('labeled_data', figure_name)
        scale_labels = figure["Label"].get("Scale Bar Label", [])
        if not os.path.isfile(figure_path):
            continue
        if scale_labels == []:
            continue
        for label in scale_labels:
            text = clean_text(label["text"])
            if not text:
                continue
            num = exsclaim_dict.get(text, 0)
            exsclaim_dict[text] = num + 1

    labels = 0
    for figure in labelbox_dict:
        figure_name = figure["External ID"]
        figure_path = os.path.join("labeled_data", figure_name)
        labelbox_name = figure["ID"]
        masters = figure["Label"].get("Master Image", [])
        scale_bars = figure["Label"].get("Scale Bar Line", [])
        scale_labels = figure["Label"].get("Scale Bar Label", [])
        if not os.path.isfile(figure_path):
            continue
        if scale_labels == []:
            continue
        i = 0
        for label in scale_labels:
            text = clean_text(label["text"])
            if not text or exsclaim_dict[text] < 10:
                continue
            geometry = label["geometry"]
            os.makedirs(os.path.join("scale_label_dataset", text), exist_ok=True)
            image = Image.open(figure_path).convert("RGB")
            cropped_image = image.crop(convert_box_format(geometry))

            cropped_image.save(os.path.join("scale_label_dataset", text, figure_name.split(".")[0] + "-" + str(i) + ".jpg"))
            i += 1

def download_labelbox_images(image_dictionary):
    """ Download images from Labelbox to prevent mismatched images """      
    figure_name = image_dictionary["External ID"]
    figure_url = image_dictionary["Labeled Data"]
    labeled_data = "labeled_data"
    os.makedirs(labeled_data, exist_ok=True)
    figure_path = os.path.join(labeled_data, figure_name)
    image_data = requests.get(figure_url).content
    with open(figure_path, "wb") as f:
        f.write(image_data)

if __name__ == "__main__":
    make_scale_reader_dataset()

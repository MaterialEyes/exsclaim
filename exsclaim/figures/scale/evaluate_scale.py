import json
import os
from torch import optim, nn, utils
import torchvision.transforms as T
from torchvision import datasets, transforms, models
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import torch
import pathlib
from PIL import Image
from pytorch_model_summary import summary
import argparse
from operator import itemgetter
from exsclaim.figures.scale.ctc import ctcBeamSearch
from exsclaim.figures.scale.lm import LanguageModel
from exsclaim.figures.scale.process import non_max_suppression_malisiewicz
from exsclaim.figures.models.crnn import CRNN
import exsclaim.utilities.boxes as boxes
import cv2
import random


def convert_to_rgb(image):
    return image.convert("RGB")

def create_scale_bar_objects(scale_bar_lines, scale_bar_labels):
    """ Match scale bar lines with labels to create scale bar jsons
    
    Args:
        scale_bar_lines (list of dicts): A list of dictionaries
            representing predicted scale bars with 'geometry', 'length',
            and 'confidence' attributes.
        scale_bar_labels (list of dicts): A list of dictionaries
            representing predicted scale bar labesl with 'geometry',
            'text', 'confidence', 'box_confidence', 'nm' attributes.
    Returns:
        scale_bar_jsons (list of Scale Bar JSONS): Scale Bar JSONS that
            were made from pairing scale labels and scale lines
        unassigned_labels (list of dicts): List of dictionaries
            representing scale bar labels that were not matched.
    """
    scale_bar_jsons = []
    paired_labels = set()
    for line in scale_bar_lines:
        x_line, y_line = boxes.find_box_center(line["geometry"])
        best_distance = 1000000
        best_label = None
        for label_index, label in enumerate(scale_bar_labels):
            x_label, y_label = boxes.find_box_center(label["geometry"])
            distance = (x_label - x_line)**2 + (y_label - y_line)**2
            if distance < best_distance:
                best_distance = distance
                best_index = label_index
                best_label = label
        # If the best match is not very good, keep this line unassigned
        if best_distance > 5000:
            best_index = -1
            best_label = None
            best_distance = -1
            continue
        paired_labels.add(best_index)
        scale_bar_json = {
            "label" : best_label,
            "geometry" : line["geometry"],
            "confidence" : float(line.get("confidence", 0)),
            "length" : line.get("length", None),
            "label_line_distance" : best_distance
        }
        scale_bar_jsons.append(scale_bar_json)
    # Check which labels were left unassigned
    unassigned_labels = []
    for i, label in enumerate(scale_bar_labels):
        if i not in paired_labels:
            unassigned_labels.append(label)
    return scale_bar_jsons, unassigned_labels

def detect_scale_objects(image, scale_bar_detection_checkpoint):
    """ Detects bounding boxes of scale bars and scale bar labels 
    Args:
        image (PIL Image): A PIL image object
    Returns:
        scale_bar_info (list): A list of lists with the following 
            pattern: [[x1,y1,x2,y2, confidence, label],...] where
            label is 1 for scale bars and 2 for scale bar labelss 
    """
    scale_bar_detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    input_features = scale_bar_detection_model.roi_heads.box_predictor.cls_score.in_features
    number_classes = 3      # background, scale bar, scale bar label
    scale_bar_detection_model.roi_heads.box_predictor = FastRCNNPredictor(input_features, number_classes)
    cuda = torch.cuda.is_available() and (gpu_id >= 0)
    if cuda:
        scale_bar_detection_model.load_state_dict(torch.load(scale_bar_detection_checkpoint)["state_dict"])
        scale_bar_detection_model = scale_bar_detection_model.cuda()
    else:
        scale_bar_detection_model.load_state_dict(torch.load(scale_bar_detection_checkpoint, map_location="cpu")["model_state_dict"])
    # prediction
    scale_bar_detection_model.eval()
    with torch.no_grad():
        outputs = scale_bar_detection_model([image])
    # post-process 
    scale_bar_info = []
    for i, box in enumerate(outputs[0]["boxes"]):
        confidence = outputs[0]["scores"][i]
        if confidence > 0.5:
            x1, y1, x2, y2 = box
            label = outputs[0]['labels'][i]
            scale_bar_info.append([x1, y1, x2, y2, confidence, label])
    scale_bar_info = non_max_suppression_malisiewicz(np.asarray(scale_bar_info), 0.4)
    return scale_bar_info

def postprocess_ctc(results):
    classes = "0123456789mMcCuUnN .A"
    idx_to_class = classes + "-"
    for result, confidence in results:
        confidence = float(confidence)
        word = ""
        for step in result:
            word += idx_to_class[step]
        word = word.strip()
        word = "".join(word.split("-"))
        print(word)
        try:
            number, unit = word.split()
            number = float(number)
            if unit.lower() == "n":
                unit = "nm"
            elif unit.lower() == "c":
                unit = "cm"
            elif unit.lower() == "u":
                unit = "um"
            if unit.lower() in ["nm", "mm", "cm", "um", "a"]:
                return number, unit, confidence
        except Exception as e:
            continue
    return -1, "m", 0

def determine_scale(figure_path, detection_checkpoint, recognition_checkpoint, figure_json=None):
    """ Adds scale information to figure by reading and measuring scale bars 

    Args:
        figure_path (str): A path to the image (.png, .jpg, or .gif)
            file containing the article figure
        figure_json (dict): A Figure JSON
    Returns:
        figure_json (dict): A dictionary with classified image_objects
            extracted from figure
    """
    if figure_json is None:
        figure_json = {}
    convert_to_nm = {
        "a"  : 0.1,
        "nm" : 1.0,
        "um" : 1000.0,
        "mm" : 1000000.0,
        "cm" : 10000000.0,
        "m"  : 1000000000.0,
    }
    unassigned = figure_json.get("unassigned", {})
    unassigned_scale_labels = unassigned.get("scale_bar_labels", [])
    master_images = figure_json.get("master_images", [])
    image = Image.open(figure_path).convert("RGB")
    tensor_image = T.ToTensor()(image)
    # Detect scale bar objects
    scale_bar_info = detect_scale_objects(tensor_image, detection_checkpoint)
    label_names = ["background", "scale bar", "scale label"]
    scale_bars = []
    scale_labels = []
    for scale_object in scale_bar_info:
        x1, y1, x2, y2, confidence, classification = scale_object
        geometry = boxes.convert_coords_to_labelbox([int(x1), int(y1),
                                                    int(x2), int(y2)])
        if label_names[int(classification)] == "scale bar":
            scale_bar_json = {
                "geometry" : geometry,
                "confidence" : float(confidence),
                "length" : int(x2 - x1)
            }
            scale_bars.append(scale_bar_json)
        elif label_names[int(classification)] == "scale label":

            scale_bar_label_image = image.crop((int(x1), int(y1),
                                                int(x2), int(y2)))
            ## Read Scale Text
            scale_label_text, label_confidence = read_scale_bar_label(
                scale_bar_model, scale_bar_label_image)
            if scale_label_text is None:
                print("non label detected")
                continue
            magnitude, unit = scale_label_text.split(" ")
            magnitude = float(magnitude)
            length_in_nm = magnitude * convert_to_nm[unit.strip().lower()]
            label_json = {
                "geometry" : geometry,
                "text" : scale_label_text,
                "label_confidence" : float(label_confidence),
                "box_confidence" : float(confidence),
                "nm" : length_in_nm,
                "unit": unit
            }
            scale_labels.append(label_json)
    # Match scale bars to labels and to subfigures (master images)
    scale_bar_jsons, unassigned_labels = (
        create_scale_bar_objects(scale_bars, scale_labels))
    
    return scale_bar_jsons

def match_scale_bars(correct, predicted):
    matched = []
    paired_predicted = set()
    unmatched_correct = 0
    for correct_scale in correct:
        x_correct, y_correct = boxes.find_box_center(correct_scale["geometry"])
        best_distance = None
        best_prediction = None
        best_index = None
        for predicted_index, predicted_scale in enumerate(predicted):
            if predicted_index in paired_predicted:
                continue
            x_predicted, y_predicted = boxes.find_box_center(predicted_scale["geometry"])
            distance = (x_predicted - x_correct)**2 + (y_predicted - y_correct)**2
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_index = predicted_index
                best_prediction = predicted_scale
        # If the best match is not very good, keep this line unassigned
        if best_distance is None or best_distance > 100:
            best_index = -1
            best_prediction = None
            best_distance = -1
            unmatched_correct += 1
            continue
        paired_predicted.add(best_index)
        matching = (correct_scale, best_prediction)
        matched.append(matching)
    unmatched_predicted = len(predicted) - len(paired_predicted)
    return matched, unmatched_correct, unmatched_predicted

def super_resolution(image):
    current_file = pathlib.Path(__file__).resolve(strict=True)
    model = "LapSRN_x8.pb"
    modelName = model.split(os.path.sep)[-1].split("_")[0].lower()
    modelScale = model.split("_x")[-1]
    modelScale = int(modelScale[:modelScale.find(".")])


    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    model_path = current_file.parent / model
    sr.readModel(str(model_path))
    sr.setModel(modelName, modelScale)
    image = np.array(image)
    image = sr.upsample(image)
    image = Image.fromarray(image)
    return image

def read_scale_bar_label(scale_bar_model, scale_bar_label_image):
    # preprocess
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #scale_bar_label_image = super_resolution(scale_bar_label_image)
    resize_transform = transforms.Compose([
        transforms.Resize((128, 512)),
        transforms.Lambda(convert_to_rgb),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])       
    classes = "0123456789mMcCuUnN .A"
    idx_to_class = classes + "-"
    image = resize_transform(scale_bar_label_image)
    image = image.unsqueeze(0)
    image = image.to(device)
    # run image on model
    logps = scale_bar_model(image)
    probs = torch.exp(logps)
    probs = probs.squeeze(0)
    # postprocess
    language_model_file ="corpus.txt"
    current_file = pathlib.Path(__file__).resolve(strict=True)
    language_model = LanguageModel(current_file.parent / language_model_file, classes)
    top_results = ctcBeamSearch(probs, classes, lm=language_model, beamWidth=15)

    magnitude, unit, confidence = postprocess_ctc(top_results)
    convert_to_nm = {
        "a"  : 0.1,
        "nm" : 1.0,
        "um" : 1000.0,
        "mm" : 1000000.0,
        "cm" : 10000000.0,
        "m"  : 1000000000.0,
    }
    nm = magnitude * convert_to_nm[unit.strip().lower()]
    return {"unit": unit, "number": magnitude, "label_confidence": float(confidence), "nm": nm}

def split_label(text):
    """ Split a scale bar label into a number and a unit """
    text = text.strip()
    for i, char in enumerate(text):
        try:
            float(text[:i+1])
        except:
            if i > 0:
                return text[:i].strip(), ''.join(text[i:].strip())
            break
    return 0, None

def test_label_reading(model_name, epoch=None):
    """ Run test training set on the specified model and checkpoint

    Args:
        model_name (str): Name of model
        epoch (int): Epoch number or None to use highest epoch checkpoint
    """
    # load test images and test image labels
    current_file = pathlib.Path(__file__).resolve(strict=True)
    exsclaim_root = current_file.parent.parent.parent.parent
    tests = exsclaim_root / 'exsclaim' / 'tests' / 'data'
    test_json = tests / "scale_bars_dataset_test.json"
    train_json = tests/ "scale_bars_dataset_train.json"
    test_images = tests / 'images' / 'labeled_data'
    # load scale bar model
    checkpoint_directory = exsclaim_root / "training" / "checkpoints"
    if epoch is None:
        epoch = (
            sorted([int(file.split("-")[-1].split(".")[0]) 
            for file in os.listdir(checkpoint_directory) 
            if file.split("-")[0] == model_name])[-1]
        )
    scale_bar_label_checkpoint = (
        checkpoint_directory / (model_name + "-" + str(epoch) + ".pt")
    )
    try:
        all_configurations_path = (
            exsclaim_root / "training" / "scale_label_reader.json"
        )
        with open(all_configurations_path, "r") as f:
            all_configurations = json.load(f)
        configuration = all_configurations[model_name]
    except Exception as e:
        print(e)
        configuration = None
    scale_bar_model = CRNN(configuration=configuration)
    cuda = torch.cuda.is_available()
    if cuda:
        checkpoint = torch.load(scale_bar_label_checkpoint)
        scale_bar_model = scale_bar_model.cuda()
    else:
        checkpoint = torch.load(scale_bar_label_checkpoint, map_location='cpu')
    scale_bar_model.load_state_dict(checkpoint["model_state_dict"])
    scale_bar_model.eval()
    # save constants
    convert_to_nm = {
        "a"  : 0.1,
        "nm" : 1.0,
        "um" : 1000.0,
        "mm" : 1000000.0,
        "cm" : 10000000.0,
        "m"  : 1000000000.0,
    }
    # initialize results lists
    correct_nms = []
    predicted_nms = []
    nm_percentage_off = []
    confidences = []
    cunits = []
    punits = []
    cnums = []
    pnums = []
    incorrect = 0
    # combine test and training data (reader was trained on synthetic data)
    with open(test_json, "r") as f:
        test_dict = json.load(f)
    with open(train_json, "r") as f:
        train_dict = json.load(f)
    test_dict.update(train_dict)
    keys = list(test_dict.keys())
    random.shuffle(keys)
    # test model on each scale bar
    for k, figure_name in enumerate(keys):
        correct_labels = []
        predicted_labels = []
        figure = Image.open(test_images / figure_name).convert("RGB")
        for label in test_dict[figure_name].get("scale_labels", []):
            geometry = label.get("geometry")
            text = label["text"]
            magnitude, unit = split_label(text.strip())
            if unit not in convert_to_nm:
                print("unit is none ", text)
                continue
            magnitude = float(magnitude)
            nm = magnitude * convert_to_nm[unit.strip().lower()]
            correct_labels.append({"geometry":geometry, "nm": nm, "number": magnitude, "unit": unit})
                        
            subfigure = figure.crop((boxes.convert_labelbox_to_coords(geometry)))
            predicted_label = read_scale_bar_label(scale_bar_model, subfigure)
            if predicted_label is None:
                continue
            if predicted_label["nm"] != nm:
                incorrect += 1
                subfigure.save("incorrect/" + str(predicted_label["nm"]) + figure_name)
            predicted_labels.append(predicted_label)
            print("Correct: {}\tPredicted: {} {}".format(text, predicted_label["number"], predicted_label["unit"]))

        for correct, predicted in zip(correct_labels, predicted_labels):
            correct_nm = correct["nm"]
            predicted_nm = predicted["nm"]
            confidence = predicted["label_confidence"]
            correct_nms.append(correct_nm)
            predicted_nms.append(predicted_nm)
            confidences.append(confidence)
            nm_percentage_off.append(abs(predicted_nm - correct_nm)/ correct_nm)
            punits.append(predicted["unit"])
            cunits.append(correct["unit"])
            pnums.append(predicted["number"])
            cnums.append(correct["number"])
        
        # if k > 50:
        #     break
    #print(incorrect / len(cunits))
    with open(str(model_name) + "-" + str(epoch) + "nosrn_newlm.json", "w+") as f:
        results = {
            "correct_nms": correct_nms, 
            "predicted_nms": predicted_nms,
            "confidences": confidences,
            "correct_numbers": cnums,
            "predicted_numbers": pnums,
            "predicted_units": punits,
            "correct_units": cunits
        }
        json.dump(results, f)


if __name__ == "__main__":
    current_file = pathlib.Path(__file__).resolve(strict=True)
    recognition_checkpoint = current_file.parent.parent.parent.parent / "training" /  "checkpoints" / "alpha-160.pt"
    test_label_reading(recognition_checkpoint)
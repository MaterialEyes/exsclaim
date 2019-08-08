import os
import glob
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import namedtuple


def labelbox_to_namedtuple(lb):
    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    return Rectangle(lb[0]["x"],lb[0]["y"],lb[2]["x"],lb[2]["y"])

def area(a, b):  # returns None if rectangles don't intersect
    a = labelbox_to_namedtuple(a)
    b = labelbox_to_namedtuple(b)
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx>=0) and (dy>=0):
        return dx*dy

def between_dist(a, b):  # returns None if rectangles don't intersect
    a = labelbox_to_namedtuple(a)
    b = labelbox_to_namedtuple(b)
    ax = (a.xmin + float((a.xmax-a.xmin))/2)
    ay = (a.ymin + float((a.ymax-a.ymin))/2)
    bx = (b.xmin + float((b.xmax-b.xmin))/2)
    by = (b.ymin + float((b.ymax-b.ymin))/2)
    dx = np.abs(ax-bx)
    dy = np.abs(ay-by)
    return np.sqrt(dx**2+dy**2)

with open(os.getcwd()+'/temp.json', "r") as read_file:
    data = json.load(read_file)

d = []
for entry in data:

    # Collect all the unassigned images 
    unassigned_images = entry["subfigs"]["unassigned"]["bbox"]["image"]
    
    s = dict(zip(range(len(unassigned_images)), ["none"]*len(unassigned_images)))    

    # First take unassigned scale bar bboxs and assign to images based on greatest intersection area
    for idx in range(len(entry["subfigs"]["unassigned"]["bbox"]["scale"])):

        scalebar_bbox  = [eval(b) for b in entry["subfigs"]["unassigned"]["bbox"]["scale"][idx]]
        scalebar_label = entry["subfigs"]["unassigned"]["info"]["scale"][idx]

        # Compute intersection area (bbox of scalebar vs bbox of subfigure image)
        intersect_areas  = np.zeros(len(unassigned_images))

        for i in range(len(unassigned_images)):
            
            subfigure_bbox = [eval(a) for a in unassigned_images[i]]
            
            overlap_area = area(scalebar_bbox ,subfigure_bbox)
            
            if overlap_area != None:
                intersect_areas[i] = overlap_area

        if np.sum(intersect_areas)>0:
            corr_idx = np.argmax(intersect_areas)
            s[corr_idx] = scalebar_label
        else:
            s[corr_idx] = ""

    # Release unassigned scale text and scalebar bbox
    entry["subfigs"]["unassigned"]["info"]["scale"] = []
    entry["subfigs"]["unassigned"]["bbox"]["scale"] = []

    # Only iterate through sublabel keys which have an been assigned a sublabel bbox
    for key in [a for a in entry["subfigs"].keys() if entry["subfigs"][a]["bbox"]["label"] != []]:

        # Get all unassigned bounding boxes
        sublabel_bbox = [eval(a) for a in entry["subfigs"][key]["bbox"]["label"][0]]

        # Compute intersection area (bbox of subfigure label vs bbox of subfigure image)
        intersect_areas  = np.zeros(len(unassigned_images))
        
        # Compute distance between subfigure label and subfigure images (distance between centroids)
        distance_between = np.zeros(len(unassigned_images))
        
        for i in range(len(unassigned_images)):
            
            subfigure_bbox = [eval(a) for a in unassigned_images[i]]
            
            overlap_area = area(sublabel_bbox,subfigure_bbox)
            distance_to  = between_dist(sublabel_bbox,subfigure_bbox)
            
            if overlap_area != None:
                intersect_areas[i] = overlap_area
            
            if distance_to != None:
                distance_between[i] = distance_to
            else:
                distance_between[i] = 1e5

        # Overlap is given priority...
        if np.sum(intersect_areas)>0:
            corr_idx = np.argmax(intersect_areas)
        # Select closest dependent bbox
        else:
            corr_idx = np.argmin(distance_between)

        if entry["subfigs"][key]["info"]["keywords"] != []:
            
            # Assign image to subfigure label and remove from unassigned
            entry["subfigs"][key]["bbox"]["image"] = unassigned_images[corr_idx]
            entry["subfigs"]["unassigned"]["bbox"]["image"].remove(unassigned_images[corr_idx])

            # Assign scalebar to subfigure label
            if s[corr_idx] != 'none':
                entry["subfigs"][key]["info"]["scale"] = s[corr_idx]

    d.append(entry)

with open(os.getcwd()+'/assigned.json', 'w') as fout:
    json.dump(d, fout)


os.remove(os.getcwd()+'/temp.json')

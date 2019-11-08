import argparse
import json
from operator import itemgetter


def parse_command_line():
    ap = argparse.ArgumentParser()
    
    ap.add_argument('-e', '--exsclaim_json', type=str,
                    help='Path to exsclaim JSON (https://gitlab.com/MaterialEyes/exsclaim/wikis/JSON-Format)')

    ap.add_argument('-t', '--testing', type=str, default='False',
                    help='true if you want to feed a labelbox json instead')

    args = vars(ap.parse_args())
    
    return args


def calculate_match_score(object_one, object_two):
    """ finds score where higher numbers indicate stronger relation """
    overlap = calculate_overlap(object_one, object_two)
    if overlap > 0:
        return 100 * overlap
    else:
        return -1 * calculate_distance(object_one, object_two)


def calculate_overlap(object_one, object_two):
    """ Calculates the % of the smaller object contained in the larger """
    coords_one = find_coords(object_one)
    coords_two = find_coords(object_two)
    
    # Find larger object
    area_one = find_area(coords_one)
    area_two = find_area(coords_two)
    if area_one >= area_two:
        large = coords_one
        small = coords_two
    else:
        large = coords_two
        small = coords_one
        
    # Find area of intersection
    intersect_height = find_intersection_length(large[2:], small[2:])
    intersect_width = find_intersection_length(large[:2], small[:2])
    intersection = intersect_height * intersect_width
    
    return intersection / min(area_one, area_two)


def calculate_distance(object_one, object_two):
    """ Calculates the shortest distance of object_one to object_two """
    x1, x1b, y1, y1b = find_coords(object_one)
    x2, x2b, y2, y2b = find_coords(object_two)
    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    if top and left:
        return dist((x1, y1b), (x2b, y2))
    elif left and bottom:
        return dist((x1, y1), (x2b, y2b))
    elif bottom and right:
        return dist((x1b, y1), (x2, y2b))
    elif right and top:
        return dist((x1b, y1b), (x2, y2))
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    else:             # rectangles intersect
        return 0.    


def dist(p1, p2):
    """ calculates euclidean distance """
    x1, y1 = p1
    x2, y2 = p2
    return ((x2 - x1)**2 + (y2 - y1)**2)**(0.5)    


def find_coords(object_geometry):
    """ Conversts geometry coords to (x1, x2, y1, y2) tuple """
    x_coords = [object_geometry[i]["x"] for i in range(len(object_geometry))]
    y_coords = [object_geometry[i]["y"] for i in range(len(object_geometry))]
    object_coords = (min(x_coords), max(x_coords),
                     min(y_coords), max(y_coords))
    return object_coords


def find_area(object_coords):
    """ Finds area of (x1, x2, y1, y2) tuple """
    height = object_coords[3] - object_coords[2]
    width = object_coords[1] - object_coords[0] 
    return height * width


def find_intersection_length(a, b):
    """ finds length for which two lines intersect """
    a1, a2 = a
    b1, b2 = b

    if b1 > a2 or a1 > b2:
        return 0
    if a2 > b2:
        if a1 > b1:
            return b2 - a1
        else:
            return b2 - b1
    else:
        if a1 > b1:
            return a2 - a1
        else:
            return a2 - b1


def assign_subfigure_labels(figure):
    """ Matches Subfigure Labels with master_images 
    
    param figure: a Figure JSON
    returns: (masters, unassigned) where masters is a list of
        master_images JSONs and unassigned is the unassigned JSON 
    """
    masters = []
    unassigned = figure["unassigned"]
    num_masters = len(unassigned["master_images"])

    if num_masters == 0:
        return masters, unassigned

    # track which master_images and labels have been paired
    label_to_label = {}
    master_index_to_master = {}
    assigned = set()
    not_assigned = set()
    scores = [] # each elt is: (score, label, master_index)
    
    # look through the unassigned, add them to tracking
    for index, master in enumerate(unassigned["master_images"]):
        master_index_to_master[index] = master
        not_assigned.add(index)

    # if there are no labels, there should be one master image and a caption with "0" as its label text
    null_label = {"geometry" : [{"x" : 0, "y": 0}, {"x" : 1, "y" : 1}], 
                  "text" : "0"}

    # get list of unassigned figure labels
    unassigned_labels = unassigned["subfigure_labels"] if len(unassigned["subfigure_labels"]) > 0 else [null_label]

    # split low/high confidence labels based on text transcription
    low_confidence_labels = [a for a in unassigned_labels if a['text'] == ""]
    high_confidence_labels = [a for a in unassigned_labels if a['text'] != ""]

    # only consider subfigure labels with high-confidence text transcription!
    # note: text field = "0", single master image with no subfigure label (still high confidence)
    for label in high_confidence_labels:
        label_geometry = label["geometry"]
        label_to_label[label["text"]] = label
        not_assigned.add(label["text"])
        for master_index in master_index_to_master:
            master_geometry = master_index_to_master[master_index]["geometry"]
            score = calculate_match_score(master_geometry, label_geometry)
            scores.append((score, label["text"], master_index))

    # greedily assign the highest scoring pair 
    scores = sorted(scores, key=itemgetter(0))
    iters = len(scores)
    for i in range(iters):
        best = scores.pop()
        score, label, master_index = best
        if label in assigned or master_index in assigned:
            continue

        # these subfigure_labels and master_images will be assigned
        assigned.add(label)
        assigned.add(master_index) 
        not_assigned.remove(label)
        not_assigned.remove(master_index)    

        # create a master_images JSON
        master_json = {}
        label_json = {"text" : label,
                      "geometry": label_to_label[label]["geometry"]}
        master_json = {"label" : label_json, 
                       "geometry" : master_index_to_master[master_index]["geometry"]}
        if "classification" in master_index_to_master[master_index]:
            master_json["classification"] = master_index_to_master[master_index]["classification"]
        masters.append(master_json)
    
    # update unassigned JSON
    unassigned_masters = []
    unassigned_labels = []
    for i in not_assigned:
        if i in master_index_to_master:
            unassigned_masters.append(master_index_to_master[i])
        else:
            unassigned_labels.append(label_to_label[i])

    unassigned["master_images"] = unassigned_masters
    unassigned["subfigure_labels"] = unassigned_labels+low_confidence_labels 
    
    return masters, unassigned


def make_scale_bars(figure):
    """ matches all scale bar lines to the nearest scale bar labels 

    param figure: a Figure JSON

    returns (scale bars, unassigned): where scale bars is a list of 
        scale_bar JSONs and unassigned is the updated unassigned JSON 
    """
    unassigned = figure["unassigned"]
    scale_bar_lines = unassigned.get("scale_bar_lines", [])
    scale_bars = []

    # return scale_bar JSONs with no label
    if unassigned.get("scale_bar_labels", []) == []:
        return scale_bars, unassigned
    
    # create mappings to jsons
    not_assigned = set()
    index_to_json = {}
    for index, label in enumerate(unassigned["scale_bar_labels"]):
        index_to_json[index] = label
        not_assigned.add(index)

    # match Lines to Labels
    for line in scale_bar_lines:
        scores = []  # (score, label) tuples
        line_geo = line["geometry"] 
        for label_index in index_to_json:
            label = index_to_json[label_index]
            label_geo = label["geometry"]
            score = calculate_match_score(line_geo, label_geo)
            scores.append((score, label_index))

        # use highest scoring label
        best_index = max(scores, key=itemgetter(0))[1]
        not_assigned.discard(best_index)
        best_match = index_to_json[best_index]
        # create scale_bar JSON
        label_json = {"text" : best_match["text"], 
                      "geometry" : best_match["geometry"]}
        scale_bar_json = {"label" : label_json, "geometry" : line_geo}
        scale_bars.append(scale_bar_json)
    
    # update unassigned
    unassigned_labels = []
    for index in not_assigned:
        unassigned_labels.append(index_to_json[index])
    unassigned["scale_bar_labels"] = unassigned_labels
    unassigned["scale_bar_lines"] = []        
    return scale_bars, unassigned


def assign_dependent_images(figure):
    """ Assigns all Dependent Images in unassigned to a master_images 
    
    param figure: a Figure JSON

    returns (masters, unassigned): where masters is a list of master_images
        JSONs and unassinged is the updated unassigned JSON
    """
    unassigned = figure["unassigned"]
    dependent_images = unassigned.get("dependent_images", [])
    masters = figure.get("master_images", [])
    
    # if there is nothing new here...
    if len(masters) * len(dependent_images) == 0:
        return masters, unassigned
    
    unassigned_dependents = []
    for dependent in dependent_images:
        matched = False
        for master in masters:
            score = calculate_match_score(dependent["geometry"],
                                          master["geometry"])
            # if a dependent is mostly in a master_images, match them
            if score > 50:
                matched = True
                break
        # if the dependent is not mostly in any master, keep it unassigned
        if not matched:
            unassigned_dependents.append(dependent)
            continue
        master_dependents = master.get("dependent_images", [])
        assigned_dependent = {"geometry" : dependent["geometry"], \
                              "classification": dependent["classification"]}
        master_dependents.append(assigned_dependent)
        master["dependent_images"] = master_dependents
    
    unassigned["dependent_images"] = unassigned_dependents
    return masters, unassigned


def assign_inset_images(figure):
    """ Assigns all Inset Images in unassigned to a master_images 
    
    param figure: a Figure JSON

    returns (masters, unassigned): where masters is a list of master_images
        JSONs and unassinged is the updated unassigned JSON
    """
    unassigned = figure["unassigned"]
    inset_images = unassigned.get("inset_images", [])
    masters = figure.get("master_images", [])
    
    # if there is nothing new here...
    if len(masters) * len(inset_images) == 0:
        return masters, unassigned
    
    unassigned_insets = []
    for inset in inset_images:
        matched = False
        for master in masters:
            score = calculate_match_score(inset["geometry"],
                                          master["geometry"])
            # if a dependent is mostly in a master_images, match them
            if score > 50:
                matched = True
                break
        # if the dependent is not mostly in any master, keep it unassigned
        if not matched:
            unassigned_insets.append(inset)
            continue
        master_insets = master.get("inset_images", [])
        assigned_inset = {"geometry" : inset["geometry"], \
                          "classification": inset["classification"]}
        master_insets.append(assigned_inset)
        master["inset_images"] = master_insets
    
    unassigned["inset_images"] = unassigned_insets
    return masters, unassigned


def assign_scale_bars(figure, scale_bars):
    """ Assigns all scale_bar JSON  unassigned to a Subfigure 
    
    param figure: a Figure JSON
    param scale_bars: a list of scale_bar JSONs

    returns (masters, unassigned): where masters is a list of master_images
        JSONs and unassinged is the updated unassigned JSON
    """
    unassigned = figure["unassigned"]
    new_masters = []
    old_masters = figure.get("master_images", [])
    # track unassigned scale_bars
    not_assigned = {i for i in range(len(scale_bars))}

    # assign scale bars to master_images
    for master in old_masters:
        master_scale_bars = []
        for i, scale_bar in enumerate(scale_bars):
            score = calculate_match_score(master["geometry"],
                                          scale_bar["geometry"])
            if score > 50:
                master_scale_bars.append(scale_bar)
                not_assigned.discard(i)
        # assign scale bars to dependent/inset w/in master or to master
        new_master = assign_master_scale_bars(master, master_scale_bars)
        new_masters.append(new_master)
    
    # collect unassigned scalebars
    unassigned_scale_bars = [scale_bars[i] for i in not_assigned]
    unassigned["scale_bar"] = unassigned_scale_bars
    
    return new_masters, unassigned


def assign_master_scale_bars(master, scale_bars):
    """ Assigns each scale bar to a portion of the master_images

    param master: a master_images JSON
    param scale_bars: a list of scale_bar JSONs contained within master

    returns new_master: a master_images JSON incorporating scale_bars
    """
    insets = master.get("inset_images", [])
    dependents = master.get("dependent_images", [])

    assigned = set()
    
    for inset in insets:
        for index, scale_bar in enumerate(scale_bars):
            if index in assigned:
                continue
            score = calculate_match_score(inset["geometry"], 
                                          scale_bar["geometry"])
            if score > 50:
                assigned.add(index)
                inset["scale_bar"] = scale_bar
    for dependent in dependents:
        for index, scale_bar in enumerate(scale_bars):
            if index in assigned:
                continue
            score = calculate_match_score(dependent["geometry"],
                                          scale_bar["geometry"])
            if score > 50:
                assigned.add(index)
                dependent["scale_bar"] = scale_bar

    master_bars = master.get("scale_bar", [])
    for index, scale_bar in enumerate(scale_bars):
        if index not in assigned:
            master_bars.append(scale_bar)

    try:
        master_bars_dict = master_bars[0]
    except:
        master_bars_dict = {}

    master["scale_bar"] = master_bars_dict
    master["inset_images"] = insets
    master["dependent_images"] = dependents
    return master


def assign_captions(figure):
    """ Assigns all captions to master_images JSONs

    param figure: a Figure JSON

    returns (masters, unassigned): where masters is a list of master_images
        JSONs and unassigned is the updated unassigned JSON
    """
    unassigned = figure.get("unassigned", [])
    masters = []

    captions = unassigned.get("captions", {})

    # not_assigned = set(captions.keys())
    not_assigned = set([a['label'] for a in captions])
    for index, master_image in enumerate(figure.get("master_images", [])):
        label_json = master_image.get("subfigure_label", {})
        subfigure_label = label_json.get("text", index)            
        processed_label = subfigure_label.replace(")","")
        processed_label = processed_label.replace("(","")
        processed_label = processed_label.replace(".","")
        paired = False
        for caption_label in captions:
            processed_caption_label = caption_label['label'].replace(")","")
            processed_capiton_label = processed_caption_label.replace("(","")
            processed_caption_label = processed_caption_label.replace(".","")
            if (processed_caption_label.lower() == processed_label.lower()) and \
               (processed_caption_label.lower() in [a.lower() for a in not_assigned]):
                # master_image["caption"] = captions[caption_label]["caption"]
                master_image["caption"] = caption_label['description']
                # master_image["keywords"] = captions[caption_label]["keywords"]
                master_image["keywords"] = caption_label['keywords']
                master_image["general"] = caption_label['general']
                masters.append(master_image)
                not_assigned.remove(caption_label['label'])
                paired = True
                break
        if paired:
            continue

        master_image["caption"] = []
        master_image["keywords"]= []
        master_image["general"] = []
        masters.append(master_image)

    # new_unassigned_captions = {}
    new_unassigned_captions = []
    for caption_label in captions:
        if caption_label['label'] in not_assigned:
            # new_unassigned_captions[caption_label] = captions[caption_label]
            new_unassigned_captions.append(caption_label)

    unassigned["captions"] = new_unassigned_captions
    return masters, unassigned


def cluster_figure(figure):
    masters, unassigned = assign_subfigure_labels(figure)
    figure["master_images"] = masters
    figure["unassigned"] = unassigned
 
    masters, unassigned = assign_inset_images(figure)
    figure["master_images"] = masters
    figure["unassigned"] = unassigned    

    masters, unassigned = assign_dependent_images(figure)
    figure["master_images"] = masters
    figure["unassigned"] = unassigned

    scale_bars, unassigned = make_scale_bars(figure) 
    figure["unassigned"] = unassigned

    masters, unassigned = assign_scale_bars(figure, scale_bars)
    figure["master_images"] = masters
    figure["unassigned"] = unassigned

    masters, unassigned = assign_captions(figure)
    figure["master_images"] = masters
    figure["unassigned"] = unassigned
     
    return figure 

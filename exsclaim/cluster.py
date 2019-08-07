import json
from operator import itemgetter


def cluster_subfigures(data):
    figure_json = {}
    masters = data["unassigned"]["Master Image"]
    if len(masters) == 1:
        pass
    for master in data["unassigned"]["Master Image"]:
        master_json = create_master_json(master)
    
def find_dependent_images(master):
    return []

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
    """ Matches Subfigure Labels with Master Image 
    
    param figure: a Figure JSON
    returns: (masters, unassigned) where masters is a list of
        Master Image JSONs and unassigned is the unassigned JSON 
    """
    masters = []
    unassigned = figure["unassigned"]
    num_masters = len(unassigned["Master Image"])
    if num_masters == 0:
        return masters, unassigned
    # track which master images and labels have been paired
    label_to_label = {}
    master_index_to_master = {}
    assigned = set()
    not_assigned = set()
    scores = [] # each elt is: (score, label, master_index)
    # look through the unassigned, add them to tracking
    for index, master in enumerate(unassigned["Master Image"]):
        master_index_to_master[index] = master
        not_assigned.add(index)
    for label in unassigned["Subfigure Label"]:
        label_geometry = label["geometry"]
        label_to_label[label["text"]] = label
        not_assigned.add(label["text"])
        for master_index in master_index_to_master:
            master_geometry = master_index_to_master[master_index]["geometry"]
            score = calculate_match_score(master_geometry, label_geometry)
            scores.append((score, label["text"], master_index))
    # greedily assigne the highest scoring pair 
    scores = sorted(scores, key=itemgetter(0))
    iters = len(scores)
    for i in range(iters):
        best = scores.pop()
        score, label, master_index = best
        if label in assigned or master_index in assigned:
            continue
        # this Subfigure Label and Master Image will be assigned
        assigned.add(label)
        assigned.add(master_index) 
        not_assigned.remove(label)
        not_assigned.remove(master_index)       
        # create a Master Image JSON
        master_json = {}
        label_json = {"text" : label,
                      "geometry": label_to_label[label]["geometry"]}
        master_json = {"label" : label_json, 
                       "geometry" : master_index_to_master[master_index]["geometry"]}
        ##TODO: also add classification from master object
        masters.append(master_json)
    
    # update unassigned JSON
    unassigned_masters = []
    unassigned_labels = []
    for i in not_assigned:
        if i in master_index_to_master:
            unassigned_masters.append(master_index_to_master[i])
        else:
            unassigned_labels.append(label_to_label[i])
    unassigned["Master Image"] = unassigned_masters
    unassigned["Subfigure Label"] = unassigned_labels
    
    return masters, unassigned

def make_scale_bars(figure):
    """ matches all scale bar lines to the nearest scale bar labels 

    param figure: a Figure JSON

    returns (scale bars, unassigned): where scale bars is a list of 
        Scale Bar JSONs and unassigned is the updated unassigned JSON 
    """
    unassigned = figure["unassigned"]
    scale_bar_lines = unassigned.get("Scale Bar Line", [])
    scale_bars = []

    # return Scale Bar JSONs with no label
    if unassigned.get("Scale Bar Label", []) == []:
        return scale_bars, unassigned
    
    # create mappings to jsons
    not_assigned = set()
    index = 0
    index_to_json = {}
    for label in unassigned["Scale Bar Label"]:
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
        # create Scale Bar JSON
        label_json = {"text" : best_match["text"], 
                      "geometry" : best_match["geometry"]}
        scale_bar_json = {"label" : label_json, "geometry" : line_geo}
        scale_bars.append(scale_bar_json)
    
    # update unassigned
    unassigned_labels = []
    for index in not_assigned:
        unassigned_labels.append(index_to_json[index])
    unassigned["Scale Bar Label"] = unassigned_labels
    unassigned["Scale Bar Line"] = []        
    return scale_bars, unassigned


def assign_dependent_images(figure):
    """ Assigns all Dependent Images in unassigned to a Master Image 
    
    param figure: a Figure JSON

    returns (masters, unassigned): where masters is a list of Master Image
        JSONs and unassinged is the updated unassigned JSON
    """
    unassigned = figure["unassigned"]
    dependent_images = unassigned.get("Dependent Image", [])
    masters = figure.get("Master Image", [])
    
    # if there is nothing new here...
    if len(masters) * len(dependent_images) == 0:
        return masters, unassigned
    
    unassigned_dependents = []
    for dependent in dependent_images:
        matched = False
        for master in masters:
            score = calculate_match_score(dependent["geometry"],
                                          master["geometry"])
            # if a dependent is mostly in a Master Image, match them
            if score > 50:
                matched = True
        # if the dependent is not mostly in any master, keep it unassigned
        if not matched:
            unassigned_dependents.append(dependent)
            continue
        master_dependents = master.get("Dependent Images", [])
        assigned_dependent = {"geometry" : dependent["geometry"]}
        master_dependents.append(assigned_dependent)
        master["Dependent Images"] = master_dependents
    
    unassigned["Dependent Image"] = unassigned_dependents
    return masters, unassigned


def assign_inset_images(figure):
    """ Assigns all Inset Images in unassigned to a Master Image 
    
    param figure: a Figure JSON

    returns (masters, unassigned): where masters is a list of Master Image
        JSONs and unassinged is the updated unassigned JSON
    """
    unassigned = figure["unassigned"]
    inset_images = unassigned.get("Inset Image", [])
    masters = figure.get("Master Image", [])
    
    # if there is nothing new here...
    if len(masters) * len(inset_images) == 0:
        return masters, unassigned
    
    unassigned_insets = []
    for inset in inset_images:
        matched = False
        for master in masters:
            score = calculate_match_score(inset["geometry"],
                                          master["geometry"])
            # if a dependent is mostly in a Master Image, match them
            if score > 50:
                matched = True
        # if the dependent is not mostly in any master, keep it unassigned
        if not matched:
            unassigned_insets.append(inset)
            continue
        master_insets = master.get("Inset Images", [])
        assigned_inset = {"geometry" : inset["geometry"]}
        master_insets.append(assigned_inset)
        master["Inset Images"] = master_Insets
    
    unassigned["Inset Image"] = unassigned_insets
    return masters, unassigned

def assign_scale_bars(figure, scale_bars):
    """ Assigns all Scale Bar JSON  unassigned to a Subfigure 
    
    param figure: a Figure JSON
    param scale_bars: a list of Scale Bar JSONs

    returns (masters, unassigned): where masters is a list of Master Image
        JSONs and unassinged is the updated unassigned JSON
    """
    pass

def assign_captions(figure):
    """ Assigns all captions to Master Image JSONs

    param figure: a Figure JSON

    returns (masters, unassigned): where masters is a list of Master Image
        JSONs and unassigned is the updated unassigned JSON
    """
    pass

with open("exsclaim.json", "r") as f:
    exsclaim_json = json.load(f)

for figure in exsclaim_json:
    figure = exsclaim_json[figure]
    print("\n\n", assign_subfigure_labels(figure))
    

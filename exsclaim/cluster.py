import json




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



with open("exsclaim.json", "r") as f:
    exsclaim_json = json.load(f)

for figure in exsclaim_json:
    figure = exsclaim_json[figure]
    print(figure["article_url"])
    print(figure["image_url"])
    master_to_sub = {}
    unassigned = figure["unassigned"]
    for master in unassigned["Master Image"]:
        master_geo = master["geometry"]
        master_to_sub[master["confidence"]] = []
        for label in unassigned["Subfigure Label"]:
            label_geo = label["geometry"]
            master_to_sub[master["confidence"]].append((label["text"], calculate_match_score(master_geo, label_geo)))
    break

print(master_to_sub)

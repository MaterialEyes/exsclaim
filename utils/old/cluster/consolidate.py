import os
import yaml
import json
import glob
import difflib 
import scipy.misc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import match_template

def find_bbox(image,template,display=False):
    """
    Finds a bounding box by you can move the cropped patches around the original image and 
    calculate the minimum difference point

    :param image: The main image containing template
    :param template: The template to be moved around the image

    :return: Bounding box in labelbox format
    """
    image = image[:,:,0]
    coin  = template[:,:,0]

    result = match_template(image, coin)
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]
    
    hcoin, wcoin = coin.shape
    rect = plt.Rectangle((x, y), wcoin, hcoin, edgecolor='r', facecolor='none')

    if display:
        fig = plt.figure(figsize=(8, 3))
        ax1 = plt.subplot(1, 3, 1)
        ax2 = plt.subplot(1, 3, 2)
        ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)

        ax1.imshow(coin, cmap=plt.cm.gray)
        ax1.set_axis_off()
        ax1.set_title('template')

        ax2.imshow(image, cmap=plt.cm.gray)
        ax2.set_axis_off()
        ax2.set_title('image')
        ax2.add_patch(rect)

        ax3.imshow(result)
        ax3.set_axis_off()
        ax3.set_title('match_template result')
        ax3.autoscale(False)
        ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)
        plt.show()

    return [{"x":x,"y":y},{"x":x,"y":y+hcoin},{"x":x+wcoin,"y":y+hcoin},{"x":x+wcoin,"y":y}]

def load_label_yaml(folder):
    """
    Open image label text results (yaml from ASTER)
    """   
    with open(os.getcwd()+"/"+folder+"/image_labels.yaml", "r") as stream:
        try:
            yf = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return yf

def image_list(folder):
    return glob.glob(os.getcwd()+"/"+folder+"/*.png")

def related_subset(full_list,row):
    return [a for a in full_list if row["article"].split(".")[0][0:-1]+"_"+row["fig id"] in a.split("origin_")[-1]]

def sublabel_char_resolve(char,char_type):
    """
    Resolves common alpha/numeric errors made by ASTER
    """
    if char_type.lower() == "alpha":
        return char.replace("1","i")
    else:
        return char

def scalebar_char_resolve(char):
    """
    Resolves common alpha/numeric errors made by ASTER
    """
    char = char.replace("-m","nm")
    char = char.replace("So","50")
    char = char.replace("so","50")
    return char

def populate_image_dict(row,fig_img,subfig_img,scale_img,label_img,scale_dict,label_dict):
    
    # Organize all paths related to the row entry
    figure_image_path     = related_subset(fig_img,row)[0]
    subfigure_image_paths = related_subset(subfig_img,row)
    scalebar_image_paths  = related_subset(scale_img,row)
    sublabel_image_paths  = related_subset(label_img,row)

    # Image associated with the figure described in the row
    figure_image          = scipy.misc.imread(figure_image_path)
    
    # List of explicit sublabels implied in caption text (vocabulary for text recognition refinement)
    explicit_sublabels  = [a.lower() for a in list(eval(row["caption text (explicit)"]).keys())]
    
    # Common scalebar labels (vocabulary for text recognition refinement)
    common_scale_labels = [x+y for x in ["1","2","5","10","15","20","25","30","50","75","80","100","200","250","300","500"] for y in [" nm"," um"," mm"]]

    # Get subfigure bounding boxes by matching subfigure image patches to the image of the whole figure
    subfigure_bboxes = []
    for subfigure_image_path in subfigure_image_paths:
        subfigure_image = scipy.misc.imread(subfigure_image_path)
        subfigure_bboxes.append([str(a) for a in find_bbox(figure_image,subfigure_image)])

    # Get scalebar bounding boxes by matching scalebar image patches to the image of the whole figure
    # Also, read/resolve the Aster output for each scalebar image 
    scalebar_bboxes = []
    scalebar_text   = []
    for scalebar_image_path in scalebar_image_paths:
        scalebar_image = scipy.misc.imread(scalebar_image_path)
        scalebar_bboxes.append([str(a) for a in find_bbox(figure_image,scalebar_image)])
        resolved_char = scalebar_char_resolve(scale_dict[scalebar_image_path.split("/")[-1].split(".")[0]])
        closest_match = difflib.get_close_matches(resolved_char.lower(),common_scale_labels,cutoff=0.5)
        if closest_match != []:
            scalebar_text.append(closest_match[0])
        else:
            scalebar_text.append("")

    # All subfigure and scalebar bboxes will be set to "unassigned"!
    d = {}
    d["html"]    =  row["article"]
    d["id"]      =  row["fig id"]
    d["image"]   =  figure_image_path.split("/")[-1]
    d["caption"] =  row["caption"]

    d["subfigs"] = {}
    d["subfigs"]["unassigned"] = {}
    d["subfigs"]["unassigned"]["info"] = {}
    d["subfigs"]["unassigned"]["bbox"] = {}
    d["subfigs"]["unassigned"]["bbox"]["label"] = []
    d["subfigs"]["unassigned"]["bbox"]["image"] = subfigure_bboxes 
    d["subfigs"]["unassigned"]["bbox"]["scale"] = scalebar_bboxes
    d["subfigs"]["unassigned"]["info"]["scale"] = scalebar_text

    # The sublabel images will serve as the anchor that subfigure images and then scaling information will be clustered to. 
    for entry in explicit_sublabels:
        d["subfigs"][entry] = {}
        d["subfigs"][entry]["bbox"] = {}
        d["subfigs"][entry]["info"] = {}
        d["subfigs"][entry]["bbox"]["label"] = []
        d["subfigs"][entry]["bbox"]["image"] = []
        d["subfigs"][entry]["bbox"]["scale"] = []
        d["subfigs"][entry]["info"]["caption"]  = []
        d["subfigs"][entry]["info"]["keywords"] = []
        d["subfigs"][entry]["info"]["scale"]    = []

    # Relevant keys to index sublabel dictionary
    sublabel_keys  = related_subset(list(label_dict.keys()),row)

    # Resort keys placing "()" keys first. Assumption: if ASTER recognizes a full open/closed parenthesis around letter, letter is more accurate!
    by_paren_count = [2-(label_dict[a].count("(")+label_dict[a].count(")")) for a in sublabel_keys]
    sublabel_keys  = [x for _,x in sorted(zip(by_paren_count,sublabel_keys))]

    # Recognize sublabel text (in slkeys order), and assigen proper caption components
    for key in sublabel_keys:
        resolved_char  = sublabel_char_resolve(label_dict[key],"alpha")
        closest_match  = difflib.get_close_matches(resolved_char.lower(),explicit_sublabels,cutoff=0.5)
        sublabel_image = scipy.misc.imread(os.getcwd()+"/sub/"+key+".png")
        if closest_match != []:
            d["subfigs"][closest_match[0]]["bbox"]["label"]    = [[str(a) for a in find_bbox(figure_image,sublabel_image)]]
            d["subfigs"][closest_match[0]]["info"]["caption"]  = eval(row["caption text (explicit)"])[closest_match[0]]
            d["subfigs"][closest_match[0]]["info"]["keywords"] = eval(row["caption text (keywords)"])[closest_match[0]]
            explicit_sublabels.remove(closest_match[0])
    # # print("Remaining Choices: ",explicit_sublabels)
    return d

# List of original figure images (One for each row of the .csv)
figure_images   = image_list('original')
# List of dependent images (Multple per original figure)
depend_images   = image_list('image')
# List of cropped scalebar images
scalebar_images = image_list('scale')
# List of cropped subcaption label
sublabel_images = image_list('sub')

scalebar_text   = load_label_yaml('scale')
sublabel_text   = load_label_yaml('sub')

for root, dirs, files in os.walk("csv"):

    for file in files:

        filename = os.fsdecode(os.path.join(root, file))
        df = pd.read_csv(filename) 
        d = []

        for index, row in df.iterrows():
            # Create a dictionary for each row and consolidate related information
            d.append(populate_image_dict(row,figure_images,depend_images,scalebar_images,sublabel_images,scalebar_text,sublabel_text))

with open(os.getcwd()+'/temp.json', 'w') as fout:
    json.dump(d, fout)








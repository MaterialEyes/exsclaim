import os
import glob
import json
import scipy.misc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import namedtuple

def labelbox_to_rect(lb):
    width  = np.abs(lb[1]["y"] - lb[0]["y"])
    height = np.abs(lb[2]["x"] - lb[1]["x"])
    return patches.Rectangle((lb[0]["x"],lb[0]["y"]),height,width,linewidth=2,edgecolor='r',facecolor='none')

def labelbox_to_patch(lb,img):
    x1,y1=lb[0]["x"],lb[0]["y"]
    x2,y2=lb[2]["x"],lb[2]["y"]
    return img[y1:y2,x1:x2]

directory = "extracted"
if not os.path.exists(directory):
    os.makedirs(directory)

df = pd.DataFrame(columns= ['image','caption','keywords','scale'])

with open(os.getcwd()+'/assigned.json', "r") as read_file:
    data = json.load(read_file)

total_count=0
for entry in data:
    print("Figure :",entry["image"])
    # Only iterate through sublabel keys which have an been extracted a sublabel bbox
    for key in [a for a in entry["subfigs"].keys() if entry["subfigs"][a]["bbox"]["label"] != []]:
        if entry["subfigs"][key]["info"]["keywords"] != []:
            image = scipy.misc.imread(os.getcwd()+"/original/"+entry["image"])
            patch = labelbox_to_patch([eval(a) for a in entry["subfigs"][key]["bbox"]["image"]],image)
            plt.imsave("extracted/"+entry["html"].split(".")[0]+"_"+entry["id"]+str(key)+".png",patch)
            if entry["subfigs"][key]["info"]["scale"] == []:
                scale_record = ""
            else:
                scale_record = entry["subfigs"][key]["info"]["scale"]

            df.loc[total_count] = pd.Series({'image':entry["html"].split(".")[0]+"_"+entry["id"]+str(key)+".png", 'caption':entry["subfigs"][key]["info"]["caption"][0], 'keywords':entry["subfigs"][key]["info"]["keywords"][0], 'scale':scale_record})
            total_count+=1

print("Total Images Extracted: ",total_count)
df.to_csv(os.getcwd()+'/extracted/key.csv',index=False)
from PIL import Image, ImageDraw, ImageFont
from skimage import io
import os
import cv2 as cv
import random
import numpy as np
import argparse
from skimage.color import rgba2rgb
from skimage.util import img_as_ubyte
import json


# for command line usage
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--sample_images", type=str, default="no_text",
                help="path to directory of images to use as background")
ap.add_argument("-n", "--number", type=int,
                help="desired number of sample images desired")
ap.add_argument("-o", "--output", type=str, default = "generated_examples",
                help="output directory path (no trailing /)")
ap.add_argument("-f", "--font_collection", type=int, default=0,
                help="")

args = vars(ap.parse_args())

input_directory = args["sample_images"]
output_directory = args["output"] + "/"
samples = args["number"]
font_idx = args["font_collection"]


directory = os.fsencode(input_directory)

lowercase = [chr(i) for i in range(97,123)]
uppercase = [chr(j) for j in range(65,91)]
two_paren = ['(' + k + ')' for k in uppercase+lowercase]
one_paren = [k + ')' for k in uppercase+lowercase]
numbers = [str(i) for i in range(1, 11)]
roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]
all_labels = lowercase+uppercase+two_paren+one_paren+numbers+roman

common_letters = [chr(i) for i in range(97, 107)] + [chr(i) for i in range(65, 75)]
common_paren = ['(' + k + ')' for k in common_letters]
common_labels = common_letters + common_paren

scales = [i for i in range(0, 25)] + [5*i for i in range(6,51)] + [100*i for i in range(3,11)]
scales = [str(i) for i in scales]
units = ["um", "nm", "A"]


font_path = "/Library/Fonts/"

fonts = [['/Library/Fonts/Arial.ttf','/Library/Fonts/Georgia.ttf',\
         '/Library/Fonts/Tahoma.ttf','/Library/Fonts/Verdana.ttf',\
         '/System/Library/Fonts/Helvetica.ttc',\
         '/Applications/Microsoft Word.app/Contents/Resources/DFonts/Calibri.ttf',\
         '/Applications/Microsoft Word.app/Contents/Resources/DFonts/Calibri.ttf',
         '/Applications/Microsoft Word.app/Contents/Resources/DFonts/Cambria.ttc',
         '/Applications/Microsoft Word.app/Contents/Resources/DFonts/tahoma.ttf'],
         ['/Library/Fonts/Arial.ttf','/Library/Fonts/Georgia.ttf',\
         '/Library/Fonts/Tahoma.ttf','/Library/Fonts/Verdana.ttf',\
         '/System/Library/Fonts/Helvetica.ttc',\
         '/Applications/Microsoft Word.app/Contents/Resources/DFonts/Calibri.ttf',\
         '/Applications/Microsoft Word.app/Contents/Resources/DFonts/Calibri.ttf',\
         '/Applications/Microsoft Word.app/Contents/Resources/DFonts/Cambria.ttc',\
         '/Applications/Microsoft Word.app/Contents/Resources/DFonts/Lucida Sans.ttf',\
         '/Applications/Microsoft Word.app/Contents/Resources/DFonts/tahoma.ttf',
         '/Applications/Microsoft Word.app/Contents/Resources/DFonts/Verdana.ttf',
         '/Applications/Microsoft Word.app/Contents/Resources/DFonts/BellMT.ttf',
         '/Applications/Microsoft Word.app/Contents/Resources/DFonts/Candara.ttf',
         '/Applications/Microsoft Word.app/Contents/Resources/DFonts/Perpetua.ttf'],
		 ['verdanab.ttf', 'arial.ttf', 'Carlito-Regular.ttf', 'Candara.ttf', 'georgia.ttf',
		  'tahoma.ttf'],
         ['/usr/share/fonts/dejavu/DejaVuSans' + i for i in ['.ttf', 'Mono.ttf','Mono-Bold.ttf',
          '-Bold.ttf']]
		]

def label_generator():
    """ randomly returns a single string, with added weight for labels """
    selection = random.choices(["common", "labels", "scale"], weights=[20,1,4])[0]
    if selection == "common":
        label = random.choice(common_labels)
    elif selection == "labels":
        label = random.choice(all_labels)
    else:
        label = random.choice(scales) + " " + random.choice(units)
    return label

def image_text_generator(image, image_name, font_idx):
    """ generates an image with text and a txt file with text's coordinates """
    width, height = image.size

    text = label_generator()
    font_type = random.choice(fonts[font_idx])
    font = ImageFont.truetype(font_type, random.randint(10, min(height, 16)))
    text_width, text_height = font.getsize(text)
 
    startX = random.randint(0, max(1,width-text_width))
    startY = random.randint(0, max(1,height-text_height))
    box = (startX, startY,startX + text_width, startY + text_height)

    # draw text on image
    draw = ImageDraw.Draw(image)
    color = find_color(image, box)
    draw.text((startX, startY), text, color, font=font)
    
    move_up, move_left = random.randint(1,5), random.randint(1,5)
    crop_y1 = max(0, startY - move_up )
    crop_y2 = min(height, startY + text_height + random.randint(1,5))
    crop_x1 = max(0, startX - move_left)
    crop_x2 = min(width, startX + text_width + random.randint(1,5))

    # save the image
    image = image.crop((crop_x1,crop_y1, crop_x2, crop_y2))
    save(image, image_name)
    return text

def format_location(box, text):
    """ Formats text location according to EAST training standard """
    startX, startY, endX, endY = box
    coordinate_tuple = (startX, startY, endX, startY,
                        endX, endY, startX, endY, text)
    coordinate_string = str(coordinate_tuple)
    final_string = coordinate_string[1:-1]
    return final_string.replace(' ', '')


def intersect(height, width, boxes, location):
    """ return true if box at location intersects one of boxes """
    startX, startY, endX, endY = location
    if endX > width or endY > height:
        return True
    for box in boxes:
        x1, y1, x2, y2 = box
        if endX >= x1 and x2 >= startX and endY >= y1 and y2 >= startY:
            return True


def find_color(image, box):
    """ finds color for text to contrast background

    param image: an image object
    param box: 4-tuple containing starting and ending x coordinate of text box
        on image and starting and ending y coordinates of text box
    returns: 3-tuple of color of text to create contrast from background """
    startX, startY, endX, endY = box
    image_array = np.array(image)
    grayscale = cv.cvtColor(image_array, cv.COLOR_BGR2GRAY)
    new_image = grayscale[startY:endY, startX:endX]
    mean = np.mean(new_image)
    if mean < 120:
        return (255, 255, 255)
    elif mean < 135:
        return (255, 0, 0)
    else:
        return (0, 0, 0)


def save(image, image_name):
    """ saves image with name image_name to output_directory

    param image: an image object
    param image_name: the name of the image (must include image extension like .jpg, .png
    """
    os.makedirs(output_directory + "data/", exist_ok=True)
    destination = output_directory + "data/" +  image_name
    try:
        #image = rgba2rgb(np.array(image))
        #image = img_as_ubyte(image)
        #io.imsave(destination, image)
        image.save(destination, "JPEG", quality=80, optimize=True, progressive=True)
    except IOError:
        PIL.ImageFile.MAXBLOCK = image.size[0] * image.size[1]
        image.save(destination, "JPEG", quality=80, optimize=True, progressive=True)


desc = {"abc" : uppercase + lowercase + [")", "("] + [str(i) for i in range(0,10)],
        "train" : [] }

# randomly creates samples number of images with 1 to 3 text boxes and
# corresponding text files with locations of text
for i in range(0, samples):
    # open image
    file = random.choice(os.listdir(directory))
    image_name = os.fsdecode(file)
    image_path = input_directory + "/" + image_name
    image = Image.open(image_path)

    # modify image name
    image_list = image_name.split('.')
    image_name = str(i)  + '-' + image_list[0] + '.' + image_list[1]
    text = image_text_generator(image, image_name, font_idx)

    # input image data
    data = {"name" : image_name, "text" : text}
    desc["train"].append(data)

    # output progress
    if i % 100 == 0:
        print("created {} samples".format(i))

# dump json data
with open(output_directory + "desc.json", "w+") as f:
    json.dump(desc, f)

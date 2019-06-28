from PIL import Image, ImageDraw, ImageFont
#from skimage import io
import os
import cv2 as cv
import random
import numpy as np
import argparse

# for command line usage
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--sample_images", type=str, default="no_text",
                help="path to directory of images to use as background")
ap.add_argument("-n", "--number", type=int,
                help="desired number of sample images desired")
ap.add_argument("-o", "--output", type=str, default = "generated_examples",
                help="output directory path (no trailing /)")
args = vars(ap.parse_args())

input_directory = args["sample_images"]
output_directory = args["output"] + "/"
samples = args["number"]

directory = os.fsencode(input_directory)

lowercase = [chr(i) for i in range(97,123)]
uppercase = [chr(j) for j in range(65,91)]
two_paren = ['(' + k + ')' for k in uppercase+lowercase]
one_paren = [k + ')' for k in uppercase+lowercase]
labels = lowercase+uppercase+two_paren+one_paren

units = ["nm", 'um']

numbers = ([str(i) for i in range(1, 20)] +
           ['50', '500', '5000', '100', '1000', '250'])

words = ["Nano-materials", "carbon", "substrate", "tesseract", "atomic", "the",
         "quick", "brown", "fox", "jumped", "over", "red", "fence"]

fonts = ["arial.ttf", 'calibri.ttf', 'tahoma.ttf',
         'verdana.ttf']

def string_generator():
    """ randomly returns a single string, with added weight for labels """
    rand = random.randint(1, 20)
    if rand < 10:
        return random.choice(labels)
    elif rand < 13:
        return random.choice(units)
    elif rand < 17:
        return random.choice(numbers)
    else:
        return random.choice(words)


def image_text_generator(image, image_name):
    """ generates an image with text and a txt file with text's coordinates """
    width, height = image.size
    boxes = []
    for i in range(0, random.randint(1,3)):
        text = string_generator()
        font_type = random.choice(fonts)
        font = ImageFont.truetype(font_type, random.randint(15,20))
        text_width, text_height = font.getsize(text)

        startX = random.randint(0, max(1,width-8))
        startY = random.randint(0, max(1,height-15))
        box = (startX, startY,
               startX + text_width, startY + text_height)
        # if the new text properly fits on the image, add it
        if not intersect(height, width, boxes, box):
            boxes.append(box)

            # draw text on image
            draw = ImageDraw.Draw(image)
            color = find_color(image, box)
            draw.text((startX, startY), text, color, font=font)

            # save the image
            save(image, image_name)

            # write the box location to output file
            location_text = format_location(box, text)
            filename = output_directory + get_text_name(image_name)
            location_file = open(filename, "a+")
            location_file.write(location_text + "\n")
            location_file.close()


def get_text_name(image_name):
    """ creates name of txt file corresponding image_name """
    image_list = image_name.split('.')
    return image_list[0] + '.txt'


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
    os.makedirs(output_directory, exist_ok=True)
    destination = output_directory + image_name
    try:
        image.save(destination, "JPEG", quality=80, optimize=True, progressive=True)
    except IOError:
        PIL.ImageFile.MAXBLOCK = image.size[0] * image.size[1]
        image.save(destination, "JPEG", quality=80, optimize=True, progressive=True)


# randomly creates samples number of images with 1 to 3 text boxes and
# corresponding text files with locations of text
for i in range(0, samples):
    file = random.choice(os.listdir(directory))
    image_name = os.fsdecode(file)
    image_path = input_directory + "/" + image_name
    image = Image.open(image_path)

    # modify image name
    image_list = image_name.split('.')
    image_name = image_list[0] + '-' + str(i) + '.' + image_list[1]

    image_text_generator(image, image_name)

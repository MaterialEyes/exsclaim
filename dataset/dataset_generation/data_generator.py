from PIL import Image, ImageDraw, ImageFont, ImageFile
#from skimage import io
import os
import cv2 as cv
import random
import numpy as np
import argparse
import json

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
ubuntu_fonts = ['/usr/share/fonts/truetype/dejavu/DejaVuSans' + i for i in ['.ttf', 'Mono.ttf','Mono-Bold.ttf',
          '-Bold.ttf']]

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


def draw_text_on_image_east(image, image_name, number_of_text_boxes = 1):
    """ generates an image with text and a txt file with text's coordinates """
    width, height = image.size
    boxes = []
    for i in range(0, ):
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

def draw_text_on_image(image, text):
    """ generates an image with text and a txt file with text's coordinates """
    width, height = image.size
    font_type = random.choice(ubuntu_fonts)
    font = ImageFont.truetype(font_type, random.randint(10, min(height, 16)))
    text_width, text_height = font.getsize(text)
 
    startX = random.randint(0, max(1,width-text_width))
    startY = random.randint(0, max(1,height-text_height))
    box = (startX, startY,startX + text_width, startY + text_height)

    # draw text on image
    draw = ImageDraw.Draw(image)
    color = find_color(image, box)

    draw.text((startX, startY), text, fill=color, font=font)
    
    # add random buffers to bounding box edges
    move_up, move_left = random.randint(1,5), random.randint(1,5)
    crop_y1 = max(0, startY - move_up )
    crop_y2 = min(height, startY + text_height + random.randint(1,5))
    crop_x1 = max(0, startX - move_left)
    crop_x2 = min(width, startX + text_width + random.randint(1,5))

    # save the image
    image = image.crop((crop_x1,crop_y1, crop_x2, crop_y2))

    return image


def get_text_name(image_path):
    """ creates name of txt file corresponding image_name """
    image_name = image_path.split('.')[0]
    return image_name + '.txt'


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

    Args:
        image: an image object
        box: 4-tuple containing starting and ending x coordinate of text box
            on image and starting and ending y coordinates of text box
    Returns
        color (tuple): color of text to create contrast from background
    """
    startX, startY, endX, endY = box
    image_array = np.array(image)
    try:
        grayscale = cv.cvtColor(image_array, cv.COLOR_BGR2GRAY)
    except:
        grayscale = image_array
    new_image = grayscale[startY:endY, startX:endX]
    mean = np.mean(new_image)
    if mean < 120:
        return (254, 254, 254)
    elif mean < 135:
        return (255, 0, 0)
    else:
        return (0, 0, 0)


def save(image, image_path):
    """ saves image to image_path

    Args:
        image (): an image object
        image_path (string): relative path to desired save location
    """
    path_components = image_path.split("/")
    output_directory = "/".join(path_components[:-1])
    os.makedirs(output_directory, exist_ok=True)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    try:
        image.save(image_path, "JPEG", quality=80, optimize=True, progressive=True)
    except IOError:
        ImageFile.MAXBLOCK = image.size[0] * image.size[1]
        image.save(image_path, "JPEG", quality=80, optimize=True, progressive=True)


def generate_dataset_east(samples, input_directory, output_directory):
    """ creates dataset ready to be trained using EAST model

    Args:
        samples (int): Number of images to be generated
        input_directory (string): Directory of background images to be used
        output_directory (string): Directory to output images to
    Modifies:
        Fills output_directory with images containing 1 to 3 text boxes
        and corresponding text files with the locations of the text
    """
    for i in range(0, samples):
        file = random.choice(os.listdir(input_directory))
        image_name = os.fsdecode(file)
        image_path = input_directory + "/" + image_name
        image = Image.open(image_path)

        # modify image name
        image_list = image_name.split('.')
        image_name = image_list[0] + '-' + str(i) + '.' + image_list[1]

        draw_text_on_image_east(image, image_name, random.randint(1, 3))

def generate_dataset_json(samples, input_directory, output_directory):
    desc = {"abc" : uppercase + lowercase + [")", "("] + [str(i) for i in range(0,10)],
            "train" : [] }

    # randomly creates samples number of images with 1 to 3 text boxes and
    # corresponding text files with locations of text
    for i in range(0, samples):
        # open image
        file = random.choice(os.listdir(input_directory))
        image_name = os.fsdecode(file)
        image_path = input_directory + "/" + image_name
        image = Image.open(image_path)

        # modify image name
        image_list = image_name.split('.')
        image_name = str(i)  + '-' + image_list[0] + '.' + image_list[1]
        text = "" ## CHANGE IF USING THIS
        text = draw_text_on_image(image, text)

        # input image data
        data = {"name" : image_name, "text" : text}
        desc["train"].append(data)

        # output progress
        if i % 100 == 0:
            print("created {} samples".format(i))

    # dump json data
    with open(output_directory + "desc.json", "w+") as f:
        json.dump(desc, f)


def generate_dataset_scale_bar_labels(samples, input_directory, output_directory):
    """ creates dataset ready for pytorch classifier training

    Args:
        samples (int): Number of images to be generated
        input_directory (string): Directory of background images to be used
        output_directory (string): Directory to output images to
    Modifies:
        Fills output_directory with subdirectories containing cropped images
            with text matching the directory name
    """
    scales = [round((10**scale) * i, 1) for i in range(1, 10) for scale in range(-1, 3)] + [2.5, 25, 250]
    scales = [str(i) for i in scales]
    units = ["um", "nm", "A"]

    for i in range(samples):
        # select text
        text = random.choice(scales) + " " + random.choice(units)
        # open image
        image_name = random.choice(os.listdir(input_directory))
        image_name = os.fsdecode(image_name)
        image_path = input_directory + image_name
        image = Image.open(image_path).convert("RGB")

        cropped_image = draw_text_on_image(image, text)
        
        # save image
        save(cropped_image, os.path.join(output_directory, text, str(i) + ".png"))

        if i % 500 == 0:
            print("Created {} samples".format(i))



if __name__ == "__main__":
    generate_dataset_scale_bar_labels(50, "no_text/", "scale_label_data")


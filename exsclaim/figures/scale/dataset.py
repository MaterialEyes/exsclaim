import json
import torch
import numpy as np
import os
from PIL import Image, ImageFont, ImageDraw
import random
import pathlib
from matplotlib import font_manager
#from ...utilities.boxes import convert_labelbox_to_coords
from torchvision import transforms

def draw_text_on_image(image, text):
    """ generates an image with text and a txt file with text's coordinates """
    width, height = image.size

    # system_fonts = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    current_dir = pathlib.Path(__file__).resolve(strict=True).parent
    font_directory = current_dir.parent.parent.parent.parent / "dejavu-fonts-ttf-2.37" / 'ttf'
    font_type = random.choice(os.listdir(font_directory))
    font_path = font_directory / font_type
    max_width = int((1.5 * width) / len(text))
    max_font_size = min(height, 24, max_width)
    font_size = random.randint(min(max_font_size-1, 9), max_font_size)
    font = ImageFont.truetype(str(font_path), font_size, encoding="unic")
    text_width, text_height = font.getsize(text)
    startX = random.randint(0, max(1,width-text_width))
    startY = random.randint(0, max(1,height-text_height))
    box = (startX, startY,startX + text_width, startY + text_height)
    # draw text on image
    draw = ImageDraw.Draw(image)
    color = find_color(image, box)

    draw.text((startX, startY), text, fill=color, font=font)
    
    # add random buffers to bounding box edges
    move_up, move_left = random.randint(0,5), random.randint(0,5)
    crop_y1 = max(0, startY - move_up )
    crop_y2 = min(height, startY + text_height + random.randint(1,5))
    crop_x1 = max(0, startX - move_left)
    crop_x2 = min(width, startX + text_width + random.randint(1,5))

    # save the image
    image = image.crop((crop_x1,crop_y1, crop_x2, crop_y2))

    return image

def get_unit():
    units = ["u", "U", "\u03bc", "m", "M", "c", "C", "n", "N", "A", "\u212b"]
    unit1 = random.choice(units)
    unit2 = random.choice(units)
    text = unit1 + unit2
    if random.randint(0, 2) == 0:
        text = unit1
    label = ""
    for character in text:
        if character == "\u212b":
            label_char = "A"
        elif character == "\u03bc":
            label_char = "u"
        else:
            label_char = character
        label += label_char
    return text, label

def get_number(length):
    nonzero = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    if length <= 2:
        text = random.choice(nonzero)
        text += random.choice(digits)
        return text[:length]

    digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    first = random.choice(digits)

    for i in range(length-2):
        if first == "0":
            first += "."
        elif "." in first:
            first += random.choice(digits)
        else:
            first += random.choice(digits + ["."]) 
    first += random.choice(digits)
    return first

def no_pattern(length):
    characters = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", " ", "."] 
    characters += ["u", "U", "\u03bc", "m", "M", "c", "C", "n", "N", "A", "\u212b"]
    text = ""
    for i in range(length-1):
        text += random.choice(characters)

    label = ""
    for character in text:
        if character == "\u212b":
            label_char = "A"
        elif character == "\u03bc":
            label_char = "u"
        else:
            label_char = character
        label += label_char
    return text, label

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
    low_nums = random.randint(0, 15)
    if mean < 120:
        return (254 - low_nums, 254 - low_nums, 254 - low_nums)
    elif mean < 135:
        return random.choice([(254, low_nums, low_nums), (low_nums, 254, low_nums), (low_nums, low_nums, 254)])
    else:
        return (low_nums, low_nums, low_nums)

class ScaleLabelDataset():
    """ Dataset used to train CRNN to read scale bar labels """
    def make_encoding(self, label):
        max_length = 32
        char_to_int = {
            "0":    0,
            "1":    1,
            "2":    2,
            "3":    3,
            "4":    4,
            "5":    5,
            "6":    6,
            "7":    7,
            "8":    8,
            "9":    9,
            "m":    10,
            "M":    11,
            "c":    12,
            "C":    13,
            "u":    14,
            "U":    15,
            "n":    16,
            "N":    17,
            " ":    18,
            ".":    19,
            "A":    20,
            "-":    21
        }
        target = torch.tensor(max_length * [21])
    
        for i, char in enumerate(label):
            target[i] = char_to_int[char]
        return target
         
    def __init__(self, transforms, text="random_separate"):
        self.text = text
        self.transforms = transforms
        current_dir = pathlib.Path(__file__).resolve(strict=True).parent
        self.background_images = current_dir / 'background'
  
    def __getitem__(self, idx):
        # constants
        SYNTH_BACKGOUND = 2 # out of 10
        ## select background image
        # generate a random number create synthetic image
        random_number = random.randint(0, 9)
        if random_number < SYNTH_BACKGOUND:
            # generate an image with solid background
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            background_image = Image.new("RGB", (200, 200), color=color)
        # use a natural background
        else:
            background_image_name = random.choice(
                os.listdir(self.background_images)
            )
            background_image_path = self.background_images / background_image_name
            background_image = (
                Image.open(background_image_path).convert("RGB")
            )

        if self.text == "random_separate":
            ## select text to write on background
            length = random.randint(1, 5)
            number = get_number(length)
            unit, label = get_unit()
            space = random.choice(["", " ", "  "])
            text = number + space + unit
            label = number + " " + label
        elif self.text == "complete_random":
            length = random.randint(3, 8)
            text, label = no_pattern(length)
        # draw text and crop
        cropped_image = draw_text_on_image(background_image, text)        
        if self.transforms is not None:
            cropped_image = self.transforms(cropped_image)

        background_image.close()
        target = self.make_encoding(label)
        return cropped_image, target

    def __len__(self):
        return 5000

class ScaleBarDataset():
    """ Dataset used to train Faster-RCNN to detect scale labels and lines """
    def __init__(self, root, transforms, test=True, size=None):
        ## initiates a dataset from a json
        self.root = root
        self.transforms = transforms
        if test:
            scale_bar_dataset = os.path.join(root, "scale_bars_dataset_test.json")
        else:
            scale_bar_dataset = os.path.join(root, "scale_bars_dataset_train.json")

        self.test = test
        with open(scale_bar_dataset, "r") as f:
            self.data = json.load(f)
        all_figures = os.path.join(root, "images", "labeled_data")
        self.images = [figure for figure in self.data 
                       if os.path.isfile(os.path.join(all_figures,
                                                      figure))]
        if size != None:
            self.images = random.sample(self.images, size)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.root,"images", "labeled_data", self.images[idx])
        with Image.open(image_path).convert("RGB") as image:
            image_name = self.images[idx]

            boxes = []
            labels = []
            for scale_bar in self.data[image_name].setdefault("scale_bars", []):
                boxes.append(convert_labelbox_to_coords(scale_bar["geometry"]))
                labels.append(1)
            for scale_label in self.data[image_name].setdefault("scale_labels", []):
                boxes.append(convert_labelbox_to_coords(scale_label["geometry"]))
                labels.append(2)
            
            num_objs = len(boxes)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd

            if self.transforms is not None:
                new_image = self.transforms(image)

        return new_image, target
        
    def __len__(self):
        return len(self.images)



if __name__ == "__main__":
    normalize_transform = transforms.Compose([transforms.GaussianBlur((3,3), sigma=(0.1, 2.0)),
                                            transforms.Resize((128, 512)),
                                           transforms.ToTensor(),
                                           ])
    resize_transform = transforms.Compose([transforms.Resize((128, 512)),
                                           transforms.ToTensor()]) 
#transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    dataset = ScaleLabelDataset(transforms=normalize_transform)

    for i in range(100):
        image, label = dataset[i]
        image = transforms.ToPILImage()(image)
        image.save("generated/" + str(i) + ".jpg", "JPEG")
from difflib import SequenceMatcher
import os
import ocr_methods as ocr
import pre_processing_methods as pre
import text_detection_methods as td
from PIL import Image
import cv2 as cv
import argparse


<<<<<<< HEAD
=======
# SELECT WHICH FUNCTIONS TO USE FROM RESPECTIVE FILES
td_function = td.east_text_detector
ppm_function = pre.ace
ocr_function = ocr.pytesseract_ocr


>>>>>>> 99f9687... Added README.md
# for command line usage
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--test_images", type=str, default="images/small_nonisolated",
                help="path to directory of test images")
ap.add_argument("-e", "--expected", type=str, default="raw2proc.txt", 
                help="path to text file containing expected output strings")
ap.add_argument("-s", "--save", type=int, default="0",
                help="1 if you want to save the file")
args = vars(ap.parse_args())

# parse command line arguments
directory_name = args["test_images"]

expected_file = open(args["expected"], "r")
expected_lines = expected_file.readlines()

save_images = True if args["save"] == 1 else False


# intitialize which functions will be used
fp_function = pre.none
td_function = td.none
ppm_function = pre.none
ocr_function = ocr.tesserocr_ocr


def save(image, image_name):
    """ saves image with name image_name to images/output_images
    
    param image: an image object
    param image_name: the name of the image (must include image extension like .jpg, .png
    """
    destination = "images/output_images/" + image_name
    try:
        image.save(destination, "JPEG", quality=80, optimize=True, progressive=True)
    except IOError:
        PIL.ImageFile.MAXBLOCK = image.size[0] * image.size[1]
        image.save(destination, "JPEG", quality=80, optimize=True, progressive=True)
        

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()
        
        
def text_from_image(image_name):
    """ uses ocr and ppm functions to convert image to text """
    image_path = directory_name + "/" + image_name
    image_array = cv.imread(image_path)
    first_processed = fp_function(image_array)
    images = td_function(first_processed)
    
    # initialize counter and output text
    i = 0
    text = ""
    
    # extract text from each text box
    for image_array in images:
        processed_image_array = ppm_function(image_array)
        image = Image.fromarray(processed_image_array)
        
        # send processed textbox image to output folder
        if save_images:
            save(image, str(i) + "-" + image_name)
            
        # add text to output string
        text += " " + ocr_function(image)
        i += 1
    return text

total_score = 0
total_adjusted = 0
total_weighted = 0
total_chars = 0
total_adjusted_weighted = 0

# iterate through each file in input directory and print its text
directory = os.fsencode(directory_name)
for file, expected in zip(os.listdir(directory), expected_lines):
    filename = os.fsdecode(file)
    print("File Name: ", filename)
    
    # print the expected text and what the function came up with
    result = text_from_image(filename)
    print("\nResult: ", result)
    print("Expected: ", expected)
    
    # calculate the score
    score = similar(result, expected)
    adjusted_score = similar("".join(expected.split()), "".join(result.split()))
    print("Score: ", score, "\t\tAdjusted Score: ", adjusted_score)
    weight = len(expected)
    total_score += score
    total_adjusted += adjusted_score
    total_weighted += score * weight
    total_adjusted_weighted += adjusted_score * weight
    total_chars += weight
       
    print('-----------------------------------------------------')
    
print("=====================================================\nScore:\t", 
      total_score, "\nAdjusted Score:\t", total_adjusted, 
      "\nAverage Weighted:\t", total_weighted/total_chars,
     "\nAverage Adjusted:\t", total_adjusted_weighted/total_chars)

    
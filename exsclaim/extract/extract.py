import os
import ocr_methods as ocr
import pre_processing_methods as pre
import text_detection_methods as td
from PIL import Image
import cv2 as cv
import argparse


# for command line usage
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--test_images", type=str,
                help="path to directory of test images")
ap.add_argument("-o", "--output_directory", type=str, 
                help="path to directory to save
args = vars(ap.parse_args())

# parse command line arguments
directory_name = args["test_images"]


# intitialize which functions will be used
fp_function = pre.none
td_function = td.east_text_detector
ppm_function = pre.ace
ocr_function = ocr.pytesseract_ocr
              
        
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
                  
        # add text to output string
        text += " " + ocr_function(image)
        i += 1
    return text


# iterate through each file in input directory and print its text
directory = os.fsencode(directory_name)
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print("File Name: ", filename)
    
    # print the expected text and what the function came up with
    result = text_from_image(filename)
    print("\nResult: ", result)     
    print('-----------------------------------------------------')
    
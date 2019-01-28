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
                help="path to directory to save")
ap.add_argument("-p", "--print_results", type=str, default="0",
                help="0 to print to terminal, 1 to save to file")
args = vars(ap.parse_args())

# parse command line arguments
directory_name = args["test_images"]


# SELECT WHICH FUNCTIONS TO USE FROM RESPECTIVE FILES
fp_function = pre.none
td_function = td.east_text_detector_save
ppm_function = pre.ace
ocr_function = ocr.pytesseract_verbose_ocr
              
# Prepare for displaying results
save_text = True if args["print_results"] == "1" else False
    
        
def text_from_image(image_name, directory=""):
    """ uses ocr and ppm functions to convert image to text """
    image_array = cv.imread(image_name)
    first_processed = fp_function(image_array)
    images = td_function(first_processed, image_name, directory)
    
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
for root, dirs, files in os.walk(directory):
    for file in files:
        filename = os.path.join(root, file)
        filename = os.fsdecode(filename)
        directory = os.fsdecode(root)
        
        result = text_from_image(filename, directory)
        
        if save_text:
            os.makedirs("output/" + directory, exist_ok=True)
            output_text = open("output/" + directory + "/output_text.txt", "a+")
            output_text.write("Filename: " + filename + "\n" + result + "\n\n")
            output_text.close()
        else:
            print("File Name: ", filename) 
            # print the expected text and what the function came up with
            print("\nResult: ", result)     
            print('-----------------------------------------------------')
    
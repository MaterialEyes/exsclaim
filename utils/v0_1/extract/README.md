# MaterialEyes Text Extraction

This directory contains modules and functions for the extraction of text from images

### General Use

To run the program from the command line, run:
```
$ python extract.py [--test_images (-t) | --output_directory (-o) | --print_results (-p)]
    test_images: specify the relative path to the images you wish to perform OCR on
                       this directory may contain subdirectories, but all descendant files should be images
    print_results: in both cases results will be printed like "Filename: [filename] \n Result: [result]"    
                    0 - save results to a text files where all the images in each file will be saved in
                        output/path/to/directory/output_text.txt
                    1 - print results to the terminal
```
At the top of extract.py the following code: 
```
# SELECT WHICH FUNCTIONS TO USE FROM RESPECTIVE FILES
fp_function = pre.none
td_function = td.east_text_detector
ppm_function = pre.ace
ocr_function = ocr.pytesseract_ocr
```
This will run an OCR system that will feed EAST raw images, that will then be processed with ACE, before being fed to pytesseract which will convert images to text. To change the OCR system, change the functions to other methods defined in the respective documents. 

### Functions

Image processing functions are contained in pre_processing.py. Each function in this module should take in an image array and output an image array. One can add functions here to test. Text detection methods are contained in text_detection_methods.py. Each function in this module should take in an image array and output a python list of image arrays. Finally text_recognition methods are found in ocr_methods.py. These methods should take in an image array and output a string. 

In each module, there are list defined after all of the functions. These list are composed of two element tuples containing (function: function reference, string: function name) pairs. This is for use in tester.py

#### Notes on Functions

In text_detection_methods, there is an east_text_detection_save method, which besides converting image arrays to a list of image arrays will save original image arrays with text boxes drawn on them to output/path/to/filename. 

In ocr_methods, there is a function pytesseract_builder, which takes in a desired page segmentation method and OCR engine method and outputs a text recognition function using those methods. Explanation of psm and oem available [here](https://github.com/tesseract-ocr/tesseract/wiki/Command-Line-Usage). To modify these for a single function, simply use the optional second and third parameters of the pytesseract_ocr function. 

### Testing

To test and score images against their expected text, use the tester.py file. 
```
$ python tester.py [--test_images (-t) | --expected (-e) | --output (-o) | --image_output (-img) | --iterate (--iter) | --save_results (-s)]
    test_images: specify the relative path to the images you wish to perform OCR on
                       this directory should contain only images
    expected: specify the path to a text file containing the expected output. This should be formatted such that ith line of text contains
            the text expected from the ith image in the directory. For a given image the expected text should simply be all text that 
            appears on the image with separate instances of text separated by spaces in order from top to bottom, then left to right. 
    output: specify the path to the directory where the results text file should be written. The results text file will have the name of the 
            directory of input images and be formatted as "first processing method \\t text detection method \\t pre-processing method
            \\t text recognition method \\t score \\n"
    image_output: 1 - save images fed to text recognition step to specified output directory
                  0 - do not save images
    iterate: 0 - just run tests using methods specified at the top of tester.py
             1 - run tests on all permutations of methods, as defined in the lists of tuples at the bottom of the text_detection_methods,
                 pre_processing_methods, and ocr_methods files
    save results: 0 - print results to terminal
                  1 - save results to output text file
```
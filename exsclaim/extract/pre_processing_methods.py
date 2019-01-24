import cv2 as cv
from PIL import Image
import pillowfight
import numpy as np


def none(image_array):
    """ returns image_array """
    return image_array

def grayscale(image_array):
    """ Converts image to an image in grayscale """
    grayscale = cv.cvtColor(image_array, cv.COLOR_BGR2GRAY)
    return grayscale

def gaussian_threshold(image_array):
    """ Computes gaussian threshold on image_array """
    grayscale = cv.cvtColor(image_array, cv.COLOR_BGR2GRAY)
    gaussian = cv.adaptiveThreshold(grayscale,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,21,0)
    return gaussian

def stroke_width_transformation(image_array):
    """ performs stroke-width-transformation on image_array """
    image = Image.fromarray(image_array)
    swt_image = pillowfight.swt(image, output_type=pillowfight.SWT_OUTPUT_ORIGINAL_BOXES)
    swt_array = np.array(swt_image)
    return swt_array

def resize(image_array):
    resized = cv.resize(image_array, None, fx=3, fy=3, interpolation = cv.INTER_CUBIC)
    return resized

def bilateral_filter(image_array):
    removal = cv.bilateralFilter(image_array, 9, 75, 75)
    return removal

def ace(image_array):
    """ performs Automatic Color Equalization """
    image = Image.fromarray(image_array)
    ace_image = pillowfight.ace(image, slope=10, limit=1000, samples=100, seed=6)
    ace_array = np.array(ace_image)
    return ace_array

def canny_edge_detection(image_array):
    """ performs Canny's Edge Detection """
    image = Image.fromarray(image_array)
    canny_image = pillowfight.canny(image)
    canny_array = np.array(canny_image)
    return canny_array

def extreme_threshold(image_array):
    resized = cv.resize(image_array, None, fx=4, fy=4, interpolation = cv.INTER_CUBIC)
    grayscale = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(grayscale, 30, 255, cv.THRESH_BINARY)
    removal = cv.bilateralFilter(thresh, 9, 75, 75)
    return removal

def upperleft_focus(image_array):
    dims = image_array.shape
    height, width = dims[0:2]
    resized = image_array[0:int(.1*width), 0:int(.25*height)]
    return resized

def resize_filter_focus(image_array):
    resized = resize(image_array)
    filtered = bilateral_filter(resized)
    #filt1 = bilateral_filter(filtered)
    #filt2 = bilateral_filter(filt1)
    return filtered

def resize_swt(image_array):
    resized = resize(image_array)
    return stroke_width_transformation(resized)

def sixela(image_array):
    gray = resize_filter_focus(image_array)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    imgTH = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel)
    _, imgBin = cv.threshold(imgTH, 0, 250, cv.THRESH_OTSU)
    imgdil = cv.dilate(imgBin, kernel)
    _, imgBin_Inv = cv.threshold(imgdil, 0, 250, cv.THRESH_BINARY_INV)
    return imgBin_Inv



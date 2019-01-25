## adapted from https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/

from imutils.object_detection import non_max_suppression
from PIL import Image
import numpy as np
import cv2
import os

def none(image_array):
    return [image_array]


def east_text_detector_save(image_array, image_name, image_path):
    """ Runs EAST saving images with boxes drawn on them """
    return east_text_detector(image_array, image_name, image_path, save=True)


def east_text_detector(image_array, im_name = "", image_path = "", east_detector="frozen_east_text_detection.pb", min_confidence=0.5, 
                      newW = 320, newH = 320, save=False):
    """ Returns images cropped around bounding boxes of text in image represented by image_path 
    
    param image_path: path to input image
    param east_detector: path to input EAST text detector
    param min_confidence: minimum probability required to inspect a region
    param newW: resize to selected width (must be multiple of 32)
    param newH: resize to selected height (must be multiple of 32)
    """
    image = image_array
    orig = image.copy()
    (orig_height, orig_width) = image.shape[:2]
    
    # set the new width and height and then determine the ratio in change
    # for both the width and height
    rW = orig_width / float(newW)
    rH = orig_height / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (height, width) = image.shape[:2]
    
    
    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]
    
    # load the pre-trained EAST text detector
    net = cv2.dnn.readNet(east_detector)
    
    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (width, height),
        (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    
    ## check if we found any text boxes at all
    if numRows == 0:
        return orig
    
    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < min_confidence:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    
    boxes = sorted(boxes, key = lambda x: x[0])
    boxes = sorted(boxes, key = lambda y: y[1])
            
    # return a new image for each text box
    text_boxes = []

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        
        new_image = orig.copy()
        
        # ensure that the textbox doesn't go beyond the image
        startY = max(0, startY-1)
        startX = max(0, startX-1)
        endY = min(orig_height, endY-1)
        endX = min(orig_width, endX-1)
        
        text_box = new_image[startY:endY, startX:endX]
        text_boxes.append(text_box)
        
        # to save a copy of the image with text boxes drawn
        if save:
            cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
    
    if save:
        original_image = Image.fromarray(orig)
        display(original_image, im_name, image_path)
        
    if len(text_boxes) == 0:
        return [orig]
    else:
        return text_boxes
        
        
# show the output image
def display(image, image_name, image_path):
    """ saves images to output_east directory """
    os.makedirs("output/" + image_path, exist_ok=True)
    destination = "output/" + image_name
    try:
        image.save(destination, "JPEG", quality=80, optimize=True, progressive=True)
    except IOError:
        PIL.ImageFile.MAXBLOCK = img.size[0] * img.size[1]
        image.save(destination, "JPEG", quality=80, optimize=True, progressive=True)


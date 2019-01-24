from PIL import Image
import pytesseract
import tesserocr
import pyocr
import pyocr.builders

#pytesseract.pytesseract.tesseract_cmd = r'/usr/share/tesseract-ocr'

def pytesseract_ocr(image):
    """ Uses pytesseract to find the text in an image

    param image: image object
    returns: a string of all text in the image
    """
    return pytesseract.image_to_string(image, config = "--psm 10")

def tesserocr_ocr(image):
    """ Uses tesserocr to find the text in an image

    param image: image object
    returns: a string of all the text in the image
    """
    return tesserocr.image_to_text(image)

def pyocr_ocr(image):
    """ Uses pyocr to find the text in an image

    param image: image object
    returns: a string of all the text in the image
    """
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool found")
        sys.exit(1)
    tool = tools[0]
    return tool.image_to_string(image,
                                lang="eng",
                                builder=pyocr.builders.TextBuilder())

def pyocr_ocr_boxes(image):
    """ Uses pyocr to find the text in an image

    param image: image object
    returns: a string of all the text in the image
    """
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool found")
        sys.exit(1)
    tool = tools[0]
    lines =  tool.image_to_string(image,
                                lang="eng",
                                builder=pyocr.builders.LineBoxBuilder())
    output = ''
    for line in lines:
        output += line.content
        output += "\n ------ with position: "
        output += str(line.position)        
        output += "\n"
    return output


    

functions = [pytesseract_ocr, tesserocr_ocr, pyocr_ocr]

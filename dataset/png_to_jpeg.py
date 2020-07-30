import glob
from PIL import Image

def png_to_jpeg():
    for img in glob.glob("*.png"):
        im = Image.open(img).save(img.split('.png')[0]+".jpg", "JPEG")

png_to_jpeg()

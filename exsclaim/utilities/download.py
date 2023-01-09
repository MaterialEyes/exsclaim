### Function for downloading from Google Drive ###
import os

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?id="+id
    cmd = "gdown %s -O %s"%(URL, destination)
    os.system(cmd)
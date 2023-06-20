import os

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?id="+id
    cmd = "gdown %s -O %s"%(URL, destination)
    os.system(cmd)  
    

model_path = "./exsclaim/figures/checkpoints/"
os.umask(0)
os.makedirs(model_path, mode=0o777, exist_ok=True)

download_file_from_google_drive('1ZodeH37Nd4ZbA0_1G_MkLKuuiyk7VUXR', './exsclaim/figures/checkpoints/classifier_model.pt')
download_file_from_google_drive('1Hh7IPTEc-oTWDGAxI9o0lKrv9MBgP4rm', './exsclaim/figures/checkpoints/object_detection_model.pt')
download_file_from_google_drive('1rZaxCPEWKGwvwYYa8jLINpUt20h0jo8y', './exsclaim/figures/checkpoints/text_recognition_model.pt')
download_file_from_google_drive('1B4_rMbP3a1XguHHX4EnJ6tSlyCCRIiy4', './exsclaim/figures/checkpoints/scale_bar_detection_model.pt')
download_file_from_google_drive('1oGjPG698LdSGvv3FhrLYh_1FhcmYYKpu', './exsclaim/figures/checkpoints/scale_label_recognition_model.pt')
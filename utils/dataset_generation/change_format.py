import json
import os
import shutil

with open("generated_examples/desc.json", "r") as f:
    data = json.load(f)
    image_path = "generated_examples/data"
    for point in data["train"]:
        new_dir = "train/" + point["text"]
        os.makedirs(new_dir, exist_ok=True)
        shutil.copy2(image_path + "/" + point["name"], new_dir)
    
    for point in data["test"]:
        new_dir = "test/" + point["text"]
        os.makedirs(new_dir, exist_ok=True)
        shutil.copy2(image_path + "/" + point["name"], new_dir)


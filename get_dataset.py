import os
import shutil
import zipfile

import requests

from config import config

"""
Fetch the dataset from the COCO repository and extract it to data/images.
Currently fetches the 2017 Validation dataset, containing 5000 images and totalling 788MB.
"""

print("Creating directories...")
os.makedirs(config.get_path(config.config["datasets"]["images"]), exist_ok=True)

COCO_DATASET_URL = "http://images.cocodataset.org/zips/val2017.zip"

print("Downloading dataset...")

r = requests.get(COCO_DATASET_URL, stream=True)
with open("data.zip", "wb") as fd:
    for chunk in r.iter_content(chunk_size=128):
        fd.write(chunk)

print("Extracting dataset...")

image_dir = config.get_path(config.config["datasets"]["images"])

z = zipfile.ZipFile("data.zip", "r")
z.extractall(image_dir)

for root, dirs, files in os.walk(config.get_path(os.path.join(image_dir, "val2017"))):
    for file in files:
        shutil.move(
            os.path.join(root, file), config.get_path(os.path.join(root, "..", file))
        )
os.rmdir(os.path.join(image_dir, "val2017"))

print("Deleting zip file...")

os.remove("data.zip")

print("Done!")

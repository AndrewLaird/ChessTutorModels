# assume this is run after detect.py has been run, this means that the images in data/images
# have corresponding data in labels
from PIL import Image
import numpy as np
import pandas as pd
import os
import random
import sklearn
import skimage
import skimage.io
import matplotlib.pyplot as plt
import pathlib

PROJECT_DIR = str(pathlib.Path(__file__).parent.absolute()) + "/"
IMAGES_DIR = PROJECT_DIR + "data/images"
LABELS_DIR = PROJECT_DIR + "data/cropped/labels"
CROPPED_DIR = PROJECT_DIR + "data/cropped/"

files = []
for (dirpath, dirnames, filenames) in os.walk(IMAGES_DIR):
    files.extend(filenames)
    break


def map_files_to_labels(image):
    text_path = image + ".txt"
    return text_path


labels = [map_files_to_labels(file) for file in files]


def crop_image(file, label, output_dir):
    width, height = image.size
    # find the largest of the squares here
    # but for now just get the first one
    class_id, center_x, center_y, box_width, box_height = [
        float(x) for x in list(label.iloc[0])[0].split(" ")
    ]
    pixel_center_x = width * center_x
    pixel_center_y = height * center_y
    pixel_box_width = width * box_width
    pixel_box_height = height * box_height
    box_top_left_x = pixel_center_x - pixel_box_width / 2
    box_top_left_y = pixel_center_y - pixel_box_height / 2

    left = box_top_left_x
    top = box_top_left_y
    right = box_top_left_x + pixel_box_width
    bottom = box_top_left_y + pixel_box_height
    cropped_image = image.crop((left, top, right, bottom))

    # image.show()
    cropped_image.save(output_dir + files[i])


for i in range(len(files)):
    image = Image.open(IMAGES_DIR + "/" + files[i])
    if not os.path.exists(LABELS_DIR + "/" + labels[i]):
        continue
    label = pd.read_csv(LABELS_DIR + "/" + labels[i], header=None)
    crop_image(image, label, CROPPED_DIR)

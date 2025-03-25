"""Data preprocessing for WHM CAPTCHAs"""
import numpy as np
import os
import shutil
from PIL import Image
from imutils import paths

import object.config as config

def save_images_to_folder(foldername, images, labels):
    # (Re-)create folder if existing
    if os.path.exists(foldername):
        shutil.rmtree(foldername)
    os.mkdir(foldername)

    # Count for each key
    counts = {}

    for image, label in zip(images, labels):
        # Create target folder if not existing
        target_path = os.path.join(foldername, label)
        if not os.path.exists(target_path):
            os.makedirs(target_path)

        # Save image with unique name
        count = counts.get(label, 1)
        p = os.path.join(target_path, f"{str(count).zfill(6)}.png")
        image.save(p, "PNG")

        # Increment count for current key
        counts[label] = count + 1

def create():

    image_list = []
    label_list = []

    # Iterate over training images
    for file in paths.list_images(config.RAW_DIR):
        im = Image.open(file)
        im.load()
        rgba_image = im.convert('RGBA')
        image_list.append(rgba_image)
        label_list.append(file.split(os.path.sep)[-2])

    # Permute the list, so that we don't introduce a bias by having 100 dogs in a row
    sequence = np.random.permutation(len(image_list))
    permuted_image_list = [image_list[index] for index in sequence]
    permuted_label_list = [label_list[index] for index in sequence]

    # Split data into training and validation data
    X_train = permuted_image_list[:int(config.SPLIT_RATE * len(permuted_image_list))]
    X_test = permuted_image_list[int(config.SPLIT_RATE * len(permuted_image_list)):]
    Y_train = permuted_label_list[:int(config.SPLIT_RATE * len(permuted_image_list))]
    Y_test = permuted_label_list[int(config.SPLIT_RATE * len(permuted_image_list)):]

    # Save validation data to a separate folder
    save_images_to_folder(config.VAL_DIR, images=X_test, labels=Y_test)


    # Rotate & save training data to separate folder
    X_rotated = []
    Y_rotated = []
    degree_step = 30  # increase in angle between every iteration
    for image, label in zip(X_train, Y_train):
        degree = 0
        while degree < 360:
            X_rotated.append(image.rotate(degree))
            Y_rotated.append(label)
            degree += degree_step

    save_images_to_folder(config.TRAIN_DIR, X_rotated, Y_rotated)


if __name__ == '__main__':
    create()

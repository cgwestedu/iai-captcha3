"""Preparing training & validation data for TRZ network training"""
import os
import numpy as np
import cv2
from imutils import paths
from sklearn.model_selection import train_test_split
import imutils
from sklearn.preprocessing import LabelBinarizer
import pickle

import text.config as config

def training_val_data():
    data = []
    labels = []

    # Iterate over training images
    for image_file in paths.list_images(config.TRAIN_DIR):
        # Load image and convert to grayscale
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize letter to 35x35 (wxh) pixels
        image = resize_to_fit(image, 35, 35)

        # Add third channel dimension for Keras
        image = np.expand_dims(image, axis=2)

        # Label letter based on folder
        label = image_file.split(os.path.sep)[-2]

        # Add letter image and label to training data
        data.append(image)
        labels.append(label)


    # Divide by 255 to normalize to range 0 to 1 (pixel values range from 0 to 256)
    data = np.array(data, dtype=np.float32) / 255.0
    labels = np.array(labels)

    # Split into training and validation data
    (X_train, X_val, Y_train, Y_val) = train_test_split(data, labels, train_size=config.SPLIT_RATE, random_state=0)

    # Convert the labels (letters) into one-hot encodings that Keras can work with
    lb = LabelBinarizer().fit(Y_train)
    Y_train = lb.transform(Y_train)
    Y_val = lb.transform(Y_val)

    # Save the mapping from labels to one-hot encodings.
    with open(config.MODEL_LABELS_FILENAME, "wb") as f:
        pickle.dump(lb, f)

    return X_train, X_val, Y_train, Y_val

def resize_to_fit(image, width, height):
    """
    A helper function to resize an image to fit within a given size
    :param image: image to resize
    :param width: desired width in pixels
    :param height: desired height in pixels
    :return: the resized image
    """

    # grab the dimensions of the image, then initialize
    # the padding values
    (h, w) = image.shape[:2]

    # if the width is greater than the height then resize along
    # the width
    if w > h:
        image = imutils.resize(image, width=width)

    # otherwise, the height is greater than the width so resize
    # along the height
    else:
        image = imutils.resize(image, height=height)

    # determine the padding values for the width and height to
    # obtain the target dimensions
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    # pad the image then apply one more resizing to handle any
    # rounding issues
    image = cv2.copyMakeBorder(image, top=padH, bottom=padH, left=padW, right=padW, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
    image = cv2.resize(image, (width, height))

    # return the pre-processed image
    return image
"""Preparing training & validation data for WHM network training"""
import os
import numpy as np
from PIL import Image
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
import pickle

import object.config as config


def training_val_data():

    # Loading the training dataset
    X_train = []
    Y_train = []
    # Iterate over training images
    for image_file in paths.list_images(config.TRAIN_DIR):
        # Load image
        image = Image.open(image_file)
        image.load()

        # Label based on folder
        label = image_file.split(os.path.sep)[-2]

        # Add image and label to training data
        X_train.append(image)
        Y_train.append(label)

    # Permute the list, so that we don't introduce a bias by having the rotated version of a pic in a row
    sequence = np.random.permutation(len(X_train))
    X_train = [X_train[index] for index in sequence]
    X_train = [np.array(img, dtype=np.float32)/255 for img in X_train]   # Divide by 255 to normalize to range 0 to 1 (pixel values range from 0 to 256)
    X_train = np.asarray(X_train)
    Y_train = [Y_train[index] for index in sequence]
    Y_train = np.asarray(Y_train)

    # Loading the validation dataset
    X_test = []
    Y_test = []
    # Iterate over validation images
    for image_file in paths.list_images(config.VAL_DIR):
        # Load image
        image = Image.open(image_file)
        image.load()

        # Label based on folder
        label = image_file.split(os.path.sep)[-2]

        # Add image and label to training data
        X_test.append(image)
        Y_test.append(label)

    X_test = [np.array(img, dtype=np.float32)/255 for img in X_test]
    X_test = np.asarray(X_test)
    Y_test = np.asarray(Y_test)

    # Convert the labels (letters) into one-hot encodings that Keras can work with
    lb = LabelBinarizer().fit(Y_train)
    Y_train = lb.transform(Y_train)
    Y_test = lb.transform(Y_test)

    # Save the mapping from labels to one-hot encodings.
    with open(config.MODEL_LABELS_FILENAME, "wb") as f:
        pickle.dump(lb, f)

    return X_train, X_test, Y_train, Y_test




"""Train network and store trained model for chosen subfolder"""
import os
import numpy as np
import cv2
import pickle
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import datetime

import network
import text.config as config
import prepare_training

from keras import callbacks


def train_model():
    USE_GPU = False     # Change to True when using GPU-KONG
    if USE_GPU:
        GPU_COUNT = 2
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_COUNT)

    # Get training and validation data from specified subfolder
    X_train, X_val, Y_train, Y_val = prepare_training.training_val_data()

    # Convert the labels (letters) into one-hot encodings that Keras can work with
    lb = LabelBinarizer().fit(Y_train)
    Y_train = lb.transform(Y_train)
    Y_val = lb.transform(Y_val)

    with open(config.MODEL_LABELS_FILENAME, "rb") as f:
        lb = pickle.load(f)

    model = network.get_model(len(lb.classes_))

    earlystopping = callbacks.EarlyStopping(monitor="val_accuracy", mode="max", verbose=1, patience=5)
    modelcheckpoint = callbacks.ModelCheckpoint(os.path.join(config.MODELS_DIR, f"model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"), monitor="val_accuracy", mode="max", verbose=1, save_best_only=True)
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
              callbacks=[earlystopping, modelcheckpoint])


print("Finished fitting")


if __name__ == '__main__':
    train_model()


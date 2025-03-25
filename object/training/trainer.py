"""Train network and store trained model for chosen subfolder"""
import os
import pickle
from sklearn.preprocessing import LabelBinarizer
from keras import callbacks
import datetime

import network
import object.config as config
import prepare_training


def train_model():
    USE_GPU = False     # Change to True when using GPU-KONG
    if USE_GPU:
        GPU_COUNT = 2
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_COUNT)

    X_train, X_val, Y_train, Y_val = prepare_training.training_val_data()

    with open(config.MODEL_LABELS_FILENAME, "rb") as f:
        lb = pickle.load(f)

    model = network.get_model(len(lb.classes_))

    earlystopping = callbacks.EarlyStopping(monitor="val_accuracy", mode="max", verbose=1, patience=5)
    modelcheckpoint = callbacks.ModelCheckpoint(os.path.join(config.MODELS_DIR, f"model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"), monitor="val_accuracy", mode="max", verbose=1, save_best_only=True)
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, callbacks=[earlystopping, modelcheckpoint])

    print("Finished fitting")


if __name__ == '__main__':
    train_model()


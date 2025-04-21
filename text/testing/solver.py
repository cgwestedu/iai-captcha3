"""CAPTCHA solver for TRZ that can be used by passing the path to the image as argument"""
import os
import tensorflow as tf
import pickle
import numpy as np
import sys
import json
import keras

import text.preprocessing.image_splitter as image_splitter
import text.training.prepare_training as prepare_training
import text.config as config

MODEL_DATE = "20250421_002546"
MODEL_PATH = os.path.join(config.MODELS_DIR, f"model_{MODEL_DATE}.keras")

class Solver:
    def __init__(self, network_path=MODEL_PATH, encoding_path=config.MODEL_LABELS_FILENAME):
        self.network_path = network_path
        self.loaded_model = keras.models.load_model(self.network_path)  # load trained network

        with open(encoding_path, "rb") as f:
            self.encoder = pickle.load(f)


    def solve(self, img_file):

        letter_images = image_splitter.split(img_file)

        # If image couldn't be split properly, skip image instead of bad training data
        if letter_images is None:
            return None

        result = []

        for letter_image in letter_images:

            # Resize letter to 30x35 (wxh) pixels to match training data
            letter_image = prepare_training.resize_to_fit(letter_image, 35, 35)

            # Add third channel dimension for Keras
            letter_image = np.expand_dims(letter_image, axis=2)
            letter_image = np.expand_dims(letter_image, axis=0)

            # Predict with given model
            prediction = self.loaded_model.predict(letter_image)

            # Convert one-hot-encoded prediction back to letter
            result.append(self.encoder.inverse_transform(prediction)[0])


        return result


def main():
    if len(sys.argv) != 2:
        raise ValueError("Please provide the path to the test image-files as an argument!")

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress tensorflow info messages
    solver = Solver()
    result = solver.solve(img_file=sys.argv[1])
    formatted_result = json.dumps(result)
    print(formatted_result)


if __name__ == '__main__':
    main()


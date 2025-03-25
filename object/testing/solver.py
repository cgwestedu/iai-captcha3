"""CAPTCHA solver for WHM that can be used by passing the path to the HTML as argument"""
import tensorflow as tf
import glob
import os
from PIL import Image
import numpy as np
import shutil
import sys
import json
import pickle
import keras

import object.testing.image_extractor as image_extractor
import object.config as config

MODEL_DATE = "20250324_180838"
MODEL_PATH = os.path.join(config.MODELS_DIR, f"model_{MODEL_DATE}.keras")

class Solver:
    def __init__(self, network_path=MODEL_PATH, encoding_path=config.MODEL_LABELS_FILENAME):
        self.network_path = network_path
        self.loaded_model = keras.models.load_model(self.network_path)  # load trained network

        with open(encoding_path, "rb") as f:
            self.encoder = pickle.load(f)


    def solve(self, filename):
        img_foldername = filename.replace(".html", "") + "_images"
        success = image_extractor.extract(html_filename=filename, images_foldername=img_foldername)
        result = []

        if success:
            image_list = []
            captcha_ids = []
            for image in glob.glob(f"{img_foldername}{os.sep}*{os.sep}*.png"):
                label = image.split(os.sep)[img_foldername.count(os.sep) + 1]  # save the label of the searched class
                captcha_ids.append(
                    image.split(os.sep)[img_foldername.count(os.sep) + 2].replace(".png", "")
                )  # save ids used to identify images
                img = Image.open(image)
                img.load()
                image_list.append(np.asarray(img.convert("RGBA"), dtype=np.float32) / 255)

            # use neural network to predict pictures of searched class
            predictions = self.loaded_model.predict(np.asarray(image_list))
            predictions = self.encoder.inverse_transform(predictions)
            result = [int(captcha_ids[i]) for i in range(len(predictions)) if predictions[i] == label]
            result.sort()

            # Clean up extracted images
            shutil.rmtree(img_foldername)

        return result


def main():
    if len(sys.argv) != 2:
        raise ValueError("Please provide the pathname to the HTML-file as an argument!")

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress tensorflow info messages
    solver = Solver()
    result = solver.solve(filename=sys.argv[1])
    formatted_result = json.dumps(result)
    print(formatted_result)


if __name__ == "__main__":
    main()


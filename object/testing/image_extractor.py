"""Extracting images of one WHM CAPTCHA"""
import os
import shutil
from lxml import html
import base64


def get_class_name(label):
    """Extracts the Class-name from the first label in the CAPTCHA´s HTML"""
    return label.xpath("./text()")[0].split("containing ")[1].split("(")[0]

def extract(html_filename, images_foldername):
    # (Re-)create folder if already existing
    if os.path.exists(images_foldername):
        shutil.rmtree(images_foldername)
    os.mkdir(images_foldername)

    with open(html_filename, "r") as html_page:
        source_code = html_page.read()

    root = html.fromstring(source_code)
    labels = root.xpath("//label")

    if len(labels) > 0:
        class_name = get_class_name(labels[0])

        try:
            os.mkdir(f"{images_foldername}{os.sep}{class_name}")
        except OSError:
            print('Creation of the directory failed')
            return False

        for x in range(1, len(labels)):
            image = labels[x].xpath("./@style")[0].split(",")[1]
            filename = str(x) + ".png"
            with open(f"{images_foldername}{os.sep}{class_name}{os.sep}{filename}", "wb") as img:
                img.write(base64.b64decode(image))
        return True
    else:
        return False





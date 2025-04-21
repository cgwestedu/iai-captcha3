"""Splitting TRZ CAPTCHA into single characters"""
import cv2
import numpy as np


def split(filepath):

    # Load image and convert to grayscale
    image = cv2.imread(filepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert to binary with threshold (white letters, black background)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Add black padding around image to prevent letter joining with frame of image
    binary = cv2.copyMakeBorder(binary, 8, 8, 8, 8, cv2.BORDER_CONSTANT)

    # Square:1-Kernel
    kernel = np.array([[1, 1],
                       [1, 1]], dtype=np.uint8)

    # Erosion: Erode pixels only if all pixels under kernel is 1
    erosion = cv2.erode(binary, kernel, iterations=1)

    # New Kernel to fill possible gaps in letters, since lines are gone
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]], dtype=np.uint8)

    # Dilation: Dilate pixels if at least one pixel under kernel is 1
    morphed = cv2.dilate(erosion, kernel, iterations=1)

    # Find only external contours and extract biggest six (first one is frame)
    contours, hierarchy = cv2.findContours(image=morphed, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    max_six_contours = sorted(list(contours), key=len, reverse=True)[:6]

    # Invert for further processing (black on white)
    morphed = cv2.bitwise_not(morphed)

    # Loop through each of the six contours and extract region of letter
    letter_regions = []
    i = 0
    for contour in max_six_contours:
        # We only have six letters
        if len(letter_regions) >= 6:
            break

        # Get the rectangle that contains the contour
        (x, y, w, h) = cv2.boundingRect(contour)

        # If biggest contour: compare width and height of the contour to detect letters conjoined into one
        if i == 0 and w / h > 1.1:  # Conjoined letters: Split into two letter regions
            half_width = int(w / 2)
            letter_regions.append((x, y, half_width, h))
            letter_regions.append((x + half_width, y, half_width, h))
            i = 0
        else:
            # Single letter region
            letter_regions.append((x, y, w, h))
            i = 1

    # If more/less than 6 letters found, skip image instead of bad training data
    if len(letter_regions) != 6:
        print("Not six letters")
        return None

    # Sort letters from left-to-right based on x coordinate
    letter_regions = sorted(letter_regions, key=lambda x: x[0])

    result_letters = []
    # Save each letter as single image
    for box in letter_regions:
        # Coordinates of letter
        x, y, w, h = box

        # Extract letter from morphed with 2-pixel margin around edge
        letter_image = morphed[y-2: y+h+2, x-2: x+w+2]

        result_letters.append(letter_image)


    return result_letters


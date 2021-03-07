import multiprocessing
from operator import contains
from typing import List, Dict, Tuple
import cv2  # opencv-python
import skimage  # scikit-image
import os
import numpy as np
from skimage import io, util


def process(source: str, dimension: int) -> None:
    pass


def load_images(path: str) -> Dict[str, np.array]:
    images = {}
    for guid in os.listdir(path):
        guidPath = os.path.join(path, guid)
        for className in os.listdir(os.path.join(guidPath)):
            if not contains(images, className):
                images[className] = []

            classPath = os.path.join(guidPath, className)
            for file in os.listdir(classPath):
                filepath = os.path.join(classPath, file)
                image = cv2.imread(filepath)
                images[className].append(image)

    return images


def save_images(path: str, images: Dict[str, np.array]) -> None:
    for className, imagesList in images.items():
        folder = os.path.join(path, className)
        os.makedirs(folder, exist_ok=True)

        for index, image in enumerate(imagesList):
            file = os.path.join(folder, '{:04}.png'.format(index))
            cv2.imwrite(file, image)


def resize(image: np.array, dim: Tuple[int, int]) -> np.array:
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def to_monochrome(image: np.array) -> np.array:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    if np.average(image) > 127:
        image = 255 - image
    return image


def crop_fields(image: np.array) -> np.array:
    pass


images = load_images('./data/images')

for imagesList in images.values():
    for index, image in enumerate(imagesList):
        image = resize(image, (32, 32))
        image = to_monochrome(image)
        imagesList[index] = image

cv2.imshow('Sample image', images['0'][0])
cv2.waitKey(0)
cv2.destroyAllWindows()

save_images('./data/images-gen', images)

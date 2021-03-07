import multiprocessing
from operator import contains
from typing import List, Dict
import cv2  # opencv-python
import skimage  # scikit-image
import os
import numpy as np
from skimage import io


def process(source: str, dimension: int) -> None:
    pass


def load_images(path: str) -> List[Dict[str, np.array]]:
    images = {}
    for guid in os.listdir(path):
        guidPath = os.path.join(path, guid)
        for className in os.listdir(os.path.join(guidPath)):
            if not contains(images, className):
                images[className] = []

            classPath = os.path.join(guidPath, className)
            for file in os.listdir(classPath):
                filepath = os.path.join(classPath, file)
                image = io.imread(filepath)
                images[className].append(image)

    return images


def save_images(path: str, images: Dict[str, np.array]) -> None:
    for className, imagesList in images.items():
        folder = os.path.join(path, className)
        os.makedirs(folder, exist_ok=True)

        for index, image in enumerate(imagesList):
            file = os.path.join(folder, '{:4}.png'.format(index))
            io.imsave(file, image, check_contrast=False)


def resize():
    pass


def to_black_white(image):
    pass


def crop_fields():
    pass


images = load_images('./data/images')
save_images('./data/images-gen', images)

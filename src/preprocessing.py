from operator import contains
from typing import List, Dict, Tuple
import cv2
import os
import shutil
import numpy as np


def process(source: str, destination: str, dimensions: Tuple[int, int]) -> None:
    images = load_images(source)
    for imagesList in images.values():
        for index, image in enumerate(imagesList):
            image = to_bgr(image)
            image = crop(image)
            image = resize(image, dimensions)
            image = to_monochrome(image)
            imagesList[index] = image

    save_images(destination, images)


def load_images(path: str) -> Dict[str, np.array]:
    images = {}
    for guid in os.listdir(path):
        guidPath = os.path.join(path, guid)
        for className in os.listdir(os.path.join(guidPath)):
            if not contains(images, className):
                images[className] = []

            classPath = os.path.join(guidPath, className)
            for file in os.listdir(classPath):
                originalFile = os.path.join(classPath, file)
                tempFile = os.path.join(classPath, '.tmp')

                shutil.copy2(originalFile, tempFile)
                image = cv2.imread(tempFile, cv2.IMREAD_UNCHANGED)
                os.remove(tempFile)

                if image is None:
                    raise Exception(f'Couldn\'t read image at {originalFile}')

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


def to_bgr(image: np.array) -> np.array:
    image = np.copy(image)
    if image.shape[2] == 4 and np.max(image[:, :, 3]) - np.min(image[:, :, 3]) >= 128:
        alpha = image[:, :, 3]
        image[:, :, 0] = alpha
        image[:, :, 1] = alpha
        image[:, :, 2] = alpha

    return image


def to_monochrome(image: np.array) -> np.array:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)

    edges_avg = (np.average(image[0, :]) +
                 np.average(image[-1, :]) +
                 np.average(image[:, 0]) +
                 np.average(image[:, -1])) / 4
    image_avg = (image.max() - image.min()) / 2

    if edges_avg > image_avg:
        image = 255 - image

    return image


def crop(image: np.array) -> np.array:
    ratio = image.shape[1] / image.shape[0]  # columns / rows
    monochrome = to_monochrome(image)

    coords = cv2.findNonZero(monochrome)
    x, y, columns, rows = cv2.boundingRect(coords)

    if columns < rows * ratio:
        x = max(x - int(rows * ratio - columns) // 2, 0)
        columns = int(rows * ratio)
    elif columns > rows * ratio:
        y = max(y - int(columns - rows * ratio) // 2, 0)
        rows = int(columns / ratio)

    return image[y:y+rows, x:x+columns]

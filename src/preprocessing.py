from operator import contains
from typing import List, Dict, Tuple
from shutil import Error
import cv2
import os
import numpy as np


def process_image(image: np.ndarray, dimensions: Tuple[int, int]) -> np.ndarray:
    image = to_bgr(image)
    image = highlight_foreground(image)
    image = crop(image)
    image = resize(image, dimensions)
    image = to_monochrome(image)
    return image


def process_dataset(images: Dict[str, List[np.ndarray]], dimensions: Tuple[int, int]) -> Dict[str, List[np.ndarray]]:
    processed = {}
    for label, imagesList in images.items():
        processed[label] = []

        for image in imagesList:
            image = process_image(image, dimensions)
            processed[label].append(image)

    return processed

def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise Exception(f'Couldn\'t read image at {path}')

    return image


def load_images(path: str) -> Dict[str, List[np.ndarray]]:
    images = {}

    for guid in os.listdir(path):
        guidPath = os.path.join(path, guid)
        for className in os.listdir(os.path.join(guidPath)):
            if not contains(images, className):
                images[className] = []

            classPath = os.path.join(guidPath, className)
            for file in os.listdir(classPath):
                filepath = os.path.join(classPath, file)
                image = load_image(filepath)
                images[className].append(image)

    return images


def load_merged_images(path: str) -> Dict[str, List[np.ndarray]]:
    images = {}

    for className in os.listdir(os.path.join(path)):
        if not contains(images, className):
            images[className] = []

        classPath = os.path.join(path, className)
        for file in os.listdir(classPath):
            filepath = os.path.join(classPath, file)
            image = load_image(filepath)
            images[className].append(image)

    return images


def save_images(path: str, images: Dict[str, List[np.ndarray]]) -> None:
    for className, imagesList in images.items():
        folder = os.path.join(path, className)
        os.makedirs(folder, exist_ok=True)

        for index, image in enumerate(imagesList):
            file = os.path.join(folder, '{:04}.png'.format(index))
            cv2.imwrite(file, image)


def resize(image: np.ndarray, dim: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def to_bgr(image: np.ndarray) -> np.array:
    image = np.copy(image)
    if image.shape[2] == 4 and np.max(image[:, :, 3]) - np.min(image[:, :, 3]) >= 128:
        alpha = image[:, :, 3]
        image[:, :, 0] = alpha
        image[:, :, 1] = alpha
        image[:, :, 2] = alpha

    return image


def to_monochrome(image: np.ndarray) -> np.ndarray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)

    return image


def highlight_foreground(image: np.ndarray) -> np.ndarray:
    monochrome = to_monochrome(image)

    edges_avg = (np.average(monochrome[0, :]) +
                 np.average(monochrome[-1, :]) +
                 np.average(monochrome[:, 0]) +
                 np.average(monochrome[:, -1])) / 4
    image_avg = (monochrome.max() - monochrome.min()) / 2

    if edges_avg > image_avg:
        image = 255 - image

    return image


def crop(image: np.ndarray) -> np.ndarray:
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


def split(images: Dict[str, List[np.ndarray]], ratio: float) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]]]:
    if ratio < 0 or ratio > 1:
        raise Error('Incorrect ratio: ratio must be in range [0, 1]')

    if ratio == 0:
        return {}, images

    if ratio == 1:
        return images, {}

    train = {}
    test = {}

    for label, imagesList in images.items():
        edge = int(len(imagesList) * ratio)
        train[label] = imagesList[:edge]
        test[label] = imagesList[edge:]

    return train, test

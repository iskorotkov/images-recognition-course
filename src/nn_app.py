import os
from typing import List
import preprocessing
import prettytable
import nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use('tkagg')


data_folder = './data/images'
train_folder = './data/train'
val_folder = './data/val'
test_folder = './data/test'
model_save_location = './models/nn'

train_val_split = 1

dimensions = (64, 64)
shape = (dimensions[0], dimensions[1], 1)

n_classes = 36
epochs = 20
batch_size = 8

mapping = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def prepare():
    images = preprocessing.load_images(data_folder)
    train, val = preprocessing.split(images, train_val_split)
    train = preprocessing.process_dataset(train, dimensions)

    preprocessing.save_images(train_folder, train)
    preprocessing.save_images(val_folder, val)


def encode(labels: List[str]) -> np.ndarray:
    return np.array([mapping.index(label.upper()) for label in labels])


def decode(predicted: np.ndarray) -> List[str]:
    encoded = np.argmax(predicted, axis=1)
    return [mapping[index].upper() for index in encoded]


def train():
    network = nn.NeuralNetwork(n_classes, shape)

    dataset = preprocessing.load_merged_images(train_folder)

    x = []
    y = []
    for label, images in dataset.items():
        y.extend([label for _ in images])
        x.extend(images)

    x = np.array([image[:, :, np.newaxis] for image in x])
    y = encode(y)

    network.fit(x, y, epochs, batch_size)

    network.save_model(model_save_location)


def validate():
    network = nn.NeuralNetwork(n_classes, shape)
    network.load_model(model_save_location)

    images = preprocessing.load_merged_images(val_folder)
    images = preprocessing.process_dataset(images, dimensions)

    x = []
    y = []
    for label, images in images.items():
        y.extend([label for _ in images])
        x.extend(images)

    x = np.array([image[:, :, np.newaxis] for image in x])

    predicted = network.predict(x)
    predicted = decode(predicted)

    table = prettytable.PrettyTable()
    table.add_column("Actual", y)
    table.add_column("Predicted", predicted)
    print(table)

    correctness = [x == y for x, y in zip(y, predicted)]

    total = len(y)
    correct = correctness.count(True)

    print(
        f'Recognizing {total} images: {correct} were recognized correctly ({correct / total * 100}%)')


def test():
    network = nn.NeuralNetwork(n_classes, shape)
    network.load_model(model_save_location)

    images = []
    for dirpath, _, filenames in os.walk(test_folder):
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            image = preprocessing.load_image(path)
            image = preprocessing.process_image(image, dimensions)
            images.append(image)

    x = np.array([image[:, :, np.newaxis] for image in images])

    predicted = network.predict(x)
    predicted = decode(predicted)

    items = list(zip(images, predicted))
    shown = items # [item for index, item in enumerate(items) if index % 10 == 0] # Shows each 10th image.

    plt.figure('Neural network predictions')

    for index, (image, label) in enumerate(shown):
        cols = 20
        rows = (len(shown) + cols - 1) // cols

        plt.subplot(rows, cols, index + 1)
        plt.title(label)
        plt.imshow(image, cmap='gray')

    plt.show()


# prepare()
# train()
# validate()
test()

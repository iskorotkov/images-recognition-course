from typing import List
import preprocessing
import prettytable
import nn
import numpy as np


data_folder = './data/images'
train_folder = './data/train'
test_folder = './data/val'
model_save_location = './models/nn'

dimensions = (64, 64)
shape = (dimensions[0], dimensions[1], 1)

n_classes = 36
epochs = 20
batch_size = 8

mapping = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def prepare():
    images = preprocessing.load_images(data_folder)
    train, test = preprocessing.split(images, 0.85)
    train = preprocessing.process_dataset(train, dimensions)

    preprocessing.save_images(train_folder, train)
    preprocessing.save_images(test_folder, test)


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


def test():
    network = nn.NeuralNetwork(n_classes, shape)
    network.load_model(model_save_location)

    images = preprocessing.load_merged_images(test_folder)
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


prepare()
train()
test()

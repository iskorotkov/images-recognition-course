import random

import numpy as np
import tensorflow as tf
from tensorflow import keras

import dataset
import prettytable
import classification

print('TF version', tf.version.VERSION)

target = 'virginica'

# Seed
seed = 5
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Prepare dataset
head, x, labels = dataset.from_file('./data/iris.csv')
x, labels = dataset.shuffle(x, labels)

labels = np.array([1 if x == target else 0 for x in labels])

x_train, x_test = dataset.split(x, ratio=0.9)
y_train, y_test = dataset.split(labels, ratio=0.9)

model = classification.NN()
model.fit(x_train, y_train)

table = prettytable.PrettyTable(
    ['Actual', 'Predicted value', 'Predicted label'])

predictions = model.predict(x_test)
for actual, predicted in zip(y_test, predictions):
    table.add_row([actual, int(predicted > 0.5), predicted])

print(table)

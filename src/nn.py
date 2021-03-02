import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.activations import sigmoid

import dataset

print('TF version', tf.version.VERSION)

target = 'setosa'

# Seed
seed = 5
random.seed(seed)
np.random.seed(seed)

# Prepare dataset
head, x, labels = dataset.from_file('./data/iris.csv')
x, labels = dataset.shuffle(x, labels)

labels = np.array([1 if x == target else 0 for x in labels])

x_train, x_test = dataset.split(x, ratio=0.9)
y_train, y_test = dataset.split(labels, ratio=0.9)

model = keras.models.Sequential([
    keras.layers.Dense(4, activation='sigmoid'),
    keras.layers.Dense(4, activation='sigmoid'),
    keras.layers.Dense(1, activation='softmax')
])

model.compile(optimizer='adam',
              loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

fit = model.fit(x_train, y_train, epochs=5)
print('History (accuraty):', fit.history['accuracy'])

eval = model.evaluate(x_test, y_test)
print(f'Loss = {eval[0]}, accuracy = {eval[1]}')

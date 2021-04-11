from typing import Tuple
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow as tf


class NeuralNetwork:
    def __init__(self, n_classes: int, input_shape: Tuple[int, int, int]) -> None:
        model = Sequential([
            layers.experimental.preprocessing.Rescaling(
                1./255, input_shape=input_shape),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(n_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True), metrics=['accuracy'])

        self.model = model

    def fit(self, x, y, epochs: int, batch_size: int = None):
        return self.model.fit(x, y, epochs=epochs, batch_size=batch_size)

    def predict(self, x):
        return self.model.predict(x)

    def save_model(self, path: str):
        self.model.save_weights(path)

    def load_model(self, path: str):
        self.model.load_weights(path)

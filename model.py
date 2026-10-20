"""The CNN architecture."""
from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn(num_classes: int, input_shape: Tuple[int, int, int] = (28, 28, 1)) -> tf.keras.Model:
    """Build a small CNN for 28x28 grayscale images.

    Two "convolution blocks" learn visual features (edges, then strokes and
    shapes); the Dense layers use those features to decide which class it is.
    """
    return models.Sequential(
        [
            layers.Input(shape=input_shape),                       # grayscale image
            layers.Conv2D(32, 3, padding="same", activation="relu"),  # find edges
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),                                  # 28x28 -> 14x14
            layers.Dropout(0.25),                                   # fight overfitting
            layers.Conv2D(64, 3, padding="same", activation="relu"),  # find strokes/loops
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),                                  # 14x14 -> 7x7
            layers.Dropout(0.25),
            layers.Flatten(),                                       # 7*7*64 numbers in a row
            layers.Dense(128, activation="relu"),                   # combine the features
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),        # class probabilities
        ],
        name="handwriting_cnn",
    )

#!/usr/bin/env python
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

# allocate classes manually
class_names = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize images
X_train / 255.0
X_test / 255.0


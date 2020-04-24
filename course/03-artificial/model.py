"""
Code for building Artificial Neural Network

https://colab.research.google.com/drive/17jqM6EqCaDENJTcUvAuo6DOf3IKYC571
"""
import numpy as np
import tensorflow as tf
import pandas as pd
import datetime
from preprocessing import fnames, get_data

def get_model(data=None):
    """
    Returns a model designed to work with the fashion_mnist dataset
    """
    # get training_data
    if data == None:
        data = get_data()

    # get X_train and y_train from data
    X_train = data[0]
    y_train = data[1]
    del data

    # Initialise model
    model = tf.keras.models.Sequential()

    # Add first fully-connected hidden layer
    # Fully-connected means all nodes are connected
    model.add(
        tf.keras.layers.Dense(
            units=128,                          # number of neurons
            activation='relu',                  # ReLU function
            input_shape=(X_train.shape[1], )    # number of pixels = 28*28
        )
    )

    # Add second layer with dropout
    # Dropout layer means some nodes are not updated during
    # back-propagation
    model.add(tf.keras.layers.Dropout(0.2))

    # Add output layer, activated using softmax
    model.add(tf.keras.layers.Dense(
        units=10, # number of classes in the dataset (i.e 0-9)
        activation='softmax'
    ))

    # compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )

    # get a summary
    model.summary()

    # train the model
    model.fit(X_train, y_train, epochs=5)

    return model

if __name__ == '__main__':
    _ = get_model()
    del _

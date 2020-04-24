#!/usr/bin/env python
"""
Code for building Artificial Neural Network

https://colab.research.google.com/drive/17jqM6EqCaDENJTcUvAuo6DOf3IKYC571
"""
import numpy as np
import tensorflow as tf
import datetime
import pandas as pd
import os

# get dataset from tensorflow built-in datasets
from tensorflow.keras.datasets import fashion_mnist

fnames = ['X_train', 'X_test', 'y_train', 'y_test']

def _get_data() -> tuple:
    """
    Returns a preprocessed fashion_mnist dataset
    """
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    # The images in the X_train and X_test variables
    # should be normalised so that their values scale between [0, 1)
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # They also need to be reshaped to a 1-dimensional array
    X_train = X_train.reshape(-1, 28*28)
    X_test = X_test.reshape(-1, 28*28)

    # Write data to file
    data = [X_train, X_test, y_train, y_test]
    for i, fname in enumerate(fnames):
        entry = pd.DataFrame(data[i])
        entry.to_csv(f'{fname}.csv', index=False)
        del entry
    return X_train, X_test, y_train, y_test

def get_data() -> tuple:

    # import data if written to file, else take the objects from the
    # preprocessing script
    try:
        # Check that files exist
        for fname in fnames:
            assert os.path.exists(f'{fname}.csv')
        print('Files found! Importing now...')

        # import data to list of dataframes
        data = []
        for fname in fnames:
            print(f'Importing {fname}...')
            data.append(pd.read_csv(f'{fname}.csv'))

        # after import convert to numpy array
        X_train = data[0].values
        X_test = data[1].values
        y_train = data[2].values
        y_test = data[3].values

        data = (X_train, X_test, y_train, y_test)

        print('Successfully imported datasets')

    except AssertionError:
        print('Files not found, Importing manually now and writing to file...')
        X_train, X_test, y_train, y_test = _get_data()
        data = (X_train, X_test, y_train, y_test)

    return data

if __name__ == '__main__':
    _ = get_data()
    del _

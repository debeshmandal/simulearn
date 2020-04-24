"""
Code for building Artificial Neural Network

https://colab.research.google.com/drive/17jqM6EqCaDENJTcUvAuo6DOf3IKYC571
"""
import numpy as np
import tensorflow as tf
import datetime
import pandas as pd

# get dataset from tensorflow built-in datasets
from tensorflow.keras.datasets import fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# The images in the X_train and X_test variables
# should be normalised so that their values scale between [0, 1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# They also need to be reshaped to a 1-dimensional array
X_train = X_train.reshape(-1, 28*28)
X_test = X_train.reshape(-1, 28*28)

# Write data to file
fnames = ['X_test', 'X_train', 'y_train', 'y_test']
datas = [X_test, X_train, y_train, y_test]
for i, fname in enumerate(fnames):
    data = pd.DataFrame(datas[i])
    print(data.head())
    data.to_csv(f'{fname}.csv', index=False)
    del data
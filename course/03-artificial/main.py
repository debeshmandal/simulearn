#/usr/bin/env python
"""
Main script for creating an evaluating an artificial neural network.
"""
import tensorflow as tf
import numpy as np
from model import get_model
from preprocessing import get_data

def main():
    # import model and data
    X_train, X_test, y_train, y_test = get_data()
    model = get_model(data=(X_train, y_train))

    # evaluate it
    test_lost, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.2f}")

    return model

if __name__ == '__main__':
    _ = main()
    del _


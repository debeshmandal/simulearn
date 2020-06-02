#!/usr/bin/env python
"""
Script for normalising and one-hot encoding input and output data
to return a training set and test set
"""
import pandas as pd
import numpy as np
from typing import Tuple

def split(df : pd.DataFrame) -> Tuple[pd.DataFrame]:
    X = df[[i for i in df.columns if i != 'dg']]
    y = df['dg']
    return X, y

def prepare(df : pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame()
    for column in df.columns:
        if column == 'charge_position':
            output['tail'] = df[column].apply(lambda x: 1 if x=='tail' else 0)
        else:
            output[column] = df[column].apply(lambda x: (x - df[column].mean())/df[column].std())
    return output

def import_data(fname : str) -> Tuple[pd.DataFrame]:
    data = prepare(pd.read_csv(fname, index_col=0))
    train = data.sample(frac=0.8,random_state=0)
    test = data.drop(train.index)

    X_train, y_train = split(train)
    X_test, y_test = split(test)

    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    print(import_data('results.csv')[0])
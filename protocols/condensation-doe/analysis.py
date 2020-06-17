import pandas as pd
import matplotlib.pyplot as plt

from preprocess import import_data
from model import build_model
#import tensorflow_docs as tfdocs
#import tensorflow_docs.plots
#import tensorflow_docs.modeling

def fit(X_train, y_train, epochs=200):
    model = build_model(len(X_train.columns))
    print(model.summary())

    history = model.fit(
        X_train, 
        y_train, 
        epochs=epochs, 
        validation_split=0.1, 
        verbose=0,
    )

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()
    return model, hist

#plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
#plotter.plot({'Basic': history}, metric = "mae")
def plot(data, metric='mae'):
    fig, ax = plt.subplots()
    ax.plot(data['epoch'], data[metric])
    ax.set_xlabel('Epochs')
    ax.set_ylabel(f"{metric.upper()} [$\Delta G$]")

def predict(X_test, y_test, model, plot=True):
    y_pred = model.predict(X_test).flatten()
    if plot:
        fig, ax = plt.subplots()
        ax.plot(y_test, y_pred, 'bo')
        test_min = y_test.min()
        test_max = y_test.max()
        ax.plot([test_min, test_max], [test_min, test_max], 'k-')
        ax.set_xlabel('Test Values')
        ax.set_ylabel('Predicted Values')
        
    return y_pred

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = import_data('results.csv')
    model, hist = fit(X_train, y_train)
    plot(hist)
    predict(X_test, y_test, model)
    plt.show()


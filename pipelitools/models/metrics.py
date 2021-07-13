import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import os

from sklearn.metrics import classification_report, confusion_matrix

from pipelitools import utils as u


def test_metrics():
    """ """
    print('test_metrics: ok')


def plot_confusion_matrix(y_test,
                          y_pred,
                          labels,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """Plots the confusion matrix. Normalization can be applied by setting `normalize=True`.

    Parameters
    ----------
    y_test : np.ndarray
        Testing values.
    y_pred : np.ndarray
        Predicted values.
    labels : np.ndarray
        Unique labels.
    normalize : bool
         (Default value = False)
         True if need to normalize the data & represent as percentage out of 100%.
    title : str
         (Default value = 'Confusion matrix')
         Title of the plot.
    cmap : matplotlib.colors.LinearSegmentedColormap
         (Default value = plt.cm.Blues)
         Color map.

    Returns
    -------
    Confusion matrix plot.
    """

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred, labels=labels)
    np.set_printoptions(precision=2)

    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]*100

    plt.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)

    fmt = '.1f' if normalize else 'd'
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def metrics_report(model, name, X_test, y_test, y_train, data='test'):
    """ Creates a report on a model. Saves confusion matrix and metrics to classification_report_{data} folder.

    Note: it is recommended that data is specified explicitly!

    Parameters
    ----------
    model : fitted model (e.g. sklearn.model_selection._search.GridSearchCV if GridSearchCV is used).
        The fitted model.
    name : str
        Name of the fitted model.
    X_test : pd.DataFrame.
        Features used for testing.
    y_test : pd.Series
        Responses used for testing.
    y_train : pd.Series
        Responses used for training.
    data : bool
         (Default value = 'test')
        Type of report to create depending on input data: 'test' for testing, 'validation' for validation.

    Returns
    -------
    Prints out metrics from clasification_report and confusion matrix.
    They are also saved in folder classification_report_{data}.
    """

    assert data in ['test', 'validation'], "Parameter 'data' must be either 'test', or 'validation'."

    y_pred = model.predict(X_test)

    a = classification_report(y_test, y_pred, labels=np.unique(y_train))
    u.export_str(a, f"./temp_report_{data}/{name}.txt")
    print(a)

    # plot the confusion matrix
    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.values

    plot_confusion_matrix(y_test=y_test.values,
                          y_pred=y_pred,
                          labels=np.unique(y_test),
                          normalize=True,
                          title=f'Confusion matrix for {name}',
                          cmap=plt.cm.Blues)

    # save figure to folder 'fig'
    if os.path.exists("fig") is False:
        os.mkdir("fig")
    plt.savefig(f"./temp_report_{data}/{name}.png", dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    test_metrics()

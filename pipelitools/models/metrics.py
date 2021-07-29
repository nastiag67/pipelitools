import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import os
import seaborn as sns
sns.color_palette("tab10")

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_recall_curve

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
    data : bool, default='test'
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
    if os.path.exists(f"./temp_report_{data}") is False:
        os.mkdir(f"./temp_report_{data}")
    plt.savefig(f"./temp_report_{data}/{name}.png", dpi=300, bbox_inches='tight')


def learning_cuve(training, validation, name='Metric'):
    """
    training : list
        List of training metrics.

    validation : list
        List of validation metrics.
    """
    plt.plot(training, label='train', color='C0', linestyle='-')
    plt.plot(validation, label='test', color='C0', linestyle=':')
    plt.title(f"{name} learning curve")
    plt.xlabel("Epochs")
    plt.ylabel(name)
    plt.legend()
    plt.show()


def CM(name, y_train, y_pred, y_test, data='validation'):
    """
    y_pred, y_test must be np.array
    """

    assert data in ['test', 'validation'], "Parameter 'data' must be either 'test', or 'validation'."

    try:
        y_test.shape[1] and y_pred.shape[1] and y_train.shape[1]
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test, axis=1)
        y_train = np.argmax(y_train, axis=1)
    except IndexError:
        pass

    a = classification_report(y_test, y_pred, labels=np.unique(y_train))
    u.export_str(a, f"./temp_report_{data}/{name}.txt")
    print(a)

    # plot the confusion matrix
    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.values

    plot_confusion_matrix(y_test=y_test,
                          y_pred=y_pred,
                          labels=np.unique(y_test),
                          normalize=True,
                          title=f'Confusion matrix for {name}',
                          cmap=plt.cm.Blues)

    # save figure to folder 'fig'
    if os.path.exists(f"./temp_report_{data}") is False:
        os.mkdir(f"./temp_report_{data}")
    plt.savefig(f"./temp_report_{data}/{name}.png", dpi=300, bbox_inches='tight')
    plt.show()


def ROCcurve_multiclass(name, y_train, y_pred, y_test, data='validation'):

    try:
        y_test.shape[1] and y_pred.shape[1] and y_train.shape[1]
    except IndexError:
        raise ValueError(
            'In a multiclass case, y_test and y_pred must be converted into a matrix of dummy variables \
            \n(e.g. using np.array(pd.get_dummies(y_pred))).')

    print(f"ROC-AUC score: {round(roc_auc_score(y_test, y_pred, average='macro'), 4)}")

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = y_train.shape[1]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC curve and ROC area
    fpr["macro"], tpr["macro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot ROC curve
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='class {0} (area = {1:0.2f})'
                                       ''.format(i, roc_auc[i]))

    plt.plot(fpr["macro"], tpr["macro"], '--',
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.02, 1])
    plt.ylim([0, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"ROC curve for {name}")
    plt.legend(loc="lower right")

    if os.path.exists(f"./temp_report_{data}") is False:
        os.mkdir(f"./temp_report_{data}")
    plt.savefig(f"./temp_report_{data}/{name}_ROC.png", dpi=300, bbox_inches='tight')

    plt.show()


def PR_multiclass(name, y_train, y_pred, y_test, data='validation'):
    """ Precision-Recall curve for multiclass classification.
    """

    try:
        y_test.shape[1] and y_pred.shape[1] and y_train.shape[1]
    except IndexError:
        raise ValueError(
            'In a multiclass case, y_test and y_pred must be converted into a matrix of dummy variables \
            \n(e.g. using np.array(pd.get_dummies(y_pred))).')

    # precision recall curve
    precision = dict()
    recall = dict()
    n_classes = y_train.shape[1]
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_pred[:, i])
        plt.plot(recall[i], precision[i], lw=2,
                 label=f"class {i} (area = {round(auc(recall[i], precision[i]), 2)})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    plt.title(f"Precision-Recall curve for {name}")

    if os.path.exists(f"./temp_report_{data}") is False:
        os.mkdir(f"./temp_report_{data}")
    plt.savefig(f"./temp_report_{data}/{name}_PR.png", dpi=300, bbox_inches='tight')

    plt.show()


def compare_models(model, name, X_test, y_test, y_train, *args, proba=True, data='validation'):
    """ Executes the functions specified in *args applied to a particular classification model.

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
    *args : arguments
        Functions which calculate the necessary metrics (e.g. roc = mt.ROCcurve_multiclass).
        Should have the arguments: arg(name, y_train, y_pred, y_test, data='validation')
    proba : bool
        True to calculate probability of occurrence of a class.
    data : bool, default='test'
        Type of report to create depending on input data: 'test' for testing, 'validation' for validation.

    Returns
    -------
    Executes the functions specified in *args applied to a particular classification model.
    """
    if proba:
        y_pred = model.predict_proba(X_test)
    else:
        y_pred = model.predict(X_test)
        y_pred = np.array(pd.get_dummies(y_pred))

    for arg in args:
        arg(name, y_train, y_pred, y_test, data=data)


if __name__ == '__main__':
    test_metrics()

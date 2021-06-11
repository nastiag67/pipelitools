import matplotlib.pyplot as plt
import numpy as np
import itertools

from sklearn.metrics import jaccard_score, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


def test_metrics():
    print('test_metrics: ok')


def metrics(model, X_test, y_test):
    """
    """
    if type(model) is LogisticRegression:
        LR_yhat = model.predict(X_test)
        LR_yhat_prob = model.predict_proba(X_test)
        jss = jaccard_score(y_test, LR_yhat, average='micro')
        f1s = f1_score(y_test, LR_yhat, average='weighted')
        lls = log_loss(y_test, LR_yhat_prob)
        print(str(type(model)).split('.')[-1][:-2])
        print("Jaccard index: %.2f" % jss)
        print("F1-score: %.2f" % f1s)
        print("LogLoss: %.2f" % lls)

    elif type(model) is KNeighborsClassifier or type(model) is DecisionTreeClassifier or type(model) is svm.SVC:
        yhat = model.predict(X_test)
        # average is required for multiclass/multilabel targets
        jss = jaccard_score(y_test, yhat, average='micro')
        f1s = f1_score(y_test, yhat, average='weighted')
        print(str(type(model)).split('.')[-1][:-2])
        print("Jaccard index: %.2f" % jss)
        print("F1-score: %.2f" % f1s)

    print()


def classification_metrics(X_train, y_train, y_test, y_pred, model, average='binary'):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=average)
    recall = recall_score(y_test, y_pred, average=average)
    f1s = f1_score(y_test, y_pred, average=average)
    print(f"F1-score: {round(f1s, 4)}")
    print(f"Precision: {round(precision, 4)}")
    print(f"Recall: {round(recall, 4)}")
    print(f"Accuracy on train data: {round(model.score(X_train, y_train), 4)}")
    print(f"Accuracy on test data: {round(accuracy, 4)}")

    return accuracy, precision, recall, f1s


def plot_confusion_matrix(y_test,
                          y_pred,
                          labels,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred, labels=labels)
    np.set_printoptions(precision=2)

    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]*100
#         print("Normalized confusion matrix")

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


if __name__ == '__main__':
    test_metrics()
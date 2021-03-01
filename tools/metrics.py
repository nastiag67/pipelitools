from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


def test_metrics():
    print('test metrics: ok')


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


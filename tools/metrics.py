from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


def metrics(model, X_test, y_test):
    """
    Based on the count of each section, we can calculate precision and recall of each label:
    - __Precision__ is a measure of the accuracy provided that a class label has been predicted. It is defined by:
    precision = TP / (TP + FP)
    - __Recall__ is true positive rate. It is defined as: Recall =  TP / (TP + FN)
    So, we can calculate precision and recall of each class.
    __F1 score:__
    Now we are in the position to calculate the F1 scores for each label based on the precision and recall of that label
    The F1 score is the harmonic average of the precision and recall, where an F1 score reaches its best value at 1
    (perfect precision and recall) and worst at 0. It is a good way to show that a classifer has a good value for both
    recall and precision.
    """
    if type(model) is LogisticRegression:
        LR_yhat = model.predict(X_test)
        LR_yhat_prob = model.predict_proba(X_test)
        jss = jaccard_similarity_score(y_test, LR_yhat)
        f1s = f1_score(y_test, LR_yhat, average='weighted')
        lls = log_loss(y_test, LR_yhat_prob)
        print(str(type(model)).split('.')[-1][:-2])
        print("Jaccard index: %.2f" % jss)
        print("F1-score: %.2f" % f1s)
        print("LogLoss: %.2f" % lls)

    elif type(model) is KNeighborsClassifier or type(model) is DecisionTreeClassifier or type(model) is svm.SVC:
        yhat = model.predict(X_test)
        jss = jaccard_similarity_score(y_test, yhat)
        f1s = f1_score(y_test, yhat, average='weighted')
        print(str(type(model)).split('.')[-1][:-2])
        print("Jaccard index: %.2f" % jss)
        print("F1-score: %.2f" % f1s)

    print()


"""Tests for `metrics` module."""

import pytest
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_classification

from pipelitools.models import metrics as mt
from pipelitools.models import models as m


@pytest.fixture(scope="function")
def df_binary():
    X_train, y_train = make_classification(n_samples=100, n_features=2, n_informative=2,
                                           n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=1,
                                           class_sep=2, flip_y=0, weights=[0.5, 0.5], random_state=1)
    X_test, y_test = make_classification(n_samples=50, n_features=2, n_informative=2,
                                         n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=1,
                                         class_sep=2, flip_y=0, weights=[0.5, 0.5], random_state=2)
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)
    return X_train, y_train, X_test, y_test


def test_metrics_report(df_binary):
    X_train, y_train, X_test, y_test = df_binary

    name = 'LR'
    model = LogisticRegression()
    steps = [
        ('scaler', StandardScaler()),
    ]
    parameters = {
        'LR__class_weight': ['balanced'],
        'LR__multi_class': ['multinomial', 'auto'],
    }
    average = 'binary'
    multiclass = False
    metric = 'accuracy'
    randomized_search = False
    nfolds = 5
    n_jobs = None
    verbose = 0
    cls = m.Model(X_train, y_train, X_test, y_test)
    model_lr, y_pred_lr = cls.checkmodel(name, model, steps, parameters, average, multiclass, metric,
                                         randomized_search, nfolds, n_jobs, verbose)

    mt.metrics_report(model_lr, name, X_test, y_test, y_train, data='test')


















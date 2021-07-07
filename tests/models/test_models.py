"""Tests for `models` module."""

import pytest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from tools.models import models as m

indices_dict = {
    "X train": pd.DataFrame(data={'X1': [1, 2, 32, 4],
                                  'X2': [564, 698, 2, 415]}),
    "y train": pd.DataFrame(data={'y': [1, 1, 0, 1]}),
    "X test": pd.DataFrame(data={'X1': [3, 45],
                                 'X2': [514, 1]}),
    "y test": pd.DataFrame(data={'y': [1, 0]}),
}


@pytest.fixture(params=indices_dict.keys())
def X_train(request):
    """
    Fixture for dataframes.
    """
    return indices_dict[request.param].copy()


@pytest.fixture(params=indices_dict.keys())
def X_test(request):
    """
    Fixture for dataframes.
    """
    return indices_dict[request.param].copy()


@pytest.fixture(params=indices_dict.keys())
def y_train(request):
    """
    Fixture for dataframes.
    """
    return indices_dict[request.param].copy()


@pytest.fixture(params=indices_dict.keys())
def y_test(request):
    """
    Fixture for dataframes.
    """
    return indices_dict[request.param].copy()


def test_checkmodel(X_train, X_test, y_train, y_test):
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
    cls = m.Model(X_train, X_test, y_train, y_test)
    model_lr, y_pred_lr = cls.checkmodel(name, model, steps, parameters, average, multiclass, metric,
                   randomized_search, nfolds, n_jobs, verbose)
    print(y_pred_lr)

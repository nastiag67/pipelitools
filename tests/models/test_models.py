"""Tests for `models` module."""

import pytest
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_classification

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


@pytest.fixture(scope="function")
def df_multiclass():
    X_train, y_train = make_classification(n_samples=100, n_features=2, n_informative=2,
                                           n_redundant=0, n_repeated=0, n_classes=3, n_clusters_per_class=1,
                                           class_sep=2, flip_y=0, weights=[0.2, 0.3, 0.5], random_state=1)
    X_test, y_test = make_classification(n_samples=50, n_features=2, n_informative=2,
                                         n_redundant=0, n_repeated=0, n_classes=3, n_clusters_per_class=1,
                                         class_sep=2, flip_y=0, weights=[0.3, 0.3, 0.4], random_state=2)
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)
    return X_train, y_train, X_test, y_test


class TestBinaryClassification:

    def test_basic(self, df_binary):
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

    def test_default_params(self, df_binary):
        X_train, y_train, X_test, y_test = df_binary
        name = 'LR'
        model = LogisticRegression()
        steps = [
            ('scaler', StandardScaler()),
        ]
        parameters = {
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

    def test_no_steps(self, df_binary):
        X_train, y_train, X_test, y_test = df_binary
        name = 'LR'
        model = LogisticRegression()
        steps = [
            # ('scaler', StandardScaler()),
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

    def test_randomCV(self, df_binary):
        # does no converge - it's okay
        X_train, y_train, X_test, y_test = df_binary
        name = 'MLP'
        model = MLPClassifier(random_state=42)
        steps = [
            ('scaler', StandardScaler()),
        ]
        parameters = {
            'MLP__activation': ['identity', 'logistic', 'tanh', 'relu'],
            'MLP__solver': ['sgd', 'adam'],
            'MLP__alpha': [0.01, 0.05, 0.1],
        }
        average = 'binary'
        multiclass = False
        metric = 'accuracy'
        randomized_search = True
        nfolds = 5
        n_jobs = None
        verbose = 0
        cls = m.Model(X_train, y_train, X_test, y_test)
        model_lr, y_pred_lr = cls.checkmodel(name, model, steps, parameters, average, multiclass, metric,
                                             randomized_search, nfolds, n_jobs, verbose)


class TestMulticlassClassification:

    def test_basic(self, df_multiclass):
        X_train, y_train, X_test, y_test = df_multiclass
        name = 'LR'
        model = LogisticRegression()
        steps = [
            ('scaler', StandardScaler()),
        ]
        parameters = {
            'LR__class_weight': ['balanced'],
            'LR__multi_class': ['multinomial', 'auto'],
        }
        average = 'macro'
        multiclass = True
        metric = 'recall'
        randomized_search = False
        nfolds = 5
        n_jobs = None
        verbose = 0
        cls = m.Model(X_train, y_train, X_test, y_test)
        model_lr, y_pred_lr = cls.checkmodel(name, model, steps, parameters, average, multiclass, metric,
                                             randomized_search, nfolds, n_jobs, verbose)

    def test_default_params(self, df_multiclass):
        X_train, y_train, X_test, y_test = df_multiclass
        name = 'LR'
        model = LogisticRegression()
        steps = [
            ('scaler', StandardScaler()),
        ]
        parameters = {
        }
        average = 'macro'
        multiclass = True
        metric = 'recall'
        randomized_search = False
        nfolds = 5
        n_jobs = None
        verbose = 0
        cls = m.Model(X_train, y_train, X_test, y_test)
        model_lr, y_pred_lr = cls.checkmodel(name, model, steps, parameters, average, multiclass, metric,
                                             randomized_search, nfolds, n_jobs, verbose)

    def test_no_steps(self, df_multiclass):
        X_train, y_train, X_test, y_test = df_multiclass
        name = 'LR'
        model = LogisticRegression()
        steps = [
            # ('scaler', StandardScaler()),
        ]
        parameters = {
            'LR__class_weight': ['balanced'],
            'LR__multi_class': ['multinomial', 'auto'],
        }
        average = 'macro'
        multiclass = True
        metric = 'recall'
        randomized_search = False
        nfolds = 5
        n_jobs = None
        verbose = 0
        cls = m.Model(X_train, y_train, X_test, y_test)
        model_lr, y_pred_lr = cls.checkmodel(name, model, steps, parameters, average, multiclass, metric,
                                             randomized_search, nfolds, n_jobs, verbose)

    def test_randomCV(self, df_multiclass):
        # does no converge - it's okay
        X_train, y_train, X_test, y_test = df_multiclass
        name = 'MLP'
        model = MLPClassifier(random_state=42)
        steps = [
            ('scaler', StandardScaler()),
        ]
        parameters = {
            'MLP__activation': ['identity', 'logistic', 'tanh', 'relu'],
            'MLP__solver': ['sgd', 'adam'],
            'MLP__alpha': [0.01, 0.05, 0.1],
        }
        average = 'macro'
        multiclass = True
        metric = 'recall'
        randomized_search = True
        nfolds = 5
        n_jobs = None
        verbose = 0
        cls = m.Model(X_train, y_train, X_test, y_test)
        model_lr, y_pred_lr = cls.checkmodel(name, model, steps, parameters, average, multiclass, metric,
                                             randomized_search, nfolds, n_jobs, verbose)














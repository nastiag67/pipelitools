"""Tests for `clustering` module."""

import pytest
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.decomposition import PCA

from pipelitools.models import clustering as c


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


def test_basic(df_binary):
    X_train, y_train, X_test, y_test = df_binary
    ncomp = 2
    pca = PCA(n_components=ncomp, whiten=True)
    pca.fit(X_train)  # estimate the parameters of the PCA
    Z = pca.transform(X_train)

    c.n_clusters(Z, Kmax=10, n_init=100)





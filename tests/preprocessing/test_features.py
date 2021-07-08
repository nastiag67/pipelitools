import pytest
import pandas as pd

from pipelitools.preprocessing import features as f
from sklearn.datasets import make_classification

# indices_dict = {
#     "dataframe": pd.DataFrame(data={'a': [1, 2, 32, 4],
#                                     'dates': ['21.04.2021', '31.05.2020', '01.07.2021', '30.01.2000'],
#                                     'values': [564, 698, 2, 415],
#                                     'values2': [1,3,9,2]})
# }
#
#
# @pytest.fixture(params=indices_dict.keys())
# def df(request):
#     """
#     Fixture for dataframes.
#     """
#     return indices_dict[request.param].copy()


@pytest.fixture(scope="function")
def df_binary():
    X_train, y_train = make_classification(n_samples=100, n_features=50, n_informative=2,
                                           n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=1,
                                           class_sep=2, flip_y=0, weights=[0.5, 0.5], random_state=1)
    X_test, y_test = make_classification(n_samples=50, n_features=50, n_informative=2,
                                         n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=1,
                                         class_sep=2, flip_y=0, weights=[0.5, 0.5], random_state=2)
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)
    return X_train, y_train, X_test, y_test


def test_low_variance(df_binary):
    X_train, y_train, X_test, y_test = df_binary
    df = pd.DataFrame(X_train)
    df['y'] = y_train
    cls = f.FeatureSelectionPipeline(df)


def test_RFE_selection(df_binary):
    X_train, y_train, X_test, y_test = df_binary
    df = pd.DataFrame(X_train)
    df['y'] = y_train
    cls = f.FeatureSelectionPipeline(df)













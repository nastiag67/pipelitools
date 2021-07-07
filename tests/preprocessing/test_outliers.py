"""Tests for `outliers` module."""

import pytest
import pandas as pd

from tools.preprocessing import outliers as o

indices_dict = {
    "dataframe": pd.DataFrame(data={'a': [1, 2, 567, 4],
                                    'dates': ['21.04.2021', '31.05.2020', '01.07.2021', '30.01.2000'],
                                    'values': [5674, 6988, 2, 5515]})
}


@pytest.fixture(params=indices_dict.keys())
def df(request):
    """
    Fixture for dataframes.
    """
    return indices_dict[request.param].copy()


def test__z_score(df):
    cls = o.Outliers(df)
    df_clean, df_outliers = cls._z_score(columns=['a', 'values'], threshold=1)
    assert list(df_clean.index) == [0,1,3]
    assert df_outliers.index == 2


def test__IQR(df):
    cls = o.Outliers(df)
    df_clean, df_outliers = cls._IQR(columns=['a', 'values'], q1=0.25)
    assert list(df_clean.index) == [0,1,3]
    assert df_outliers.index == 2


def test_show_outliers(df):
    cls = o.Outliers(df)
    df_clean, df_outliers, df = cls.show_outliers(columns=['a', 'values'], how='IQR', q1=0.25)
    # print(list(df_clean.index))
    # print('**************************')
    # print(df_outliers.index)
    assert list(df_clean.index) == [0,1,3]
    assert df_outliers.index == 2

    df_clean, df_outliers, df = cls.show_outliers(columns=['a', 'values'], how='z_score', threshold=1)
    assert list(df_clean.index) == [0,1,3]
    assert df_outliers.index == 2









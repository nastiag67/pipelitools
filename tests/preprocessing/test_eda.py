"""Tests for `eda` module."""

import pytest
import pandas as pd

from tools.preprocessing import eda

indices_dict = {
    "dataframe": pd.DataFrame(data={'a': [1, 2, 32, 4],
                                    'dates': ['21.04.2021', '31.05.2020', '01.07.2021', '30.01.2000'],
                                    'values': [564, 698, 2, 415]})
}


@pytest.fixture(params=indices_dict.keys())
def df(request):
    """
    Fixture for dataframes.
    """
    return indices_dict[request.param].copy()


def test_get_df(df):
    cls = eda.Dataset(df)
    new_df = cls.get_df()
    assert new_df.shape == df.shape


def test_get_randomdata(df):
    n = 2
    cls = eda.Dataset(df)
    new_df = cls.get_randomdata(n)
    assert new_df.shape[0] == n







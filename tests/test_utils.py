"""Tests for `utils` module."""

import pytest
import pandas as pd

from tools import utils as u


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


def test_to_dates(df):
    cols = ['dates']
    df_new = u.to_dates(df, cols)
    assert df_new.dates.dtypes == '<M8[ns]'


def test_get_new_date():
    assert u.get_new_date('21.04.3243', 5) == '26.04.3243'
    assert u.get_new_date('30.04.3243', 5) == '05.05.3243'
    assert u.get_new_date('01.04.3243', 5) == '06.04.3243'
    assert u.get_new_date('31.05.3243', 5) == '05.06.3243'
    assert u.get_new_date('30.05.3243', 5) == '04.06.3243'


def test_get_first_day():
    assert u.get_first_day('30.05.3243') == '01.05.3243'
    assert u.get_first_day('01.05.3243') == '01.05.3243'


def test_offset_end_date():
    assert u.offset_end_date('30.05.2000', 0) == '30.05.2000'
    assert u.offset_end_date('30.05.2000', 5) == '31.10.2000'
    assert u.offset_end_date('30.05.2000', -5) == '31.12.1999'


def test_offset_first_date():
    assert u.offset_end_date('31.05.2000', 5) == '31.10.2000'
    assert u.offset_end_date('31.05.2000', -5) == '31.12.1999'
    assert u.offset_end_date('31.05.2000', 0) == '31.05.2000'



























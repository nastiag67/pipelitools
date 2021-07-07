import pytest
import pandas as pd

from tools.preprocessing import features as f

indices_dict = {
    "dataframe": pd.DataFrame(data={'a': [1, 2, 32, 4],
                                    'dates': ['21.04.2021', '31.05.2020', '01.07.2021', '30.01.2000'],
                                    'values': [564, 698, 2, 415],
                                    'values2': [1,3,9,2]})
}


@pytest.fixture(params=indices_dict.keys())
def df(request):
    """
    Fixture for dataframes.
    """
    return indices_dict[request.param].copy()


def test_low_variance(df):
    cls = f.FeatureSelectionPipeline(df)
    new_df = cls.low_variance(1)
    # print(new_df)
    assert new_df.shape[1]-1 == 0


def test_RFE_selection(df):
    df0 = df[['a', 'values', 'values2']]
    cls = f.FeatureSelectionPipeline(df0)
    new_df, _ = cls.RFE_selection(n_features_to_select=2, step=1)
    # print(new_df)
    assert new_df.shape[1]-1 == 2













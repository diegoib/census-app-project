
import pytest
import pandas as pd
import joblib

@pytest.fixture(scope='session')
def data():
    df = pd.read_csv('../data/census_clean.csv')
    return df



import pytest
import pandas as pd
import joblib

@pytest.fixture(scope='session')
def data():
    df = pd.read_csv('data/census_clean.csv')
    return df

@pytest.fixture(scope='session')
def sample():
    df = pd.read_csv('data/census_clean.csv')
    df.pop('salary')
    sample = df.iloc[1].to_dict()
    return sample


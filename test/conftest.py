
import pytest
import pandas as pd


@pytest.fixture(scope='session')
def data():
    df = pd.read_csv('data/census_clean.csv')
    return df

@pytest.fixture(scope='session')
def sample_negative():
    sample = {
        "age": 39, 
        "workclass": "State-gov", 
        "fnlgt": 77516, "education": 
        "Bachelors", 
        "education-num": 13, 
        "marital-status": "Never-married", 
        "occupation": "Adm-clerical", 
        "relationship": "Not-in-family", 
        "race": "White", "sex": "Male", 
        "capital-gain": 2174, 
        "capital-loss": 0, 
        "hours-per-week": 40, 
        "native-country": "United-States"}
    return sample

@pytest.fixture(scope='session')
def sample_positive():
    sample = {
        'age': 42,
        'workclass': 'Private',
        'fnlgt': 159449,
        'education': 'Bachelors',
        'education-num': 13,
        'marital-status': 'Married-civ-spouse',
        'occupation': 'Exec-managerial',
        'relationship': 'Husband',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 5178,
        'capital-loss': 0,
        'hours-per-week': 40,
        'native-country': 'United-States'}
    return sample

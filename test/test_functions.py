import sys
sys.path.append('../')

from src.ml.data import process_data
from src.ml.model import train_model, inference, compute_model_metrics

def test_process_data(data):
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X, y, _, _ = process_data(data, categorical_features=cat_features, label="salary")

    assert X.shape[0] > 0
    assert X.shape[1] > 0
    assert y.shape[0] > 0
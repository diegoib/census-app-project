import sys
sys.path.append('./')
from src.ml.model import train_model, inference
from src.ml.data import process_data
from sklearn.linear_model import LogisticRegression


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
    X, y, _, _ = process_data(
        data, categorical_features=cat_features, label="salary")

    assert X.shape[0] > 0
    assert X.shape[1] > 0
    assert y.shape[0] > 0


def test_train_model(data):
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
    X_train, y_train, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )

    model = train_model(X_train, y_train)
    assert isinstance(model, LogisticRegression)


def test_inference(data):
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
    X_train, y_train, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )

    model = train_model(X_train, y_train)
    preds = inference(model, X_train)

    assert preds.shape[0] > 0

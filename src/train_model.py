import sys
sys.path.append('./')
from src.ml.model import train_model, compute_model_metrics, \
    inference, evaluate_slices
from src.ml.data import process_data
import os
import joblib
import logging
import pandas as pd
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def ingest_data(path):
    file_path = os.path.join(path, "data", "census_clean.csv")
    df = pd.read_csv(file_path)
    return df


def go():

    path = os.getcwd()

    logger.info("Ingesting the data")
    data = ingest_data(path)

    logger.info("Splitting the data and processing it")
    # Optional enhancement, use K-fold cross validation instead of a
    # train-test split.
    train, test = train_test_split(data, test_size=0.20)

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
        train, categorical_features=cat_features, label="salary", training=True
    )
    # Proces the test data with the process_data function.
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Train and save a model.
    logger.info("Training model and making inference")
    model = train_model(X_train, y_train)

    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    logger.info("Precision: {:.2f}".format(precision))
    logger.info("Recall: {:.2f}".format(recall))
    logger.info("Fbeta: {:.2f}".format(fbeta))

    # Evaluate slices of the data
    logger.info("Evaluating slices")

    evaluate_slices(test, y_test, preds, ['sex', 'race', 'native-country'])

    # Dumps
    logger.info("Dumping objects")

    model_path = os.path.join(path, 'model', 'finalized_model.sav')
    joblib.dump(model, model_path)

    cat_encoder_path = os.path.join(path, 'model', 'cat_encoder.sav')
    joblib.dump(encoder, cat_encoder_path)

    lb_encoder_path = os.path.join(path, 'model', 'lb_encoder.sav')
    joblib.dump(lb, lb_encoder_path)


if __name__ == "__main__":

    go()

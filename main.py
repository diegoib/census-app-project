import sys
sys.path.append('./')
from src.schemas import Item, Response
from src.ml.model import inference
from src.ml.data import process_data
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
import os


# To run dvc in Heroku
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    print("Running DVC")
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("Pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

model = joblib.load('model/finalized_model.sav')
encoder = joblib.load('model/cat_encoder.sav')

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

app = FastAPI()


@app.get("/")
async def welcome_message():
    return "Welcome to the API. Check the docs at '/docs'"


@app.post("/predict", response_model=Response)
async def predict(input_data: Item):

    data = input_data.dict()
    input_df = pd.DataFrame(
        np.array(
            list(
                data.values())).reshape(
            1, -1), columns=data.keys())
    input_df.columns = [col.replace('_', '-') for col in input_df.columns]
    X, _, _, _ = process_data(
        input_df,
        categorical_features=cat_features,
        training=False,
        encoder=encoder)
    pred = inference(model, X)[0]
    return {'prediction': pred}

from typing import List

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel, Field
import pandas as pd
import json
import numpy as np

import joblib
from src.ml.data import process_data
from src.ml.model import inference

import sys
sys.path.append('./')

# Declare the data object with its components and their type.
class Item(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')

class Response(BaseModel):
    prediction: List


app = FastAPI()

# This allows sending of data (our TaggedItem) via POST to the API.
@app.get("/")
async def welcome_message():
    return "Welcome to the API. Check the docs at '/docs'"


@app.post("/predict")#, response_model=Response)
async def predict(input_data: Item):

    data = input_data.dict()
    input_df = pd.DataFrame(np.array(list(data.values())).reshape(1,-1), columns= data.keys())
    input_df.columns = [col.replace('_', '-') for col in input_df.columns]
    
#    return input_df

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
    X, _, _, _ = process_data(input_df, 
        categorical_features=cat_features, training=False, encoder=encoder
    )
    preds = inference(model, X)
    return {'prediction': list(preds)}

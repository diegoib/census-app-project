from fastapi.testclient import TestClient
import json

import sys
sys.path.append('./')

from main import app

client = TestClient(app)

def test_api_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "Welcome to the API. Check the docs at '/docs'"

def test_api_prediction_negative(sample_negative):
    r = client.post("/predict", data=json.dumps(sample_negative))
    prediction = r.json()
    assert r.status_code == 200
    assert prediction['prediction'] == 0

def test_api_prediction_positive(sample_positive):
    r = client.post("/predict", data=json.dumps(sample_positive))
    prediction = r.json()
    assert r.status_code == 200
    assert prediction['prediction'] == 1


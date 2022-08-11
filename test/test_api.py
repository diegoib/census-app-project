from fastapi.testclient import TestClient
import json

import sys
sys.path.append('./')

from main import app

# Instantiate the testing client with our app.
client = TestClient(app)

# Write tests using the same syntax as with the requests module.
def test_api_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200

def test_api_prediction():
    
    sample = {"age": 39, "workclass": "State-gov", "fnlgt": 77516, "education": "Bachelors", 
                "education-num": 13, "marital-status": "Never-married", "occupation": "Adm-clerical", 
                "relationship": "Not-in-family", "race": "White", "sex": "Male", "capital-gain": 2174, 
                "capital-loss": 0, "hours-per-week": 40, "native-country": "United-States"}
    r = client.post("/predict", data=json.dumps(sample))
    prediction = r.json()
    assert r.status_code == 200
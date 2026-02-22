import os
from fastapi import FastAPI
import pickle
import pandas as pd,numpy as np
from pydantic import BaseModel
from streamlit import json
import json

app = FastAPI()

BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "models", "house_price_model.pkl")
feature_path = os.path.join(BASE_DIR, "..", "artifacts", "feature_columns.json")

with open(feature_path) as f:
    feature_columns = json.load(f)

with open(model_path, "rb") as f:
    model = pickle.load(f)
    
class HouseData(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    lat: float
    long: float
    sqft_living15: int
    sqft_lot15: int

@app.get("/")
def home():
    return {"message": "House Price Prediction API"}

@app.post("/predict")
def predict(data: HouseData):

    input_dict = data.dict()

    df = pd.DataFrame([input_dict])

    df = df.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(df)
    price = np.exp(prediction[0])

    return {"prediction": float(price)}
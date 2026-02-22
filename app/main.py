import os
from fastapi import FastAPI
import pickle
import pandas as pd
import numpy as np
from pydantic import BaseModel
import json
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "models", "house_price_model.pkl")
feature_path = os.path.join(BASE_DIR, "..", "artifacts", "feature_columns.json")

# Load feature columns with error handling
try:
    with open(feature_path) as f:
        feature_columns = json.load(f)
    if not isinstance(feature_columns, list):
        raise ValueError("feature_columns must be a list")
except FileNotFoundError:
    logger.error(f"Feature columns file not found at {feature_path}")
    raise
except json.JSONDecodeError as e:
    logger.error(f"Invalid JSON in feature_columns file at {feature_path}: {e}")
    raise

# Load model with error handling (using pickle which requires trusted sources)
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    logger.error(f"Model file not found at {model_path}")
    raise
except Exception as e:
    logger.error(f"Failed to load model from {model_path}: {e}")
    raise
    
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

    # Validate columns match expected feature set
    missing = set(feature_columns) - set(df.columns)
    extra = set(df.columns) - set(feature_columns)
    
    if missing:
        logger.warning(f"Missing features in input: {missing}. Using fill_value=0 for these columns.")
    if extra:
        logger.warning(f"Extra features in input that won't be used: {extra}")
    
    # Align DataFrame columns with expected feature columns
    df = df.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(df)
    # Inverse transform: expm1 is the inverse of log1p used in training
    price = np.expm1(prediction[0])

    return {"prediction": float(price)}
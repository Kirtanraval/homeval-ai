import pickle
import pandas as pd
import numpy as np

# Load trained model
with open("models/house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

print("\nEnter House Details:\n")

bedrooms = int(input("Bedrooms: "))
bathrooms = float(input("Bathrooms: "))
sqft_living = int(input("Living Area (sqft): "))
grade = int(input("Grade (1-13): "))
yr_built = int(input("Year Built: "))
zipcode = int(input("Zipcode: "))

# Feature engineering
total_area = sqft_living

# Create input sample
data = {
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'sqft_living': [sqft_living],
    'sqft_lot': [5000],
    'floors': [1],
    'waterfront': [0],
    'view': [0],
    'condition': [3],
    'grade': [grade],
    'sqft_above': [sqft_living],
    'sqft_basement': [0],
    'yr_built': [yr_built],
    'yr_renovated': [0],
    'zipcode': [zipcode],
    'lat': [47.51],
    'long': [-122.25],
    'sqft_living15': [sqft_living],
    'sqft_lot15': [5000],
    'total_area': [total_area]
}

sample = pd.DataFrame(data)

# Predict
price_log = model.predict(sample)
price = np.expm1(price_log)[0]

print(f"\nPredicted House Price: ${price:,.0f}")

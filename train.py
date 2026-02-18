from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
import pickle
import json
import os

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/home_data.csv")

df = df.drop(['id', 'date'], axis=1)

# Feature engineering (must match app.py inputs)
df['total_area'] = df['sqft_living'] + df['sqft_basement']

# Features & target
X = df.drop('price', axis=1)
y = np.log1p(df['price'])   # log transform improves stability

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor()
}

print("\n=== Base Model Comparison ===")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"\n{name}")
    print("RMSE:", rmse)
    print("R2:", r2)
    
best_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

best_model.fit(X_train, y_train)


preds_log = best_model.predict(X_test)

preds = np.expm1(preds_log)
y_test_actual = np.expm1(y_test)

rmse = np.sqrt(mean_squared_error(y_test_actual, preds))
r2 = r2_score(y_test_actual, preds)

print("\n=== Final Random Forest Results ===")
print("RMSE:", rmse)
print("R2:", r2)

with open("models/house_price_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\nModel saved → models/house_price_model.pkl")

metrics = {
    "RMSE": float(rmse),
    "R2": float(r2)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f)

print("Metrics saved → metrics.json")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n=== Top 10 Important Features ===")
print(importance_df.head(10))

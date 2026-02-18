from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
import pickle
import json


# Load Data
df = pd.read_csv("data/home_data.csv")

# Drop useless columns
df = df.drop(['id', 'date'], axis=1)

# Feature Engineering
df['total_area'] = df['sqft_living'] + df['sqft_basement']

# Features & Target
X = df.drop('price', axis=1)
y = np.log1p(df['price'])   # log transform for better accuracy

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Base Models Comparison

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

# Hyperparameter Tuning

print("\n=== Tuning Random Forest ===")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Final Evaluation

preds_log = best_model.predict(X_test)

# Convert back to real price
preds = np.expm1(preds_log)
y_test_actual = np.expm1(y_test)

rmse = np.sqrt(mean_squared_error(y_test_actual, preds))
r2 = r2_score(y_test_actual, preds)

print("\n=== Tuned Random Forest Results ===")
print("RMSE:", rmse)
print("R2:", r2)
print("Best Parameters:", grid_search.best_params_)

# Save Model
with open("models/house_price_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\nModel saved as models/house_price_model.pkl")

# Save Metrics
metrics = {
    "RMSE": float(rmse),
    "R2": float(r2)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f)

print("Metrics saved as metrics.json")


# Feature Importance
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n=== Feature Importance ===")
print(importance_df.head(10))

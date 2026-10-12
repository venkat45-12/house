"""Loads the saved model and predicts the price of one house."""
import joblib
import pandas as pd

model = joblib.load("model.joblib")

house = pd.DataFrame([{
    "area": 1800,
    "bedrooms": 3,
    "bathrooms": 2,
    "age": 10,
    "location": "suburb",   # city_center / suburb / outskirts
}])

print(f"Predicted price: {model.predict(house)[0]:,.0f}")

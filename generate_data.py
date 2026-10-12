"""Creates a synthetic housing dataset (houses.csv).
Swap this out for a real dataset (e.g. from Kaggle) once the pipeline works."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 2000

area = rng.integers(500, 4000, n)             # sq ft
bedrooms = rng.integers(1, 6, n)
bathrooms = np.clip(bedrooms - rng.integers(0, 2, n), 1, None)
age = rng.integers(0, 40, n)                  # years old
location = rng.choice(["city_center", "suburb", "outskirts"], n, p=[0.25, 0.45, 0.30])
loc_factor = pd.Series(location).map(
    {"city_center": 1.5, "suburb": 1.1, "outskirts": 0.8}).values

price = (area * 90 + bedrooms * 8000 + bathrooms * 6000 - age * 1200) * loc_factor
price += rng.normal(0, 15000, n)              # noise, like real life

df = pd.DataFrame({"area": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
                   "age": age, "location": location, "price": price.round(0)})
df.to_csv("houses.csv", index=False)
print("Saved houses.csv with", len(df), "rows")

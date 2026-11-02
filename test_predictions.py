import joblib
import pandas as pd

model = joblib.load("model/house_model.pkl")

areas = [800, 1000, 1200, 1400, 1600, 1800, 2000]

test_data = pd.DataFrame({
    "Carpet_Area_sqft": areas,
    "BHK": [3] * len(areas),
    "Bathroom": [3] * len(areas),
    "Furnishing": ["Furnished"] * len(areas),
    "Property_Type": ["Apartment"] * len(areas),
    "Locality": ["Kharadi"] * len(areas)
})

predictions = model.predict(test_data)

results = pd.DataFrame({
    "Carpet Area (sqft)": areas,
    "Predicted Price": predictions
})

print("\nNestEstimate - Prediction Test")
print("--------------------------------")

print(results.to_string(
    index=False,
    formatters={
        "Predicted Price": lambda x: f"₹{x:,.0f}"
    }
))
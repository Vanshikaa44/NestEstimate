from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model/house_model.pkl")

# Localities available in the trained model
LOCALITIES = [
    "Amanora Park Town",
    "Aundh",
    "Balewadi",
    "Baner",
    "Bavdhan",
    "Dehu Road",
    "Dhanori",
    "Hadapsar",
    "Hinjawadi",
    "Kalyani Nagar",
    "Keshav Nagar Mundhwa",
    "Kharadi",
    "Kondhwa",
    "Koregaon Park",
    "Kothrud",
    "Lohegaon",
    "Magarpatta",
    "Mahalunge",
    "Manjri Budruk",
    "Moshi",
    "Mundhwa",
    "Narhe",
    "New Kalyani Nagar",
    "Nibm Annexe Area",
    "Nibm Road",
    "Other",
    "Pashan",
    "Pimpri Chinchwad",
    "Shankar Kalat Nagar",
    "Shivajinagar",
    "Sinhgad Road",
    "Sus",
    "Talegaon Dabhade",
    "Tathawade Pimpri Chinchwad",
    "Undri",
    "Viman Nagar",
    "Wagholi",
    "Wakad",
    "Wanowrie"
]


@app.route("/")
def home():
    return render_template("index.html", localities=LOCALITIES)


@app.route("/predict", methods=["POST"])
def predict():
    area = float(request.form["area"])
    bedrooms = int(request.form["bedrooms"])
    bathrooms = int(request.form["bathrooms"])
    furnishing = request.form["furnishing"]
    property_type = request.form["property_type"]
    locality = request.form["locality"]

    input_data = pd.DataFrame(
        [[area, bedrooms, bathrooms, furnishing,
          property_type, locality]],
        columns=[
            "Carpet_Area_sqft",
            "BHK",
            "Bathroom",
            "Furnishing",
            "Property_Type",
            "Locality"
        ]
    )

    predicted_price = model.predict(input_data)[0]

    return render_template(
        "index.html",
        prediction=(
            f"₹{predicted_price / 10000000:.2f} crore"
            if predicted_price >= 10000000
            else f"₹{predicted_price / 100000:.2f} lakh"
        ),
        localities=LOCALITIES
    )


import os

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
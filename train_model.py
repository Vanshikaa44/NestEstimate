
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor

# 1. Load the Pune housing dataset
data = pd.read_csv("dataset/scrapped_data.csv")

# 2. Convert property prices into rupees
def convert_price(price):
    price = price.strip()

    if "Cr" in price:
        return float(price.replace("Cr", "").strip()) * 10000000
    elif "Lac" in price:
        return float(price.replace("Lac", "").strip()) * 100000
    return None

data["Price"] = data["Price"].apply(convert_price)

# 3. Remove exact duplicate listings
data = data.drop_duplicates()

# 4. Extract carpet area in square feet
data["Carpet_Area_sqft"] = (
    data["Carpet_Area"]
    .str.replace(" sqft", "", regex=False)
)
data["Carpet_Area_sqft"] = pd.to_numeric(
    data["Carpet_Area_sqft"], errors="coerce"
)

# 5. Extract BHK and property type from the title

data["BHK"] = pd.to_numeric(
    data["Title"].str.extract(
        r"(\d+)\s*BHK", expand=False
    ),
    errors="coerce"
)

data["Property_Type"] = data["Title"].str.extract(
    r"\b(Flat|Apartment|House|Villa|Penthouse|Builder Floor)\b",
    flags=2,
    expand=False
).str.title()

# Treat Flat and Apartment as the same category
data["Property_Type"] = data["Property_Type"].replace(
    "Flat", "Apartment"
)

# 6. Extract and clean locality from property titles

location_text = data["Title"].str.extract(
    r"for Sale in (.+)", expand=False
)

location_text = location_text.str.replace(
    r",?\s*Pune\s*$", "", regex=True
)

data["Locality"] = (
    location_text.str.split(",").str[-1].str.strip().str.title()
)

# Standardize common spelling variations
locality_mapping = {
    "Gultekadi": "Gultekdi",
    "Shivaji Nagar": "Shivajinagar",
    "Wanawari": "Wanowrie"
}

data["Locality"] = data["Locality"].replace(locality_mapping)

# Group localities with fewer than 5 listings
locality_counts = data["Locality"].value_counts()

rare_localities = locality_counts[
    locality_counts < 5
].index

data["Locality"] = data["Locality"].replace(
    rare_localities, "Other"
)

# 7. Select features and target

features = [
    "Carpet_Area_sqft",
    "BHK",
    "Bathroom",
    "Furnishing",
    "Property_Type",
    "Locality"
]

target = "Price"

data = data.dropna(subset=features + [target])

X = data[features]
y = data[target]

print("Properties used for training:", len(data))
print("\nLocality counts:")
print(data["Locality"].value_counts().head(20))

print("\nAll locality names:")
print(sorted(data["Locality"].unique()))

# 8. Prepare numerical and categorical features

numerical_features = [
    "Carpet_Area_sqft",
    "BHK",
    "Bathroom"
]

categorical_features = [
    "Furnishing",
    "Property_Type",
    "Locality"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("numbers", "passthrough", numerical_features),
        ("categories", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ]
)

# 9. Create both machine learning pipelines

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=12
        ))
    ]
)

decision_tree_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", DecisionTreeRegressor(
            random_state=42,
            max_depth=12
        ))
    ]
)
# 10 Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 10. Train and evaluate both models

models = {
    "Random Forest": random_forest_model,
    "Decision Tree": decision_tree_model
}

for name, current_model in models.items():
    current_model.fit(X_train, y_train)

    predictions = current_model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"\n{name} Results")
    print("----------------")
    print(f"Mean Absolute Error: Rs. {mae:,.0f}")
    print(f"R2 Score: {r2:.3f}")

# 11. Feature importance analysis

print("\nRandom Forest Feature Importance")
print("--------------------------------")

trained_forest = random_forest_model.named_steps["regressor"]
feature_names = random_forest_model.named_steps[
    "preprocessor"
].get_feature_names_out()

importance = pd.Series(
    trained_forest.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

print(importance)

# 12. Save the selected model

joblib.dump(random_forest_model, "model/house_model.pkl")

print("\nRandom Forest model saved successfully!")
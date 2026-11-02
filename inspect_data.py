import pandas as pd

# Load the new dataset
data = pd.read_csv("dataset/scrapped_data.csv")

# Display dataset information
print("Dataset Shape:")
print(data.shape)

print("\nColumn Names:")
print(data.columns.tolist())

print("\nFirst 5 Rows:")
print(data.head())

print("\nMissing Values:")
print(data.isnull().sum())

print("\nDataset Information:")
print(data.info())

print("\nFirst 5 Property Titles:")
print(data["Title"].head())

print("\nFirst 10 Prices:")
print(data["Price"].head())

def convert_price(price):
    price = price.strip()

    if "Cr" in price:
        return float(price.replace("Cr", "").strip()) * 10000000

    elif "Lac" in price:
        return float(price.replace("Lac", "").strip()) * 100000

    return None


data["Price_in_Rupees"] = data["Price"].apply(convert_price)

print("\nConverted Prices:")
print(data[["Price", "Price_in_Rupees"]].head(10))

print("\nPrice Conversion Check:")
print("Total properties:", len(data))
print("Valid prices:", data["Price_in_Rupees"].notna().sum())
print("Invalid prices:", data["Price_in_Rupees"].isna().sum())

print("\nDuplicate Rows:")
print(data.duplicated().sum())

print("\nDuplicate Property Titles:")
print(data["Title"].duplicated().sum())

data = data.drop_duplicates()

print("\nDataset after removing exact duplicates:")
print("Total properties:", len(data))

print("\nCarpet Area Examples:")
print(data["Carpet_Area"].head(15).to_list())

print("\nBathroom Examples:")
print(data["Bathroom"].head(15).to_list())

print("\nFurnishing Categories:")
print(data["Furnishing"].value_counts(dropna=False))

print("\nCarpet Area Examples:")
print(data["Carpet_Area"].head(15).to_list())

print("\nCarpet Area Missing Values:")
print(data["Carpet_Area"].isna().sum())

data["Carpet_Area_sqft"] = (
    data["Carpet_Area"]
    .str.replace(" sqft", "", regex=False)
    .astype(float)
)

print("\nConverted Carpet Areas:")
print(data[["Carpet_Area", "Carpet_Area_sqft"]].head(15))

print("\nValid Carpet Areas:")
print(data["Carpet_Area_sqft"].notna().sum())

data["BHK"] = data["Title"].str.extract(r"(\d+)\s*BHK").astype(float)

print("\nBHK Counts:")
print(data["BHK"].value_counts(dropna=False).sort_index())

print("\nBHK Examples:")
print(data[["Title", "BHK"]].head(10))

data["Property_Type"] = data["Title"].str.extract(
    r"\b(Flat|Apartment|House|Villa|Penthouse|Builder Floor)\b",
    flags=2
)

print("\nProperty Type Counts:")
print(data["Property_Type"].value_counts(dropna=False))

print("\nProperty Title Keywords:")
print(
    data["Title"]
    .str.extract(r"for Sale in (.+)", expand=False)
    .head(20)
    .to_list()
)

print("\nTitles not identified as Flat, House, or Villa:")
print(
    data.loc[
        data["Property_Type"].isna(),
        "Title"
    ].head(15).to_list()
)

print("\nPrice Statistics:")
print(data["Price_in_Rupees"].describe())

print("\nCarpet Area Statistics:")
print(data["Carpet_Area_sqft"].describe())

print("\nProperties with carpet area above 5000 sqft:")

print(
    data[data["Carpet_Area_sqft"] > 5000][
        ["Title", "Carpet_Area", "Price"]
    ]
)
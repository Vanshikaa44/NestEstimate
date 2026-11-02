
# NestEstimate - House Price Prediction

# About the Project

NestEstimate is a machine learning project that I developed to estimate residential property prices in Pune, Maharashtra.

The main idea behind this project is to help users get an approximate idea of a property's price by entering a few details about it.

The user can enter details like property type, locality, carpet area, number of bedrooms, bathrooms and furnishing status. Based on these details, the trained machine learning model predicts an estimated price.

I have also created a simple website using Flask so that users can interact with the model without having to write any code.

# Why I Chose This Project

I wanted to work on a project where I could understand how machine learning is used to solve a real-world problem.

Property prices depend on several factors, such as location, area and property features. I chose house price prediction because it gave me an opportunity to work with real-world data, learn regression algorithms and build a working web application.

# Features

- Estimates residential property prices in Pune.
- Allows users to select different property types.
- Includes different Pune localities.
- Takes carpet area, BHK, bathrooms and furnishing into consideration.
- Displays the predicted price in Indian Rupees.
- Provides a simple and easy-to-use website.

# Technologies Used

- Python - Main programming language
- Pandas - For handling and cleaning data
- NumPy - For numerical operations
- Scikit-learn - For building and training machine learning models
- Flask - For creating the web application
- Joblib - For saving and loading the trained model
- HTML and CSS - For designing the website

# Dataset

For this project, I used the Pune Housing Prices Dataset | Real Estate Listings by Shubham Sharma.

The dataset contains residential property listings from Pune. It includes details such as property prices, carpet area, furnishing, bathrooms and location.

The original dataset file used in this project is `scrapped_data.csv`.

Since the data comes from property listings, the prices represent advertised asking prices and not necessarily the final prices at which properties were sold.

# Data Cleaning and Preparation

Before training the model, I had to prepare the dataset.

Some of the steps I performed were:

- Converted property prices from lakh and crore into rupees.
- Removed duplicate records.
- Extracted carpet area from the dataset.
- Extracted BHK and property type from property titles.
- Cleaned locality names and standardized some spelling variations.
- Grouped localities with fewer than five listings into an Other category.
- Removed records that had missing values in the features used for training.

# Machine Learning Models

I experimented with two regression algorithms:

1. Random Forest Regressor
2. Decision Tree Regressor

After comparing their results, I selected the Random Forest model for the web application because it performed better on my test dataset.

The model uses the following features:

- Carpet Area
- BHK
- Bathrooms
- Furnishing
- Property Type
- Locality

# Model Results

I evaluated both models using Mean Absolute Error (MAE) and R² Score.

| Evaluation Metric | Random Forest | Decision Tree |
|---|---:|---:|
| MAE | ₹41,52,202 | ₹51,49,346 |
| R² Score | 0.783 | 0.617 |

The Random Forest model performed better than the Decision Tree model on this test split.

The MAE shows the average absolute difference between the predicted and actual listing prices. The R² score indicates how well the model explains variations in the test data. It should not be interpreted as prediction accuracy.

There is still scope to improve the model, especially by using a larger dataset.

# Web Application

I used Flask to connect the trained machine learning model to the website.

Users can enter their property details through a form and submit them. Flask receives the information, passes it to the model and displays the estimated property price on the website.

The application currently focuses only on residential properties in Pune. Commercial properties are not supported.

# Project Folder Structure

```text
NestEstimate/
│
── dataset/
│   └── scrapped_data.csv
│
├── model/
│   └── house_model.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── home.jpg
│
├── app.py
├── train_model.py
├── inspect_data.py
├── test_predictions.py
├── requirements.txt
└── README.md
```

# How to Run the Project

Follow these steps to run the project on your computer.

**1. Create a virtual environment**

```bash
python -m venv venv
```

**2. Activate the virtual environment (Windows)**

```bash
.\venv\Scripts\Activate.ps1
```

**3. Install the required libraries**

```bash
pip install -r requirements.txt
```

**4. Train the machine learning model**

```bash
python train_model.py
```

**5. Start the Flask application**

```bash
python app.py
```

**6. Open the application**

Open your browser and visit:

http://127.0.0.1:5000

You can now enter property details and get an estimated price.

# Future Improvements

There are several things I would like to work on further:

- Use a larger dataset to improve predictions.
- Experiment with other machine learning algorithms.
- Improve the prediction results.
- Use Docker to package the application.
- Deploy the application on Microsoft Azure.
- Implement GitHub Actions for automated testing and deployment.

# Disclaimer

This project is developed for learning and educational purposes. The predicted prices are only estimates based on the available property listing data. They may differ from actual market prices and should not be considered professional property valuations.

# Author

Developed as a machine learning and web development project.

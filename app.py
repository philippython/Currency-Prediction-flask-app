from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
import os

app = Flask(__name__)

# Get the directory of the current script
direct_path = os.path.dirname(os.path.abspath(__file__))

# Specify the relative path to the CSV file
csv_file_path = os.path.join(direct_path, 'data', 'merged_file.csv')
data = pd.read_csv(csv_file_path)

# Selecting relevant features and target variable
X = data[['Crude Oil Price in USD', 'Inflation Rate US', 'Inflation Rate NG']]
y = data['Central Rate']

# Handling missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the SVM model
svm_model = SVR(kernel='rbf', C=100, gamma=1)
svm_model.fit(X_train_scaled, y_train)

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/model')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    crude_oil_price = float(request.form['crude_oil_price'])
    inflation_rate_us = float(request.form['inflation_rate_us'])
    inflation_rate_ng = float(request.form['inflation_rate_ng'])

    # Create DataFrame for the user input
    user_input = pd.DataFrame({
        'Crude Oil Price in USD': [crude_oil_price],
        'Inflation Rate US': [inflation_rate_us],
        'Inflation Rate NG': [inflation_rate_ng]
    })

    # Standardize the user input using the same scaler
    user_input_scaled = scaler.transform(user_input.values)

    # Predict the exchange rate using the trained SVM model
    predicted_rate = svm_model.predict(user_input_scaled)

    # Display the predicted exchange rate
    return render_template('result.html', prediction=round(predicted_rate[0]))

if __name__ == '__main__':
    app.run(debug=True)

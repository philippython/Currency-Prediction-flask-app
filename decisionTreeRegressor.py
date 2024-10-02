import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  # Import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

direct_path =  os.path.dirname(__file__) #getcwd()

# Load dataset
data = pd.read_csv(direct_path + '\\merged_file.csv')


# Selecting relevant features and target variable
# Assuming columns include crude oil price, inflation rate in the US, and Nigeria's inflation rate
X = data[['Crude Oil Price in USD', 'Inflation Rate US', 'Inflation Rate NG']]  # Features
y = data['Central Rate']  # Target (Dollar to Naira Exchange Rate)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#-----------------------------------------------------------------#
# Decision Tree Regressor
#-----------------------------------------------------------------#
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = dt_model.predict(X_test_scaled)

# Optional: Print some predictions
for i in range(10):
    print(f"Actual: {y_test.iloc[i]}, Predicted: {y_pred[i]}")

features = ['Crude Oil Price in USD', 'Inflation Rate US', 'Inflation Rate NG']

# Plot each feature against the dependent variable 'Central Rate'
plt.figure(figsize=(18, 6))

for i, feature in enumerate(features):
    plt.subplot(1, len(features), i + 1)
    plt.scatter(X[feature], y, color='blue')
    plt.title(f'{feature} vs Central Rate')
    plt.xlabel(feature)
    plt.ylabel('Central Rate')

# Input crude oil price
crude_oil_price = float(input("Crude Oil Price in USD (per barell): "))
inflation_rate_us = float(input("Inflation Rate in US (%): "))
inflation_rate_ng = float(input("Inflation Rate in Nigeria (%): "))

# Create a DataFrame for the user input
user_input = pd.DataFrame({
    'Crude Oil Price in USD': [crude_oil_price],
    'Inflation Rate US': [inflation_rate_us],
    'Inflation Rate NG': [inflation_rate_ng]
})

# Standardize the user input using the same scaler used for training
user_input_scaled = scaler.transform(user_input)


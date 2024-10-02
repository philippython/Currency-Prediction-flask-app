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




#---------------------------------------------------------------#
#Linear Regression
#---------------------------------------------------------------#

from sklearn.linear_model import LinearRegression


# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Create and train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = lr_model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print("\nDisplay for Linear Regression")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")

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

# # Scatter Plot: Crude Oil Price vs Central Rate
# plt.figure(figsize=(10, 6))
# plt.scatter(X, y, color='blue')
# plt.title('Crude Oil Price vs Central Rate')
# plt.xlabel('Crude Oil Price in USD')
# plt.ylabel('Central Rate')
# plt.show()

# Prediction vs Actual Plot
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Values', color='blue', linestyle='--')
plt.plot(y_pred, label='Predicted Values', color='red', linestyle=':')
plt.title('Actual vs Predicted Central Rate')
plt.xlabel('Data Points')
plt.ylabel('Central Rate')
plt.legend()
plt.show()

# Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, color='purple')
plt.axhline(y=0, color='black', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()
#--------------------------------------------------------------#



#---------------------------------------------------------------#
# User Input for Prediction
#---------------------------------------------------------------#

print("\nEnter the following values to predict the exchange rate:")

# Input crude oil price
crude_oil_price = float(input("Crude Oil Price in USD(per barrel): "))

# Input inflation rate in the US

# Input inflation rate in Nigeria
inflation_rate_ng = float(input("Inflation Rate in Nigeria (%): "))

# Input previous central rate
# previous_central_rate = float(input("Previous Central Rate (NGN/USD): "))

# Create a DataFrame for the user input
user_input = pd.DataFrame({
    'Crude Oil Price in USD': [crude_oil_price],
    'Inflation Rate US': [inflation_rate_us],
    'Inflation Rate NG': [inflation_rate_ng]
})

# Impute any missing values in the user input (if necessary)
user_input_imputed = imputer.transform(user_input)

# Standardize the user input using the same scaler used for training
user_input_scaled = scaler.transform(user_input_imputed)

# Predict the exchange rate using the trained Linear Regression model
predicted_rate = lr_model.predict(user_input_scaled)

# Display the predicted exchange rate
print(f"\nPredicted Central Rate (Exchange Rate): {predicted_rate[0]:.2f} NGN/USD")

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
data = pd.read_csv(os.path.join(direct_path, 'data', 'merged_file.csv'))


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

from sklearn.ensemble import RandomForestRegressor

#-----------------------------------------------------------------#
# Random Forest Regressor
#-----------------------------------------------------------------#

# Create and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test_scaled)

# Evaluate the model
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mape_rf = np.mean(np.abs((y_test - y_pred_rf) / y_test)) * 100

print("\nResults for Random Forest Regressor:")
print(f"Mean Squared Error (MSE): {mse_rf}")
print(f"Root Mean Squared Error (RMSE): {rmse_rf}")
print(f"Mean Absolute Error (MAE): {mae_rf}")
print(f"Mean Absolute Percentage Error (MAPE): {mape_rf}%")

# Optional: Print some predictions
for i in range(50):
    print(f"Actual: {y_test.iloc[i]}, Predicted: {y_pred_rf[i]}")

# Visualization for Random Forest
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Values', color='blue', linestyle='--')
plt.plot(y_pred_rf, label='Predicted Values', color='red', linestyle=':')
plt.title('Actual vs Predicted Central Rate (Random Forest)')
plt.xlabel('Data Points')
plt.ylabel('Central Rate')
plt.legend()


# Residual Plot for Random Forest
residuals_rf = y_test - y_pred_rf
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_rf, residuals_rf, color='orange')
plt.axhline(y=0, color='black', linestyle='--')
plt.title('Residual Plot (Random Forest)')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()


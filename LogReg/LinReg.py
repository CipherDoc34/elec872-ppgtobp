import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# Paths to the files
combined_file = '../ProcessedDataset/combined_train.p'  # Combined training file
test_file = '../ProcessedDataset/test.p'      # Test file

# Step 1: Load Combined Training Data
with open(combined_file, 'rb') as f:
    combined_data = pickle.load(f)

# Extract training features and targets
X_train = combined_data['X_train'].squeeze(axis=-1)  # Shape: (810,000, 1030)
Y_train = combined_data['Y_train'].mean(axis=-1)      # Use mean ABP as target, Shape: (810,000, 1030)

print(f"Training data loaded: X_train shape = {X_train.shape}, Y_train shape = {Y_train.shape}")

# Step 2: Train Linear Regression Model
# Initialize the linear regression model
model = LinearRegression(n_jobs=-1)

# Train the model
model.fit(X_train, Y_train)
print("Linear regression model training completed.")

# Step 3: Load Test Data
with open(test_file, 'rb') as f:
    test_data = pickle.load(f)

# Extract test features and targets
X_test = test_data['X_test'].squeeze(axis=-1)  # Shape: (num_samples, 1030)
Y_test = test_data['Y_test'].mean(axis=-1)      # Use mean ABP as target, Shape: (num_samples,)

print(f"Test data loaded: X_test shape = {X_test.shape}, Y_test shape = {Y_test.shape}")

# Step 4: Evaluate Model on Test Data
# Predict on the test data
Y_pred = model.predict(X_test)

# Calc metrics
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

# Additional metrics
mae = mean_absolute_error(Y_test, Y_pred)

# BHS Cumulative Error Percentages
errors = abs(Y_test - Y_pred)
within_5 = (errors <= 5).mean() * 100  # % of predictions within 5 mmHg
within_10 = (errors <= 10).mean() * 100  # % of predictions within 10 mmHg
within_15 = (errors <= 15).mean() * 100  # % of predictions within 15 mmHg

# AAMI Metrics
mean_error = (Y_test - Y_pred).mean()
std_error = (Y_test - Y_pred).std()

# Display results
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"BHS: ≤5 mmHg: {within_5:.2f}%, ≤10 mmHg: {within_10:.2f}%, ≤15 mmHg: {within_15:.2f}%")
print(f"AAMI: Mean Error (ME): {mean_error:.2f}, Standard Deviation (SD): {std_error:.2f}")

# Step 5: Save the Trained Model
model_file = 'linear_regression_model.pkl'
joblib.dump(model, model_file)
print(f"Trained model saved to '{model_file}'.")

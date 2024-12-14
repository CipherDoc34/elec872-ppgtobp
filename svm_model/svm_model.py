import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data from a pickle file
with open('../train_set_raw.pkl', 'rb') as f:
    data = pickle.load(f)
    
# print(data)
x = data["x"]
y = data['y']

x = x[:len(x)//200]
y = y[:len(y)//200]

# Ensure x and y are numpy arrays
x = np.array(x)
y = np.array(y)

# Define a function to extract statistical features
def compute_features(data, window_size=5):
    features = []
    half_window = window_size // 2
    for i in range(len(data)):
        # Define the sliding window
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        window = data[start:end]
        
        # Compute features
        mean = np.mean(window)
        std = np.std(window)
        median = np.median(window)
        max_val = np.max(window)
        min_val = np.min(window)
        
        # Append the features for the current point
        features.append([mean, std, median, max_val, min_val])
    return np.array(features)

# Compute statistical features for x
stat_features = compute_features(x, window_size=5)

# Combine x and its statistical features into the feature matrix
x_features = np.hstack((x.reshape(-1, 1), stat_features))

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_features, y, test_size=0.2, random_state=42)

# Initialize and train the SVM regressor
svm = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1, verbose=1, max_iter=1_000_000)
svm.fit(x_train, y_train)

# Make predictions
y_pred_train = svm.predict(x_train)
y_pred_test = svm.predict(x_test)

# Evaluate the model
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

print(f"Training MSE: {train_mse:.4f}")
print(f"Testing MSE: {test_mse:.4f}")
print(f"Testing R^2 Score: {test_r2:.4f}")

# Plot actual vs. predicted values for the test set
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred_test, color='red', label='Predicted Data')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='blue', linestyle='--', label='Perfect Fit')
plt.title('SVM Regression: Actual vs Predicted')
plt.xlabel('Actual y')
plt.ylabel('Predicted y')
plt.legend()
plt.show()

# Plot regression line using a subset of features (if needed for visualization)
x_range = np.linspace(min(x), max(x), 1000).reshape(-1, 1)
x_range_features = compute_features(x_range.ravel(), window_size=5)
x_range_full = np.hstack((x_range, x_range_features))
y_pred_full = svm.predict(x_range_full)

plt.figure(figsize=(12, 6))
plt.scatter(x, y, color='blue', label='Actual Data', alpha=0.5)
plt.plot(x_range, y_pred_full, color='red', label='SVM Regression', linewidth=2)
plt.title('SVM Regression Line with Statistical Features')
plt.xlabel('Input (x)')
plt.ylabel('Output (y)')
plt.legend()
plt.show()

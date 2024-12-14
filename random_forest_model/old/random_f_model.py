import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data from a pickle file
with open('../train_set_raw.pkl', 'rb') as f:
    data_all = pickle.load(f)
    
# print(data)
x = data_all["x"]
y = data_all['y']

x = x[:len(x)//20]
y = y[:len(y)//20]
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
        skewness = np.mean((median - mean) ** 3) / (std ** 3)
        kurtosis = np.mean((median - mean) ** 4) / (std ** 4)
        max_val = np.max(window)
        min_val = np.min(window)
        
        # Append the features for the current point
        features.append([mean, std, median, skewness, kurtosis, max_val, min_val])
    features = np.array(features)
    return np.nan_to_num(features)

# Compute statistical features for x
stat_features = compute_features(x, window_size=5)

# Combine x and its statistical features into the feature matrix
x_features = np.hstack((x.reshape(-1, 1), stat_features))

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_features, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor
print("training....")
rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, verbose=1)  # Adjust hyperparameters as needed
rf.fit(x_train, y_train)

# Make predictions
y_pred_train = rf.predict(x_train)
y_pred_test = rf.predict(x_test)

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
plt.title('Random Forest Regression: Actual vs Predicted')
plt.xlabel('Actual y')
plt.ylabel('Predicted y')
plt.legend()
plt.savefig("fig1.png")


# Plot regression line using a subset of features (if needed for visualization)
x_range = np.linspace(min(x), max(x), 1000).reshape(-1, 1)
x_range_features = compute_features(x_range.ravel(), window_size=5)
x_range_full = np.hstack((x_range, x_range_features))
y_pred_full = rf.predict(x_range_full)

plt.figure(figsize=(12, 6))
plt.scatter(x, y, color='blue', label='Actual Data', alpha=0.5)
plt.plot(x_range, y_pred_full, color='red', label='Random Forest Regression', linewidth=2)
plt.title('Random Forest Regression Line with Statistical Features')
plt.xlabel('Input (x)')
plt.ylabel('Output (y)')
plt.legend()
plt.savefig("fig2.png")


one_sample = x_features[:1024, :]
gt_sample = y[:1024]
y_pred = rf.predict(one_sample)

output = dict(ppg=one_sample, gt=gt_sample, output=y_pred, model=rf)
pickle.dump(output, open("rf_output.pkl", "wb+"))

plt.figure(figsize=(12, 12))

fig, ax = plt.subplots(3, 1)
ax[0].plot(gt_sample, label='ground truth')
ax[0].title.set_text('Ground Truth BP')

ax[1].plot(one_sample[:,0], label='ppg')
ax[1].title.set_text('PPG')

ax[2].plot(y_pred, label='output BP')
ax[2].title.set_text('Output BP')

fig.tight_layout()
fig.savefig("singesample.png")

plt.figure(figsize=(24, 12))
plt.plot(gt_sample, color='blue', label='Ground Truth BP', linewidth=2)
plt.title('Ground Truth BP')
plt.savefig("singesamplegt.png")
plt.close()

plt.figure(figsize=(24, 12))
plt.plot(one_sample[:, 0], color='orange', label='PPG', linewidth=2)
plt.title('PPG')
plt.savefig("singesamplePPG.png")
plt.close()

plt.figure(figsize=(24, 12))
plt.plot(y_pred, color='red', label='Output BP', linewidth=2)
plt.title('Output BP')
plt.savefig("singesampleoutbp.png")
plt.close()

plt.show()
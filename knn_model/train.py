import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def compute_features(data, window_size=5):
    features = []
    half_window = window_size // 2
    for i in range(len(data)):
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

# Load data from a pickle file
with open('../train_set_raw.pkl', 'rb') as f:
    data = pickle.load(f)
    
# print(data)
x = data["x"]
y = data['y']

x = x[:len(x)//200]
y = y[:len(y)//200]

x = np.array(x).reshape(-1, 1)
y = np.array(y)

# print(len(x), len(y))

x = np.hstack((x, compute_features(x)))

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)

# Initialize and train the KNN regressor
n_neighbors = 11  # Number of neighbors (adjust as needed)
knn = KNeighborsRegressor(n_neighbors=n_neighbors)
print("training")
knn.fit(x_train, y_train)

# Make predictions
y_pred_train = knn.predict(x_train)
y_pred_test = knn.predict(x_test)

# Evaluate the model
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

print(f"Training MSE: {train_mse:.4f}")
print(f"Testing MSE: {test_mse:.4f}")
print(f"Testing R^2 Score: {test_r2:.4f}")

print(len(x_test), len(y_test))

# Plot actual vs. predicted values for the test set
plt.figure(figsize=(12, 6))
plt.scatter(x_test[:,0], y_test, color='blue', label='Actual Data')
plt.scatter(x_test[:,0], y_pred_test, color='red', label='Predicted Data', alpha=0.7)
plt.title('KNN Regression: Actual vs Predicted')
plt.xlabel('Input (x)')
plt.ylabel('Output (y)')
plt.legend()
plt.savefig("knn_avp200.png")
plt.close()

# Plot the full regression line for better visualization
x_range = np.linspace(min(x[:,0]), max(x[:,0]), 1000).reshape(-1, 1)
x_range = np.hstack((x_range, compute_features(x_range)))
y_pred_full = knn.predict(x_range)

plt.figure(figsize=(12, 6))
plt.scatter(x[:,0], y, color='blue', label='Actual Data', alpha=0.5)
plt.plot(x_range[:,0], y_pred_full, color='red', label='KNN Regression', linewidth=2)
plt.title('KNN Regression Line')
plt.xlabel('Input (x)')
plt.ylabel('Output (y)')
plt.legend()
plt.savefig("knn_regression_line200.png")
# plt.show()
plt.close()

one_sample = x[:1024, :]
gt_sample = y[:1024]
y_pred = knn.predict(one_sample)

output = dict(ppg=one_sample, gt=gt_sample, output=y_pred, model=knn)
pickle.dump(output, open("knn_output.pkl", "wb+"))

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



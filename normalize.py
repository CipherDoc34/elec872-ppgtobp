import pickle
import numpy as np

# Input and output paths
input_path = 'data.pkl'  # Path to the original pickle file
output_path = 'processed_data.pkl'  # Path to save the processed file

# Open and load data from the pickle file
with open(input_path, 'rb') as f:
    data = pickle.load(f)

# Extract raw data
X_train = np.array(data['X_train'])  # Replaced with test for test file
Y_train = np.array(data['Y_train'])  # Replaced with test for tets file

# Compute normalization parameters
max_ppg = np.max(X_train)
min_ppg = np.min(X_train)
max_abp = np.max(Y_train)
min_abp = np.min(Y_train)

# Normalize the data
X_train = (X_train - min_ppg) / (max_ppg - min_ppg)
Y_train = (Y_train - min_abp) / (max_abp - min_abp)

# Replace the original data in the dictionary with the normalized data
data['X_train'] = X_train
data['Y_train'] = Y_train

# Save the normalized data back to a new pickle file
with open(output_path, 'wb') as f:
    pickle.dump(data, f)

print(f"Data normalized and saved to {output_path}.")
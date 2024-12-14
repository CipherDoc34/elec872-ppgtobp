import numpy as np

def augment_samples_with_features(data):
    """
    Augment each sample with statistical features.

    Args:
        data (numpy.ndarray): Input data of shape (num_samples, 1024, 1).

    Returns:
        numpy.ndarray: Augmented data with shape (num_samples, 1030, 1).
    """
    # Remove the last axis to make data shape (num_samples, 1024)
    data = data.squeeze(axis=-1)

    # Calculate features
    mean = np.mean(data, axis=1)
    std = np.std(data, axis=1)
    skewness = np.mean((data - mean[:, None]) ** 3, axis=1) / (std ** 3)
    kurtosis = np.mean((data - mean[:, None]) ** 4, axis=1) / (std ** 4)
    minimum = np.min(data, axis=1)
    maximum = np.max(data, axis=1)

    # Combine features into an array
    features = np.stack([mean, std, skewness, kurtosis, minimum, maximum], axis=1)

    # Append features to each sample
    augmented_data = np.concatenate([data, features], axis=1)  # Shape: (num_samples, 1030)

    # Add the last axis back to match original format
    return augmented_data[:, :, None]  # Shape: (num_samples, 1030, 1)


#############################################################################################################

import pickle

def process_file(input_path, output_path):
    """
    Process a single pickle file to extract features and save them.

    Args:
        input_path (str): Path to the original pickle file.
        output_path (str): Path to save the processed pickle file.
    """
    # Load the original pickle file - f is just a variable name used to represent the file object created by open(...)
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    # Extract features for X_train - Will calculate features for each of the 90,000 samples.
    X_augmented = augment_samples_with_features(data['X_train'])
    Y_augmented = augment_samples_with_features(data['Y_train'])

    # Prepare the processed data
    processed_data = {
        'X_train': X_augmented,  # Augmented samples with features
        'Y_train': Y_augmented  # Original target data
    }

    # Save the processed data to a new pickle file
    with open(output_path, 'wb') as f:
        pickle.dump(processed_data, f)

    print(f"Processed file saved to {output_path}")

#############################################################################################################

# Directory paths
input_dir = 'MIMICDataset/'  # Adjust to your dataset path
output_dir = 'ProcessedDataset/'  # Directory to save processed files

# Ensure output directory exists
import os
os.makedirs(output_dir, exist_ok=True)

# Process all training files
for fold_id in range(10):
    input_file = os.path.join(input_dir, f'train{fold_id}.p')
    output_file = os.path.join(output_dir, f'processed_train{fold_id}.p')
    process_file(input_file, output_file)



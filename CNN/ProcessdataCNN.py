import numpy as np
import pickle
from scipy.signal import butter, sosfilt, welch
import os

# Directory paths
input_dir = "../MIMICDataset/"
output_dir = "../ProcessedDatasetCNN/"
os.makedirs(output_dir, exist_ok=True)

# Statistical Features Function
def extract_statistical_features(data):
    data = data.squeeze(axis=-1)
    mean = np.mean(data, axis=1)
    std = np.std(data, axis=1)
    skewness = np.mean((data - mean[:, None]) ** 3, axis=1) / (std ** 3)
    kurtosis = np.mean((data - mean[:, None]) ** 4, axis=1) / (std ** 4)
    minimum = np.min(data, axis=1)
    maximum = np.max(data, axis=1)
    features = np.stack([mean, std, skewness, kurtosis, minimum, maximum], axis=1)
    return features

# Bandpass Filter Function
def apply_bandpass_filter(data, lowcut=0.5, highcut=15.0, fs=125):
    """
    Apply a bandpass filter to the data.
    Args:
        data (numpy.ndarray): Input data of shape (num_samples, 1024, 1).
        lowcut (float): Lower cutoff frequency.
        highcut (float): Upper cutoff frequency.
        fs (int): Sampling frequency.
    Returns:
        numpy.ndarray: returns the filtered version of the input signal, where frequencies outside the specified range 
        (lowcut to highcut) are attenuated
    """
    sos = butter(4, [lowcut, highcut], btype='band', fs=fs, output='sos')
    data = data.squeeze(axis=-1)
    filtered_data = np.array([sosfilt(sos, sample) for sample in data])
    return filtered_data[:, :, None]

# Frequency Response (PSD) Function
def extract_psd_features(data, fs=125):
    """
    Extract PSD features from the data.
    Args:
        data (numpy.ndarray): Input data of shape (num_samples, 1024, 1).
        fs (int): Sampling frequency.
    Returns:
        numpy.ndarray: PSD features of shape (num_samples, num_psd_features).
    """
    data = data.squeeze(axis=-1)
    psd_features = []
    for sample in data:
        # the 1024-sample signal is divided into overlapping sections of size 256 samples each for PSD estimation.
        # By default, Welch’s method applies a 50% overlap
        f, Pxx = welch(sample, fs=fs, nperseg=256)
        psd_features.append(Pxx)
    return np.array(psd_features)

# Variance and Pearson Correlation Function
def select_features_train(data, threshold=0.95, return_indices=False):
    """
    Select features based on variance and Pearson correlation.
    Args:
        data (numpy.ndarray): Input data of shape (num_samples, num_features).
        threshold (float): Correlation threshold for feature removal.
    Returns:
        numpy.ndarray: Reduced feature set.
    """
    # Variance thresholding (remove near-zero variance features)
    # The variance thresholding step evaluates each feature (column) across all samples (rows) in the dataset
    variances = np.var(data, axis=0)
    high_variance_idx = np.where(variances > 1e-6)[0]
    # Keep all rows but only specific columns
    data = data[:, high_variance_idx]

    # Calculates the Pearson correlation matrix for all features (columns) in the dataset.
    corr_matrix = np.corrcoef(data, rowvar=False)
    correlated_features = set()
    # goes over rows
    for i in range(corr_matrix.shape[0]):
        # goes over columns
        for j in range(i + 1, corr_matrix.shape[1]):
            # If the absolute correlation value |corr_matrix[i, j]| exceeds the threshold, feature j is added to the correlated_features set
            if abs(corr_matrix[i, j]) > threshold:
                correlated_features.add(j)

    # Keeps the columns of features that arent correlated
    uncorrelated_features = [i for i in range(data.shape[1]) if i not in correlated_features]
    retained_indices = high_variance_idx[uncorrelated_features]

    return data[:, uncorrelated_features], retained_indices

def select_features_test(data, retained_indices):
    """
    Apply feature selection to test data using retained indices from training.
    """
    return data[:, retained_indices]


# Updated process_file function to concatenate features with each sample
def process_file_train(input_path, output_path):
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    # Extract original data
    X_train = data['X_train']

    # Step 1: Extract Statistical Features
    X_stats = extract_statistical_features(X_train)  # Shape: (num_samples, num_stat_features)

    # Step 2: Apply Bandpass Filter (for cleaner PSD extraction)
    X_filtered = apply_bandpass_filter(X_train)  # Shape: (num_samples, 1024, 1)

    # Step 3: Extract PSD Features
    X_psd = extract_psd_features(X_filtered)  # Shape: (num_samples, num_psd_features)

    # Step 4: Combine Original Data, Statistical Features, and PSD Features
    X_combined1 = X_train.squeeze(axis=-1)  # Shape: (num_samples, 1024)

    X_combined2 = np.concatenate([X_stats, X_psd], axis=1)  # Augmented shape

    # Step 5: Optional - Apply Feature Selection (Variance and Correlation)
    # Feature selection
    X_selected, retained_indices = select_features_train(X_combined2)

    X_selected2 = np.concatenate([X_combined1, X_selected], axis=1)

    # Prepare the processed data dictionary
    processed_data = {
        'X_train': X_selected2[:, :, None],  # Add back last axis for consistency
        'Y_train': data['Y_train']
    }

    # Save the processed file
    with open(output_path, 'wb') as f:
        pickle.dump(processed_data, f)

    print(f"Processed file saved to {output_path}")

    return retained_indices

def process_file_test(input_path, output_path, retained_indices):
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    # Extract original data
    X_test = data['X_test']

    # Step 1: Extract Statistical Features
    X_stats = extract_statistical_features(X_test)

    # Step 2: Apply Bandpass Filter (for cleaner PSD extraction)
    X_filtered = apply_bandpass_filter(X_test)

    # Step 3: Extract PSD Features
    X_psd = extract_psd_features(X_filtered)

    # Step 4: Combine Original Data, Statistical Features, and PSD Features
    X_combined1 = X_test.squeeze(axis=-1)
    X_combined2 = np.concatenate([X_stats, X_psd], axis=1)

    # Step 5: Use Retained Indices from Training
    X_selected = select_features_test(X_combined2, retained_indices)

    X_selected2 = np.concatenate([X_combined1, X_selected], axis=1)

    # Prepare the processed data dictionary
    processed_data = {
        'X_test': X_selected2[:, :, None],
        'Y_test': data['Y_test']
    }

    # Save the processed file
    with open(output_path, 'wb') as f:
        pickle.dump(processed_data, f)
    print(f"Processed file saved to {output_path}")


if __name__ == "__main__":
    # Process training files and get retained indices
    retained_indices = None
    for fold_id in range(10):
        input_path = f"../MIMICDataset/train{fold_id}.p"
        output_path = f"../ProcessedDatasetCNN/processed_train{fold_id}.p"
        if retained_indices is None:
            retained_indices = process_file_train(input_path, output_path)
        else:
            process_file_train(input_path, output_path)

    # Process test file using retained indices
    test_input_path = "../MIMICDataset/test.p"
    test_output_path = "../ProcessedDatasetCNN/processed_test.p"
    process_file_test(test_input_path, test_output_path, retained_indices)


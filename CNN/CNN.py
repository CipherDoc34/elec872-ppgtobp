from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Conv2D, MaxPooling2D, concatenate
from tensorflow.keras.optimizers import Adam
import pickle
import os
import numpy as np
import tensorflow as tf

# Print GPU devices
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# Check CUDA support
print("Built with CUDA:", tf.test.is_built_with_cuda())
# Print TensorFlow version
print("TensorFlow Version:", tf.__version__)


def cnn(input_shape):
    input = Input(shape=input_shape, name="Input")
    x1 = Conv1D(filters=32, kernel_size=3, activation='relu')(input)
    x1 = MaxPooling1D(pool_size=2)(x1)
    x1 = Conv1D(filters=64, kernel_size=3, activation='relu')(x1)
    x1 = MaxPooling1D(pool_size=2)(x1)
    x1 = Flatten()(x1)

    # Dense layers after merging
    x = Dense(128, activation='relu')(x1)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='linear', name="Output")(x)

    # Model is a class from Keras used to define the structure of a neural network.
    # It connects the input(s) and output(s) of your network, effectively building the computational graph
    model = Model(inputs=input, outputs=output)
    return model

###################################################################################################################################

# 1. K-Fold Cross-Validation: Evaluate generalization 

# Performance tracking
fold_metrics = []

# Paths to processed files
data_dir = "../ProcessedDatasetCNN/"
# Creates a list of file paths: - will have path to each train file in a list
train_files = [os.path.join(data_dir, f"processed_train{fold_id}.p") for fold_id in range(10)]


'''The enumerate function generates two outputs for each iteration:

Index:

This is the numeric index of the current item in the iterable.
The index starts from 0 by default and increments by 1 for each subsequent item.

Value:

This is the actual value (or element) of the iterable at the current index '''

# Ex: First iter = 0, processedtrain0.p

for fold, val_file in enumerate(train_files):
    print(f"Starting fold {fold + 1}/{len(train_files)}")

    # Load validation data - val_file starts at processedtrain0.py
    with open(val_file, 'rb') as f:
        val_data = pickle.load(f)
    X_val, Y_val = val_data['X_train'], val_data['Y_train']

    # Load training data (all files except the validation file)
    X_train_list, Y_train_list = [], []
    for train_file in train_files:
        if train_file != val_file:
            with open(train_file, 'rb') as f:
                train_data = pickle.load(f)
            X_train_list.append(train_data['X_train'])
            Y_train_list.append(train_data['Y_train'])

    # Combine training data
    X_train = np.concatenate(X_train_list, axis=0)
    Y_train = np.concatenate(Y_train_list, axis=0)

    # Build and compile the model
    input_shape = (X_train.shape[1], X_train.shape[2])  # -> (1037, 1) Second and third column
    
    model = cnn(input_shape)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae', 'mse'])

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    history = model.fit(
        X_train, Y_train,
        epochs=2,
        batch_size=16,
        validation_data=(X_val, Y_val),
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate the model on the validation set
    val_loss, val_mae, val_mse = model.evaluate(X_val, Y_val, verbose=0)
    print(f"Fold {fold + 1} - Validation Loss: {val_loss}, MAE: {val_mae}, MSE: {val_mse}")

    # Store metrics
    fold_metrics.append({'fold': fold + 1, 'loss': val_loss, 'mae': val_mae, 'mse': val_mse})

##########################################################################

# 2. Final Model Training: Train on full dataset with validation split for monitoring

# File paths for the combined datasets
train_file = '../ProcessedDatasetCNN/combined_train.p'
test_file = '../ProcessedDatasetCNN/processed_test.p'

# Load combined training data
with open(train_file, 'rb') as f:
    train_data = pickle.load(f)
    X_train = train_data['X_train']  # Shape: (810000, 1037, 1)
    Y_train = train_data['Y_train']  # Shape: (810000, 1024, 1)

# Load combined test data
with open(test_file, 'rb') as f:
    test_data = pickle.load(f)
    X_test = test_data['X_test']  # Shape: (num_test_samples, 1037, 1)
    Y_test = test_data['Y_test']  # Shape: (num_test_samples, 1024, 1)

# Define input shape
input_shape = (X_train.shape[1], X_train.shape[2])  # (1037, 1)

# The model's architecture is defined based on the shape of each individual sample, not the entire dataset.
# the number of samples (batch_size) is flexible and handled dynamically during training
cnn_model = cnn(input_shape)

# Compile the model
cnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mse'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = cnn_model.fit(
    X_train, Y_train,
    # Number of times to go through dataset
    epochs=50,
    # Will process 16 samples at a time of size input_shape
    batch_size=16,
    validation_split=0.1,  # Use 10% of training data for validation
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate on the test set
test_loss, test_mse = cnn_model.evaluate(X_test, Y_test, verbose=1)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test MSE: {test_mse:.4f}")

# Save the trained model
model_filename = "final_cnn_model.h5"
cnn_model.save(model_filename)
print(f"Model saved to {model_filename}")






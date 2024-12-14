import time
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, periodogram
import pandas as pd


# Filtering functions
def low_pass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def high_pass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

dt = pickle.load(open(os.path.join('data','train0.p'),'rb'))

print(dt["X_train"][0].shape)
print(type(dt["X_train"]))

x = []
for data in dt["X_train"]:
    x.extend(data)
x = np.array(x)
# print(x.shape)

# # exit()


dt = pickle.load(open(r"../data\meta.p", 'rb'))	
max_ppg = dt['max_ppg']
min_ppg = dt['min_ppg']
max_abp = dt['max_abp']
min_abp = dt['min_abp']

x = pickle.load(open("test_set_raw.pkl",'rb'))

y = x["y"][:2000] * max_abp + min_abp
x = x["x"][:2000] * max_abp + min_abp


print(x.shape)
print(x)

# Load data from a pickle file
# with open('data.pkl', 'rb') as f:
#     x, y = pickle.load(f)

# Plot the raw data
plt.figure(figsize=(12, 6))
plt.plot(x, label='Raw Data')
plt.title('Raw Data')
plt.legend()
plt.show()

# Apply filters
fs = 100  # Sampling frequency (adjust as needed)
low_filtered = low_pass_filter(x, cutoff=5, fs=fs)
high_filtered = high_pass_filter(x, cutoff=5, fs=fs)
moving_avg = moving_average(x, window_size=10)

# Plot filtered data
plt.figure(figsize=(12, 8))

fig, ax = plt.subplots(4, 1)
ax[0].plot(x, label='Raw Data')
ax[0].title.set_text('Raw Data')

ax[1].plot(low_filtered, label='Low-Pass Filter')
ax[1].title.set_text('Low-Pass Filter')

ax[2].plot(high_filtered, label='High-Pass Filter')
ax[2].title.set_text('High-Pass Filter')

ax[3].plot(moving_avg, label='Moving Average')
ax[3].title.set_text('Moving Average')
fig.tight_layout()
fig.show()

# Perform PSD analysis
frequencies, power = periodogram(x, fs=fs)

plt.figure(figsize=(12, 6))
plt.semilogy(frequencies, power)
plt.title('Power Spectral Density (PSD)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.grid()
plt.show()

# Peak and valley detection
peaks, _ = find_peaks(x, height=0)
valleys, _ = find_peaks(-x)

print(peaks, valleys)

plt.figure(figsize=(12, 6))
plt.plot(x, label='Raw Data')
plt.plot(peaks, x[peaks], 'r^', label='Peaks')
plt.plot(valleys, x[valleys], 'bv', label='Valleys')
plt.title('Peak and Valley Detection')
plt.legend()
plt.show()

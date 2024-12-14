import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from scipy.signal import find_peaks

def rolling_window(data, fn, window_size=100):
    ret = []
    peaks_ = []
    half_window = window_size // 2
    for i in range(len(data)):
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        window = data[start:end]
        peaks, _ = find_peaks(window)

        # print(f"start {start}, end: {end}, window: {end-start}")
        peaks_.append(peaks)
        ret.append(fn(window[peaks]))
    return ret, peaks

def sys_dis(data, name, graph=True):
    peaks, _ = find_peaks(data)

    sys, sys_peaks = rolling_window(data, max)
    dis, dis_peaks = rolling_window(data, min)
    # print(len(data))

    # print(len(sys))
    # print(len(dis))

    if graph:
        plt.figure(figsize=(12, 8))

        fig, ax = plt.subplots(2, 1)
        ax[0].plot(data, label='Data')
        ax[0].title.set_text('Data')

        ax[1].plot(data)
        ax[1].plot(peaks, data[peaks],'r^', label='Data Peaks')
        # ax[1].plot(dis_peaks, data[dis_peaks],'g^', label='Data Peaks')
        ax[1].title.set_text('Data Peaks')
        fig.tight_layout()
        # fig.show()
        fig.savefig(f"{name}.png")

    return dict(mean_sys=np.mean(sys), mean_dis=np.mean(dis), sys=sys, dis=dis)

with open('rf_output.pkl', 'rb') as f:
    data = pickle.load(f)

# dict(ppg=one_sample, gt=gt_sample, output=y_pred, model=knn)
dt = pickle.load(open(r"C:\git\elec872-ppgtobp\PPG2ABP\codes\data\meta9.p", 'rb'))			# loading metadata
max_ppg = dt['max_ppg']
min_ppg = dt['min_ppg']
max_abp = dt['max_abp']
min_abp = dt['min_abp']
ppg : np.ndarray
gt : np.ndarray
output : np.ndarray

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
with open('../test_set_raw.pkl', 'rb') as f:
    test = pickle.load(f)
    
# print(data)
ppg = test["x"]
gt = test['y']

ppg = ppg[:1024]
gt = gt[:1024]

ppg = np.array(ppg).reshape(-1, 1)
gt = np.array(gt)

# print(len(x), len(gt))

ppg = np.hstack((ppg, compute_features(ppg)))


rf : RandomForestRegressor
rf = data["model"]

output = rf.predict(ppg)

ppg = data["ppg"] * max_ppg + min_ppg
gt = data["gt"] * max_abp + min_abp

output = output * max_abp + min_abp

print("gt")
gt_sys_dis = sys_dis(gt, "GT")
print(f"gt: sys {gt_sys_dis['mean_sys']}, dis {gt_sys_dis['mean_dis']}")

print("output")
output_sys_dis = sys_dis(output, "Output")
print(f"output: sys {output_sys_dis['mean_sys']}, dis {output_sys_dis['mean_dis']}")

print(len(gt_sys_dis['sys']), len(output_sys_dis['sys']))
print(len(gt_sys_dis['dis']), len(output_sys_dis['dis']))

print(f"sys mae: {mean_squared_error(gt_sys_dis['sys'], output_sys_dis['sys'])}")

print(f"dis mae: {mean_squared_error(gt_sys_dis['dis'], output_sys_dis['dis'])}")

print(f"mean error: sys {np.mean(np.array(output_sys_dis['sys']) - np.array(gt_sys_dis['sys']))} dis {np.mean(np.array(output_sys_dis['dis']) - np.array(gt_sys_dis['dis']))}")

plt.figure(figsize=(12, 12))

fig, ax = plt.subplots(3, 1)
ax[0].plot(gt, label='ground truth')
ax[0].title.set_text('Ground Truth BP')

ax[1].plot(ppg[:,0], label='ppg')
ax[1].title.set_text('PPG')

ax[2].plot(output, label='output BP')
ax[2].title.set_text('Output BP')

fig.tight_layout()
# plt.show()
fig.savefig("singesample.png")
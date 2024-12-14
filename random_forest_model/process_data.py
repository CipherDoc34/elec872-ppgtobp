import os
import pickle
import numpy as np

dt = pickle.load(open(os.path.join('../data','test.p'),'rb'))

x = []
y = []
for datax, datay in zip(dt["X_test"], dt["Y_test"]):
    x.extend([d[-1] for d in datax])
    y.extend([d[-1] for d in datay])

x = np.array(x)
y = np.array(y)
print(x.shape)
print(y.shape)
data = dict(x=x, y=y)
pickle.dump( data , open("test_set_raw.pkl", "wb+"))

dt = pickle.load(open(os.path.join('../data','train.p'),'rb'))

x = []
y = []
for datax, datay in zip(dt["X_train"], dt["Y_train"]):
    x.extend([d[-1] for d in datax])
    y.extend([d[-1] for d in datay])

x = np.array(x)
y = np.array(y)
print(x.shape)
print(y.shape)
data = dict(x=x, y=y)
pickle.dump( data , open("train_set_raw.pkl", "wb+"))
import h5py
import os
import pickle
import numpy as np
from tqdm import tqdm


fs = 125 # sampling frequency
t = 10 # length of ppg episodes
dt = 5 # step size of taking the next episode

samples_in_episode = round(fs * t)
d_samples = round(fs * dt)

os.makedirs('processed_data')

f = h5py.File(os.path.join('raw_data','Part_1.mat'), 'r')

ky = 'Part_1'

for i in tqdm(range(len(f[ky])),desc='Reading Records'):

    signal = []
    bp = []

    output_str = '10s,SBP,DBP\n'

    for j in tqdm(range(len(f[f[ky][i][0]])),desc='Reading Samples from Record {}/3000'.format(i+1)):
        
        signal.append(f[f[ky][i][0]][j][0])
        bp.append(f[f[ky][i][0]][j][1])

    for j in tqdm(range(0,len(f[f[ky][i][0]])-samples_in_episode, d_samples),desc='Processing Episodes from Record {}/3000'.format(i+1)):
        
        sbp = max(bp[j:j+samples_in_episode])
        dbp = min(bp[j:j+samples_in_episode])

        output_str += '{},{},{}\n'.format(j,sbp,dbp)


    fp = open(os.path.join('processed_data','Part_1_{}.csv'.format(i)),'w')
    fp.write(output_str)
    fp.close()

    f = h5py.File('./raw_data/Part_1.mat', 'r')

    candidates = pickle.load(open('./candidates.p', 'rb'))
    samples_in_episode = round(fs * t)
    ky = 'Part_1'

    for indix in tqdm(range(len(candidates)), desc='Reading from File 1/4'):

        if(candidates[indix][0] != 1):
            continue

        record_no = int(candidates[indix][1])
        episode_st = int(candidates[indix][2])

        ppg = []
        abp = []

        for j in tqdm(range(episode_st, episode_st+samples_in_episode), desc='Reading Episode Id {}'.format(indix)):    

            ppg.append(f[f[ky][record_no][0]][j][0])
            abp.append(f[f[ky][record_no][0]][j][1])

        pickle.dump(np.array(ppg), open(os.path.join('ppgs', '{}.p'.format(indix)), 'wb'))
        pickle.dump(np.array(abp), open(os.path.join('abps', '{}.p'.format(indix)), 'wb'))


os.makedirs('data')

files = next(os.walk('abps'))[2]

np.random.shuffle(files)
data = []

for fl in tqdm(files):

    abp = pickle.load(open(os.path.join('abps',fl),'rb'))
    ppg = pickle.load(open(os.path.join('ppgs',fl),'rb'))

    data.append([abp, ppg])



f = h5py.File(os.path.join('data','data.hdf5'), 'w')

import numpy as np
from PyEMD import EMD

def emd_augment(data, sequence_length, n_IMF = 500):
    n_imf, channel_num = n_IMF, data.shape[1]
    emd_data = np.zeros((n_imf,data.shape[0],channel_num))
    max_imf = 0
    for ci in range(channel_num):
        s = data[:, ci]
        IMF = EMD().emd(s)
        r_s = np.zeros((n_imf, data.shape[0]))
        if len(IMF) > max_imf:
            max_imf = len(IMF)
        for i in range(len(IMF)):
            r_s[i] = IMF[len(IMF)-1-i] 
        if(len(IMF)==0): r_s[0] = s
        emd_data[:,:,ci] = r_s
    if max_imf < n_imf:
        emd_data = emd_data[:max_imf,:,:]
    train_data_new = np.zeros((len(data)-sequence_length+1,max_imf,sequence_length,channel_num))
    for i in range(len(data)-sequence_length+1):
        train_data_new[i] = emd_data[:,i:i+sequence_length,:]
    return train_data_new

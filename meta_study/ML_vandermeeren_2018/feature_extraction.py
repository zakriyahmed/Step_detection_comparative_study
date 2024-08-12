import os

import numpy as np
from scipy.signal import find_peaks # type: ignore
from scipy.stats import kurtosis, skew # type: ignore
from scipy.fft import fft # type: ignore
import torch 
import pandas as pd # type: ignore
import datetime


from dataloader import Dataloader


def standarize_feaures(feature_matrix):
    files = os.listdir()
    if '1mean.npy' in files and '1std.npy' in files:
        
        mean = np.load('mean.npy')
        std = np.load('std.npy')
    else:
        mean = np.mean(feature_matrix,axis=0)
        std = np.std(feature_matrix,axis=0)
        mean = np.nan_to_num(mean,nan=1e-8)
        std = np.nan_to_num(std,nan=1e-8)
        mean = np.where(mean==0,1e-8,mean)
        std = np.where(std==0,1e-8,std)

        #np.save('mean.npy',mean)
        #np.save('std.npy',std)
    #print(std)
    feature_matrix = (feature_matrix-mean)/std
    return feature_matrix


def all_features(windows,labels):
    all_features = []
    step_count = []
    for i in range(windows.shape[0]):
        features = calculate_features(windows[i])
        all_features.append(features)
        #print(torch.count_nonzero(labels[i]))
        step_count.append(torch.count_nonzero(labels[i,:,0]).item())

    all_features = standarize_feaures(np.array(all_features,dtype=np.float64))
    return all_features,step_count



def calculate_features(windows):
    """     # 1 --> minimum value, 
            # 2 --> maximun value, 
            # 3 --> mean, 
            # 4 --> variance, 
            # 5 --> energy, 
            # 6 --> mean absolute deviation, 
            # 7 --> number of samples above th, 
            # 8 --> number of peaks, 
            # 9 --> mean value of peaks, 
            # 10 --> std value of peaks, 
            # 11 --> frequency at maximum of FFT, 
            # 12 --> maximum of FFT, 
            # 13 --> std of FFT, 
            # 14 --> kurtosis of FFT, 
            # 15 --> skewness of FFT, 
            # 16 --> Energy of FFT between 1 and 5hz, 
            # 17 --> Energy of FFT, 
            # 18 --> signal magnitude area of FFT,
            # 19 --> number of valleys,
            # 20 --> mean value of valleys,
            # 21 --> std value of valleys,
            # 22 --> maximum value of valleys,
            # 23 --> minimum value of peaks,
            # 24 --> 
        """
    if len(windows[:,0])<1:
        return [0, 
                0,
                0, 
                0, 
                0,
                0,
                0,
                0, 
                0, 
                0, 
                0, 
                0, 
                0, 
                0, 
                0,
                0]

    std = np.std(windows[:,2])
    mean_val_z = np.mean(windows[:,2])
    # Mean Absolute Deviation
    mad = np.mean(np.abs(windows[:,2] - mean_val_z))


    
    

    
    # 7. Number of samples above threshold (mean value in this case)
    mean_val = np.mean(windows[:,3])
    threshold = mean_val
    above_threshold = np.sum(windows[:,3] > threshold)
    
    # 8. Number of peaks
    peaks_a, _ = find_peaks(windows[:,3])
    num_peaks_a = len(peaks_a)
    if num_peaks_a>0:
        mean_peaks_a = np.mean(peaks_a)
        std_peaks_a  = np.std(peaks_a)
    else:
        mean_peaks_a = 0
        std_peaks_a = 0
    # 11. Frequency at maximum of FFT
    fft_vals_y = fft(windows[:,1])
    fft_freq_y = np.fft.fftfreq(len(windows[:,1]))
    max_freq_y = fft_freq_y[np.argmax(np.abs(fft_vals_y))]
    
    fft_vals_z = fft(windows[:,2])
    fft_freq_z = np.fft.fftfreq(len(windows[:,2]))
    max_freq_z = fft_freq_z[np.argmax(np.abs(fft_vals_z))]
    
    


    # 12. Maximum of FFT
    max_fft_z = np.max(np.abs(fft_vals_z))
    
    # 13. Standard deviation of FFT
    std_fft_z = np.std(np.abs(fft_vals_z))
    
    # 14. Kurtosis of FFT
    kurt_fft_z = kurtosis(np.abs(fft_vals_z))
    
    # 15. Skewness of FFT
    skew_fft_z = skew(np.abs(fft_vals_z))
    
    # 16. Energy of FFT between 1 and 5 Hz
    fft_freq_z = np.fft.fftfreq(len(windows[:,2]))
    fft_energy_1_5hz_z = np.sum(np.abs(fft_vals_z)[(fft_freq_z >= 1) & (fft_freq_z <= 5)])
    
    # 17. Total energy of FFT
    fft_energy_z = np.sum(np.abs(fft_vals_z))
    fft_vals_a = fft(windows[:,3])
    fft_energy_a = np.sum(np.abs(fft_vals_a))
    # 18. windows Magnitude Area of FFT
    fft_vals_x = fft(windows[:,0])
    sma_fft = np.sum(np.abs(fft_vals_x) + np.abs(fft_vals_y) + np.abs(fft_vals_z)) / len(windows[:,0])
    
 
    
    return[std, 
           mad,
           above_threshold, 
           num_peaks_a, 
           mean_peaks_a,
           std_peaks_a,
           max_freq_y,
           max_freq_z, 
           max_fft_z, 
           std_fft_z, 
           kurt_fft_z, 
           skew_fft_z, 
           fft_energy_1_5hz_z, 
           fft_energy_z, 
           fft_energy_a,
           sma_fft]

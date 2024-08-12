import os

import numpy as np
from scipy.signal import find_peaks # type: ignore
from scipy.stats import kurtosis, skew # type: ignore
from scipy.fft import fft # type: ignore
import torch 
import pandas as pd # type: ignore
import datetime


from dataloader import Dataloader


def feature_names():
    a = ["minimum value",
         "maximun value",
         "mean",
         "variance",
         "energy",
         "mean absolute deviation",
         "number of samples above th",
         "number of peaks",
         "mean value of peaks",
         "std value of peaks",
         "frequency at maximum of FFT",
         "maximum of FFT",
         "std of FFT",
         "kurtosis of FFT",
         "skewness of FFT",
         "Energy of FFT between 1 and 5hz",
         "Energy of FFT",
         "signal magnitude area of FFT",
         "number of valleys",
         "mean value of valleys",
         "std value of valleys",
         "maximum value of valleys",
         "minimum value of peaks",
         ]
    names = []
    for j in ['_x','_y','_z','_a']:
        for i in a:
            names.append(i+j)

    return names


def all_features(windows,labels):
    all_features = []
    step_count = []
    for i in range(windows.shape[0]):
        features = extract_features(windows[i])
        all_features.append(features)
        #print(torch.count_nonzero(labels[i]))
        step_count.append(torch.count_nonzero(labels[i]).item())

    all_features = standarize_feaures(np.array(all_features,dtype=np.float64))
    return all_features,step_count

def extract_features(window):

    # Initialize an empty list for features
    features = []

    # Iterate over each feature component (x, y, z, magnitude)
    for i in range(window.shape[1]):
        component = window[:, i]
        features.extend(calculate_features(component))

    return features

def calculate_features(signal):
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
    # 1. Minimum value
    min_val = torch.min(signal)
    
    # 2. Maximum value
    max_val = torch.max(signal)
    
    # 3. Mean value
    mean_val = torch.mean(signal)
    
    # 4. Variance
    variance = torch.var(signal)
    
    # 5. Energy (sum of squares of the signal)
    energy = torch.sum(signal**2)
    
    # 6. Mean Absolute Deviation
    mad = torch.mean(torch.abs(signal - mean_val))
    
    # 7. Number of samples above threshold (mean value in this case)
    threshold = mean_val
    above_threshold = torch.sum(signal > threshold)
    
    # 8. Number of peaks
    peaks, _ = find_peaks(signal.numpy())
    num_peaks = len(peaks)
    
    # 9. Mean value of peaks
    mean_peaks = torch.mean(signal[peaks]) if num_peaks > 0 else torch.tensor(float('nan'))
    
    # 10. Standard deviation of peaks
    std_peaks = torch.std(signal[peaks]) if num_peaks > 0 else torch.tensor(float('nan'))
    
    # 11. Frequency at maximum of FFT
    fft_vals = fft(signal.numpy())
    fft_freq = np.fft.fftfreq(len(signal))
    max_freq = fft_freq[np.argmax(np.abs(fft_vals))]
    
    # 12. Maximum of FFT
    max_fft = torch.max(torch.tensor(np.abs(fft_vals)))
    
    # 13. Standard deviation of FFT
    std_fft = torch.std(torch.tensor(np.abs(fft_vals)))
    
    # 14. Kurtosis of FFT
    kurt_fft = kurtosis(np.abs(fft_vals))
    
    # 15. Skewness of FFT
    skew_fft = skew(np.abs(fft_vals))
    
    # 16. Energy of FFT between 1 and 5 Hz
    fft_freq = np.fft.fftfreq(len(signal))
    fft_energy_1_5hz = np.sum(np.abs(fft_vals)[(fft_freq >= 1) & (fft_freq <= 5)])
    
    # 17. Total energy of FFT
    fft_energy = np.sum(np.abs(fft_vals))
    
    # 18. Signal Magnitude Area of FFT
    sma_fft = np.sum(np.abs(fft_vals)) / len(signal)
    
    # 19. Number of valleys
    valleys, _ = find_peaks(-signal.numpy())
    num_valleys = len(valleys)
    
    # 20. Mean value of valleys
    mean_valleys = torch.mean(signal[valleys]) if num_valleys > 0 else torch.tensor(float('nan'))
    
    # 21. Standard deviation of valleys
    std_valleys = torch.std(signal[valleys]) if num_valleys > 0 else torch.tensor(float('nan'))
    
    # 22. Maximum value of valleys
    max_valleys = torch.max(signal[valleys]) if num_valleys > 0 else torch.tensor(float('nan'))
    
    # 23. Minimum value of peaks
    min_peaks = torch.min(signal[peaks]) if num_peaks > 0 else torch.tensor(float('nan'))
    
    return [min_val.item(), max_val.item(), mean_val.item(), variance.item(), 
            energy.item(), mad.item(), above_threshold.item(), num_peaks, 
            mean_peaks.item(), std_peaks.item(), max_freq, max_fft.item(), 
            std_fft.item(), kurt_fft, skew_fft, fft_energy_1_5hz, fft_energy, 
            sma_fft, num_valleys, mean_valleys.item(), std_valleys.item(), 
            max_valleys.item(), min_peaks.item()]


def standarize_feaures(feature_matrix):
    files = os.listdir()
    if 'mean.npy' in files and 'std.npy' in files:
        mean = np.load('mean.npy')
        std = np.load('std.npy')
    else:
        mean = np.mean(feature_matrix,axis=0)
        std = np.std(feature_matrix,axis=0)
        mean = np.nan_to_num(mean,nan=1e-8)
        std = np.nan_to_num(std,nan=1e-8)
        mean = np.where(mean==0,1e-8,mean)
        std = np.where(std==0,1e-8,std)

        np.save('mean.npy',mean)
        np.save('std.npy',std)
    print(std)
    feature_matrix = (feature_matrix-mean)/std
    return feature_matrix


class Feature():
    def __init__(self,feature_names) -> None:
        self.feature_names = feature_names
        
    def min_val(self,signal):
        return torch.min(signal)

    def max_val(self,signal):
        return torch.max(signal)
    
    def mean_val(self,signal):
        return torch.mean(signal)

    def variance(self,signal):
        return torch.var(signal)
    
    def energy(self,signal):
        return torch.sum(signal**2)
    
    def mad(self,signal):
        mean_val = self.mean_val(signal)
        return torch.mean(torch.abs(signal - mean_val))
    
    def above_threshold(self,signal,threshold):
       return torch.sum(signal > threshold)
    
    def num_peaks(self,signal):
        peaks, _ = find_peaks(signal.numpy())
        return len(peaks)
    
    def mean_peaks(self,signal):
        peaks, _ = find_peaks(signal.numpy())
        num_peaks = len(peaks)
        return torch.mean(signal[peaks]) if num_peaks > 0 else torch.tensor(float('nan'))
    
    def std_peaks(self,signal):
        peaks, _ = find_peaks(signal.numpy())
        num_peaks = len(peaks)
        return torch.std(signal[peaks]) if num_peaks > 0 else torch.tensor(float('nan'))
    
    def num_valleys(self,signal):
        valleys, _ = find_peaks(-signal.numpy())
        return len(valleys)
    
    def mean_valleys(self,signal):
        valleys, _ = find_peaks(-signal.numpy())
        num_valleys = len(valleys)
        return torch.mean(signal[valleys]) if num_valleys > 0 else torch.tensor(float('nan'))
    
    def std_valleys(self,signal):
        valleys, _ = find_peaks(-signal.numpy())
        num_valleys = len(valleys)
        return torch.std(signal[valleys]) if num_valleys > 0 else torch.tensor(float('nan'))

    def max_valleys(self,signal):
        valleys, _ = find_peaks(-signal.numpy())
        num_valleys = len(valleys)
        torch.max(signal[valleys]) if num_valleys > 0 else torch.tensor(float('nan'))

    def min_peaks(self,signal):
        peaks, _ = find_peaks(signal.numpy())
        num_peaks = len(peaks)
        return torch.min(signal[peaks]) if num_peaks > 0 else torch.tensor(float('nan'))

    def max_freq(self,signal):
        fft_vals = fft(signal.numpy())
        fft_freq = np.fft.fftfreq(len(signal))
        return fft_freq[np.argmax(np.abs(fft_vals))]
    
    def max_fft(self,signal):
        fft_vals = fft(signal.numpy())
        return torch.max(torch.tensor(np.abs(fft_vals)))
    
    def std_fft(self,signal):
        fft_vals = fft(signal.numpy())
        return torch.std(torch.tensor(np.abs(fft_vals)))
    
    def kurt_fft(self,signal):
        fft_vals = fft(signal.numpy())
        return kurtosis(np.abs(fft_vals))
    
    def skew_fft(self,signal):
        fft_vals = fft(signal.numpy())
        return skew(np.abs(fft_vals))
    
    def fft_energy_1_5hz(signal):
        fft_freq = np.fft.fftfreq(len(signal))
        fft_vals = fft(signal.numpy())
        return np.sum(np.abs(fft_vals)[(fft_freq >= 1) & (fft_freq <= 5)])
    
    def fft_energy(signal):
        fft_vals = fft(signal.numpy())
        return np.sum(np.abs(fft_vals))
    
    def sma_fft(self,signal_x,signal_y,signal_z):
        fft_vals_x = fft(signal_x.numpy())
        fft_vals_y = fft(signal_y.numpy())
        fft_vals_z = fft(signal_z.numpy())
        return (np.sum(np.abs(fft_vals_x)) + np.sum(np.abs(fft_vals_y)) + np.sum(np.abs(fft_vals_z)) )/ len(signal_x)
    

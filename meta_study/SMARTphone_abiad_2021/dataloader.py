import numpy as np
import os
import sys
import pandas as pd # type: ignore
from scipy.signal import find_peaks, peak_prominences # type: ignore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from meta_study.dataloader.base_dataloader import BaseDataLoader


class SmartDataloader():

    def __init__(self,root,ToFilter=False,AddMagnitude=True,AddGyro=True,normalize=True) -> None:
        self.root = root
        self.ToFilter = ToFilter
        self.AddMagnitude = AddMagnitude
        self.AddGyro = AddGyro
        self.normalize = normalize

        base_data = BaseDataLoader(self.root,self.ToFilter,self.AddMagnitude,self.AddGyro,self.normalize)

        data,label = base_data.data,base_data.labels


        self.sf = 80

        self.features_acc = {'index_of_max': 0.3,
                            'skew_1': 0.05,
                            'kurt': 0.3,
                            'median': 0.8,
                            'valley_prominence': 0.8,
                            'peak_prominence_1': 0.8,
                            'peak_prominence_2': 0.5,
                            'first_dominant_freq': 1.28
                            }#{'index_of_maximum_value':0.3,'skew':0.05,'kurt':0.3,'median':0.8,'Valley_prominence':0.8, 'peak_prominence':0.8, 'peak_prominence':0.5, 'First_dominant_freq':1.28}
        self.features_gyro = {
                            'index_of_min': 0.6,
                            'skew_1': 0.7,
                            'skew_2': 0.3,
                            'amplitude_of_first_dominant_freq': 1.28,
                            'signal_magnitude_area': 0.8,
                            'variance': 0.2,
                            'max_value': 0.7,
                            'valley_prominence': 0.8
                            }#{'index_of_minimum_value':0.6,'skew_07':0.7,'skew_03':0.3,'Amplitude of first dominant frequency':1.28,'signal_magnitude_area':1.28,'variance':0.8,'maximum_value':0.7,'valley_prominence':0.8}

        acc_feature_data = []
        for feature_name, window_size in self.features_acc.items():
            window_size_samples = int(window_size * self.sf)  # Convert seconds to samples (assuming 100 Hz sampling rate)
            acc_feature_data.append(self.extract_window_features(data[:, 6], {feature_name: window_size}, window_size_samples))

        gyro_feature_data = []
        for feature_name, window_size in self.features_gyro.items():
            window_size_samples = int(window_size * self.sf)  # Convert seconds to samples (assuming 100 Hz sampling rate)
            gyro_feature_data.append(self.extract_window_features(data[:, 7], {feature_name: window_size}, window_size_samples))

        combined_feature_data = np.hstack(acc_feature_data + gyro_feature_data)
        
        # Align labels (assuming labels match the number of feature windows)
        num_windows = combined_feature_data.shape[0]
        labels = label[:num_windows]
        print(len(acc_feature_data),len(acc_feature_data[0]),len(acc_feature_data[1]),len(acc_feature_data[2]),len(acc_feature_data[3]),len(acc_feature_data[4]),len(acc_feature_data[5]),len(acc_feature_data[6]),len(acc_feature_data[7]),
              len(labels))

    def extract_feature_from_window(self,signal_window,feature_set):
        features = []
        if 'index_of_max' in feature_set:
            features.append(np.argmax(signal_window))
        if 'skew' in feature_set:
            features.append(pd.Series(signal_window).skew())
        if 'kurt' in feature_set:
            features.append(pd.Series(signal_window).kurtosis())
        if 'median' in feature_set:
            features.append(np.median(signal_window))
        if 'valley_prominence' in feature_set:
            peaks,_ = find_peaks(-signal_window)
            prominences = peak_prominences(-signal_window, peaks)[0]
            features.append(prominences) 
        if 'peak_prominence' in feature_set:
            peaks,_ = find_peaks(signal_window)
            prominences = peak_prominences(signal_window, peaks)[0]
            features.append(prominences)
        if 'first_dominant_freq' in feature_set:
            features.append(np.fft.fft(signal_window).argmax())
        if 'amplitude_of_first_dominant_freq' in feature_set:
            features.append(np.abs(np.fft.fft(signal_window)).max())
        if 'signal_magnitude_area' in feature_set:
            features.append(np.sum(np.abs(signal_window)))
        if 'variance' in feature_set:
            features.append(np.var(signal_window))
        if 'index_of_min' in feature_set:
            features.append(np.argmin(signal_window))
        return features
    
    def extract_window_features(self,signals, feature_set, window_size):
        features = []
        for start in range(0, len(signals) - window_size + 1, window_size):
            window = signals[start:start + window_size]
            features.append(self.extract_feature_from_window(window, feature_set))  # Assuming single channel extraction
        return np.array(features)

            

if __name__=='__main__':
    a = SmartDataloader('/home/ann_ss22_group4/step detection/SIMUL-dataset/data/by-person/train',True,True,True,True)

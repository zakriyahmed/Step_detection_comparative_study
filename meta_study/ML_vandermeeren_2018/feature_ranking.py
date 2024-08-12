import os
import builtins
from itertools import combinations

import numpy as np
from scipy.signal import find_peaks # type: ignore
from scipy.stats import kurtosis, skew # type: ignore
from scipy.fft import fft # type: ignore
import torch 
import pandas as pd # type: ignore
import datetime


from dataloader import Dataloader
from trying import feature_names,all_features

original_print = builtins.print
# Define the custom print function
def timestamped_print(*args, **kwargs):
    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Format the message with the timestamp
    message = f"{timestamp} - " + ' '.join(map(str, args))
    # Print the formatted message
    original_print(message, **kwargs)

# Override the built-in print function
builtins.print = timestamped_print


root = '/home/ann_ss22_group4/step detection/SIMUL-dataset/data/by-person/test'
ToFilter = True
AddMagnitude = True
AddGyro = False
normalize = True
window_size = 100
stride = 100  
a = Dataloader(root,ToFilter=ToFilter,AddMagnitude=AddMagnitude,AddGyro=AddGyro,normalize=normalize,window_size=100,stride = stride, windowed_labels=True)
windows,labels = a.windows_data,a.windows_labels

feature_matrix,count = all_features(windows,labels)


def compute_histograms(features,labels):
    unique_labels = np.unique(labels)
    histograms = {label: [] for label in unique_labels}
    
    for label in unique_labels:
        label_features = features[labels == label]
        for feature in range(features.shape[1]):
            hist, _ = np.histogram(label_features[:, feature], bins=10, range=(0, 1))
            histograms[label].append(hist)
    
    return histograms

def rank_features_by_bhattacharyya(features, labels):
    histograms = compute_histograms(features, labels)
    num_features = features.shape[1]
    bhattacharyya_scores = []
    
    for feature in range(num_features):
        hist1 = histograms[0][feature]
        hist2 = histograms[1][feature]
        bc = bhattacharyya_coefficient(hist1, hist2)
        bhattacharyya_scores.append(bc)
    
    # Rank features based on Bhattacharyya coefficient
    ranked_features = np.argsort(bhattacharyya_scores)
    return ranked_features, bhattacharyya_scores

def rank_features_by_bhattacharyya(features, labels):
    histograms = compute_histograms(features, labels)
    num_features = features.shape[1]
    bhattacharyya_scores = np.zeros(num_features)
    
    for feature in range(num_features):
        pairwise_distances = []
        for (cls1, cls2) in combinations(range(9), 2):
            print(cls1,cls2,feature)
            hist1 = histograms[cls1][feature]
            hist2 = histograms[cls2][feature]
            bc = bhattacharyya_coefficient(hist1, hist2)
            pairwise_distances.append(bc)
        # Aggregate pairwise distances for this feature
        bhattacharyya_scores[feature] = np.mean(pairwise_distances)
        print(feature,len(pairwise_distances))
    # Rank features based on Bhattacharyya scores (higher is better)
    ranked_features = np.argsort(-bhattacharyya_scores)
    return ranked_features, bhattacharyya_scores

def rank_features_by_bhattacharyya(features, labels, K=10, bins=10):

    histograms = compute_histograms(features, labels)
    num_features = features.shape[1]
    num_classes=9
    # Initialize the matrix P
    P = np.zeros((num_classes - 1, K), dtype=int)
    
    for n in range(num_classes - 1):
        bhattacharyya_distances = []
        for feature in range(num_features):
            hist1 = histograms[n][feature]
            hist2 = histograms[n + 1][feature]
            bc = bhattacharyya_coefficient(hist1, hist2)
            bhattacharyya_distances.append(bc)
        
        # Rank features based on Bhattacharyya distances for the current class pair
        ranked_features = np.argsort(-np.array(bhattacharyya_distances))[:K]
        P[n, :] = ranked_features
    
    # Convert matrix P to vector v
    v = P.T.flatten()
    
    # Remove duplicates while preserving order
    _, idx = np.unique(v, return_index=True)
    v = v[np.sort(idx)]
    
    return v

def bhattacharyya_coefficient(H1,H2):
    H1 = H1+1e-8
    H2 = H2+1e-8
    H1 = H1/np.sum(H1)
    H2 = H2/np.sum(H2)
    mean_H1 = np.mean(H1)
    mean_H2 = np.mean(H2)
    #print(mean_H1,mean_H2)
    numerator = np.sum(np.sqrt(H1 * H2)) / len(H1)
    denominator = np.sqrt(mean_H1 * mean_H2)

    dBhat = np.sqrt(1 - numerator / denominator)
    #print(numerator,denominator,dBhat)
    return dBhat


ranked_features = rank_features_by_bhattacharyya(feature_matrix, count)

histograms = compute_histograms(feature_matrix,count)
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import pandas as pd # type: ignore
from IPython.display import clear_output


from dataloader import LeeDataloader

# Constants (these should be set according to your specific application or data)


# Helper functions
def update_statistics(values):
    if len(values)>0:
        return np.mean(values), np.std(values)
    else:
        return 0,0

def detect_candidate(an_minus_1, an, an_plus_1, mean_a, sigma_a, alpha):
    if an > max(an_minus_1, an_plus_1, mean_a + (sigma_a / alpha)):
        return 'peak'
    elif an < min(an_minus_1, an_plus_1,mean_a- (sigma_a/alpha)):
        return 'valley'
    else:
        return 'intermediate'

def update_peak(an, n):
    global mean_p, sigma_p, recent_peaks
    recent_peaks.append(n)
    if len(recent_peaks) > M:
        recent_peaks.pop(0)
    intervals = np.diff(recent_peaks)
    mean_p, sigma_p = update_statistics(intervals)

def update_valley(an, n):
    global mean_v, sigma_v, recent_valleys
    recent_valleys.append(n)
    if len(recent_valleys) > M:
        recent_valleys.pop(0)
    intervals = np.diff(recent_valleys)
    mean_v, sigma_v = update_statistics(intervals)

# Main step detection function
def step_detection(data,labels):
    global mean_a, sigma_a, mean_p, sigma_p, mean_v, sigma_v
    count = 0
    Sn = 'init'
    np_index = nv_index = 0  # Initializing the indices of the last peak and valley
    step_indices = []

    magnitudes = np.linalg.norm(data, axis=1)

    for n in range(1, len(magnitudes) - 1):
        an = magnitudes[n]
        an_minus_1 = magnitudes[n - 1]
        an_plus_1 = magnitudes[n + 1]
        
        # Update the step deviation
        recent_samples = magnitudes[max(0, n - K):n + 1]
        recent_labels = labels[max(0, n - K):n + 1,0]

        Sc = detect_candidate(an_minus_1, an, an_plus_1, mean_a, sigma_a, alpha)

        if Sc == 'peak':
            if Sn == 'init':
                Sn = 'peak'
                update_peak(an, n)
                np_index = n
                mean_a = (magnitudes[np_index] + magnitudes[nv_index]) / 2
            elif Sn == 'valley' and n - np_index > mean_p - sigma_p * beta:
                Sn = 'peak'
                update_peak(an, n)
                np_index = n
                mean_a = (magnitudes[np_index] + magnitudes[nv_index]) / 2
            elif Sn == 'peak' and n - np_index <= mean_p - sigma_p * beta and an > magnitudes[np_index]:
                update_peak(an, n)
                np_index = n
        elif Sc == 'valley':
            if Sn == 'peak' and n - nv_index > mean_v - sigma_v * beta:
                Sn = 'valley'
                update_valley(an, n)
                nv_index = n
                count += 1
                step_indices.append([np_index, nv_index])
                mean_a = (magnitudes[np_index] + magnitudes[nv_index]) / 2
            elif Sn == 'valley' and n - nv_index <= mean_v - sigma_v * beta and an < magnitudes[nv_index]:
                update_valley(an, n)
                nv_index = n
        
        # Update state
        Sn = Sc if Sc != 'intermediate' else Sn
        sigma_a = np.std(recent_samples)
    return count, step_indices


if __name__=="__main__":
    alphas = [2,3,4,5,6,7,8]
    betas = [2,3,4,5,6,7,8]
    ks = [10,20,30,40,50,60,70,80]
    Ms = [10,20,30,40,50,60,70,80]

    data_frame = {'alpha':[],'beta':[],'K':[],'M':[],'accuracy':[]}
    a = LeeDataloader('/home/ann_ss22_group4/step detection/SIMUL-dataset/data/by-person/test',ToFilter=True)
    data = a.data
    labels = a.label
    count_s = np.count_nonzero(labels[:,0])
    for alpha in alphas:
        for beta in betas:
            for K in ks:
                for M in Ms:

                    #alpha = 4
                    #beta = 4
                    #K = 120  # Number of recent acceleration samples for step deviation
                    #M = 60  # Number of recent peaks/valleys for interval statistics

                    # Initialize parameters
                    mean_a = 0  # Step average
                    sigma_a = 0  # Step deviation
                    mean_p = 0  # Average peak interval
                    sigma_p = 0  # Standard deviation of peak intervals
                    mean_v = 0  # Average valley interval
                    sigma_v = 0  # Standard deviation of valley intervals

                    # Placeholder for recent peaks and valleys
                    recent_peaks = []
                    recent_valleys = []
                    
                    step_count, step_indices = step_detection(data[:,0:3],labels)

                    acc = (step_count/count_s)/100

                    data_frame['alpha'].append(alpha)
                    data_frame['beta'].append(beta)
                    data_frame['K'].append(K)
                    data_frame['M'].append(M)
                    data_frame['accuracy'].append(acc)

                    df = pd.DataFrame(data_frame)
                    print(df.tail(10))
                    df.to_csv('grid_search.csv')
































"""sh_s = 18000
sh_e = 20000
label_s = np.where(labels[sh_s:sh_e,0]==1)[0]
label_e = np.where(labels[sh_s:sh_e,1]==1)[0]

plt.figure(figsize=(15,5))
plt.plot(data[sh_s:sh_e,3])
start_points = data[label_s+sh_s,3]
end_points = data[label_e+sh_s,3]
plt.scatter(label_s,start_points,marker='x',color='green')
plt.scatter(label_e,end_points,marker='x',color='red')
pred_start_points = data[np.array(step_indices)[:,0]+15000,3]
plt.scatter(np.array(step_indices)[70:115,0]-3000,pred_start_points[70:115],color='green')
pred_end_points = data[np.array(step_indices)[:,1]+15000,3]
plt.scatter(np.array(step_indices)[70:115,1]-3000,pred_end_points[70:115],color='red')"""
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import pandas as pd # type: ignore
from IPython.display import clear_output


from dataloader import LeeDataloader

# Constants (these should be set according to your specific application or data)


# Helper functions
class LeeDetector():
    def __init__(self,alpha,beta,K,M) -> None:
        self.alpha = alpha
        self.beta = beta
        self.K = K
        self.M = M

        self.mean_a = 0
        
        self.sigma_a = 0  # Step deviation
        self.mean_p = 0  # Average peak interval
        self.sigma_p = 0  # Standard deviation of peak intervals
        self.mean_v = 0  # Average valley interval
        self.sigma_v = 0  # Standard deviation of valley intervals

        self.recent_peaks = []
        self.recent_valleys = []

    def update_statistics(self,values):
        if len(values)>0:
            return np.mean(values), np.std(values)
        else:
            return 0,0
        
    def detect_candidate(self,an_minus_1, an, an_plus_1, mean_a, sigma_a, alpha):
        if an > max(an_minus_1, an_plus_1, mean_a + (sigma_a / alpha)):
            return 'peak'
        elif an < min(an_minus_1, an_plus_1,mean_a- (sigma_a/alpha)):
            return 'valley'
        else:
            return 'intermediate'
    
    def update_peak(self, n):
        self.recent_peaks.append(n)
        if len(self.recent_peaks) > self.M:
            self.recent_peaks.pop(0)
        intervals = np.diff(self.recent_peaks)
        self.mean_p, self.sigma_p = self.update_statistics(intervals)

    def update_valley(self, n):
        
        self.recent_valleys.append(n)
        if len(self.recent_valleys) > self.M:
            self.recent_valleys.pop(0)
        intervals = np.diff(self.recent_valleys)
        self.mean_v, self.sigma_v = self.update_statistics(intervals)

    def step_detection(self,data,labels):
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
            recent_samples = magnitudes[max(0, n - self.K):n + 1]
            recent_labels = labels[max(0, n - self.K):n + 1,0]

            Sc = self.detect_candidate(an_minus_1, an, an_plus_1, self.mean_a, self.sigma_a, self.alpha)

            if Sc == 'peak':
                if Sn == 'init':
                    Sn = 'peak'
                    self.update_peak(n)
                    np_index = n
                    self.mean_a = (magnitudes[np_index] + magnitudes[nv_index]) / 2
                elif Sn == 'valley' and n - np_index > self.mean_p - self.sigma_p * self.beta:
                    Sn = 'peak'
                    self.update_peak(n)
                    np_index = n
                    self.mean_a = (magnitudes[np_index] + magnitudes[nv_index]) / 2
                elif Sn == 'peak' and n - np_index <= self.mean_p - self.sigma_p * self.beta and an > magnitudes[np_index]:
                    self.update_peak(n)
                    np_index = n
            elif Sc == 'valley':
                if Sn == 'peak' and n - nv_index > self.mean_v - self.sigma_v * self.beta:
                    Sn = 'valley'
                    self.update_valley(n)
                    nv_index = n
                    count += 1
                    step_indices.append([np_index, nv_index])
                    self.mean_a = (magnitudes[np_index] + magnitudes[nv_index]) / 2
                elif Sn == 'valley' and n - nv_index <= self.mean_v - self.sigma_v * self.beta and an < magnitudes[nv_index]:
                    self.update_valley(n)
                    #nv_index = n
            
            # Update state
            Sn = Sc if Sc != 'intermediate' else Sn
            self.sigma_a = np.std(recent_samples)
        return count, step_indices





if __name__=="__main__":
    alphas = [6,7,8,9,10]
    betas = [4,5,3]
    ks = [10,20,30,40,50,60,70,80]
    Ms = [60,70,80]

    data_frame = {'alpha':[],'beta':[],'K':[],'M':[],'accuracy':[]}
    a = LeeDataloader('/home/ann_ss22_group4/step detection/SIMUL-dataset/data/by-person/train',ToFilter=True)
    data = a.data
    labels = a.label
    count_s = np.count_nonzero(labels[:,0])
    for alpha in alphas:
        for beta in betas:
            for K in ks:
                for M in Ms:
                    
                    b = LeeDetector(alpha,beta,K,M)
                    
                    step_count, step_indices = b.step_detection(data[50:,0:3],labels)

                    acc = (step_count/count_s)*100

                    data_frame['alpha'].append(alpha)
                    data_frame['beta'].append(beta)
                    data_frame['K'].append(K)
                    data_frame['M'].append(M)
                    data_frame['accuracy'].append(acc)
                    print(f"alpha:{alpha} , beta:{beta} , K:{K} , M:{M}, acc:{acc}")
                    df = pd.DataFrame(data_frame)
                    
                    df.to_csv('grid_search_new.csv')
































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
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import pandas as pd # type: ignore
from IPython.display import clear_output


from dataloader import LeeDataloader

from grid_search import *





if __name__ == '__main__':
    a = LeeDataloader('/home/ann_ss22_group4/step detection/SIMUL-dataset/data/by-person/test',ToFilter=True)

    data = a.data
    labels = a.label
    count_s = np.count_nonzero(labels[:,0])

    df = pd.read_csv('grid_search.csv')

    idx = df['accuracy'].idxmax()

    alpha = df['alpha'][idx]
    beta = df['beta'][idx]
    K = df['K'][idx]
    M = df['M'][idx]
    print(f"K:{K}\tM:{M}\talpha:{alpha}\tbeta:{beta}")

    b = LeeDetector(alpha,beta,K,M)



    step_count, step_indices = b.step_detection(data[50:,0:3],labels)
    step_indices = np.array(step_indices)

    start_pred = np.array(step_indices[:,0])
    end_pred  = np.array(step_indices[:,1])
    pred = np.zeros_like(labels)
    pred[start_pred,0] = 1
    pred[end_pred,1] = 1
    print(len(np.nonzero(pred[:,0])))
    pred_df = pd.DataFrame({'start':pred[:,0],'end':pred[:,1]})
    pred_df.to_csv('pred.csv')

    true_df = pd.DataFrame({'start':labels[:,0],'end':labels[:,1]})
    true_df.to_csv('label.csv')


    # tsi = np.nonzero(labels[:,0])[0]
    # tei = np.nonzero(labels[:,1])[0]
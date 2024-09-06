import numpy as np
import os
import sys
import pandas as pd # type: ignore
from scipy.signal import find_peaks, peak_prominences # type: ignore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from meta_study.dataloader.base_dataloader import BaseDataLoader


class LeeDataloader():

    def __init__(self,root,ToFilter=True,AddMagnitude=True,AddGyro=False,normalize=False) -> None:
        self.root = root
        self.ToFilter = ToFilter
        self.AddMagnitude = AddMagnitude
        self.AddGyro = AddGyro
        self.normalize = normalize

        base_data = BaseDataLoader(self.root,self.ToFilter,self.AddMagnitude,self.AddGyro,self.normalize,sensor='all')

        self.data,self.label = base_data.data,base_data.labels
        self.sensor_map = base_data.sensor_map
        self.sensor_labels = base_data.sensor_labels
        self.activity = base_data.activity

        


if __name__=='__main__':
    a = LeeDataloader('/home/ann_ss22_group4/step detection/SIMUL-dataset/data/by-person/train',True,False,False,False)

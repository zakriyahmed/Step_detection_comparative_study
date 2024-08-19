import torch
import pandas as pd # type: ignore
import os
print(os.getcwd())

from train import Train, Test

if __name__=='__main__':

    a = Train('/home/ann_ss22_group4/step detection/SIMUL-dataset/data/by-person/train','cuda',0.001,300)
    a.train()
    a.save_model()

    b = Test('/home/ann_ss22_group4/step detection/SIMUL-dataset/data/by-person/test','cuda',0.001,30)
    _,_,start,end = b.test()
    #print(start_with_patches.shape)
    #clean_start = utils.remove_small_brusts(start_with_patches,10)
    #clean_end = utils.remove_small_brusts(end_with_patches,10)

    #start = utils.remove_patches(start)
    #end = utils.remove_patches(end)
    df = {'start':start,'end':end}
    df = pd.DataFrame(df)

    df.to_csv('BITLST_2015.csv')


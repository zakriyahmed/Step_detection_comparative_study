import torch
import pandas as pd # type: ignore


from train import Train, Test
import utils

if __name__=='__main__':

    a = Train('/home/ann_ss22_group4/step detection/SIMUL-dataset/data/by-person/train','cuda',0.001,30)
    a.train()
    a.save_model()

    b = Test('/home/ann_ss22_group4/step detection/SIMUL-dataset/data/by-person/test','cuda',0.001,30)
    start_with_patches,end_with_patches = b.test()

    clean_start = utils.remove_small_brusts(start_with_patches,10)
    clean_end = utils.remove_small_brusts(end_with_patches,10)

    start = utils.remove_patches(clean_start)
    end = utils.remove_patches(clean_end)
    df = {'start':start,'end':end}
    df = pd.DataFrame(df)

    df.to_csv('BITLST_2015.csv')


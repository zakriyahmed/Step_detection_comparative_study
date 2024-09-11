import torch
import pandas as pd # type: ignore
import sys

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout  # Save the original stdout
        self.log = open(filename, "w", buffering=1)  # Line-buffered file

    def write(self, message):
        self.terminal.write(message)  # Write to console (terminal)
        self.log.write(message)       # Write to file
        self.log.flush()              # Ensure file gets updated immediately

    def flush(self):
        self.terminal.flush()         # Ensure terminal output is flushed
        self.log.flush()  


sys.stdout = Logger("logs.txt")
print('Staring logs\n')

from train import Train, Test
import utils

if __name__=='__main__':

    a = Train('/home/ann_ss22_group4/step detection/SIMUL-dataset/data/by-person/train','cuda',0.001,100)
    a.train()
    a.save_model()

    b = Test('/home/ann_ss22_group4/step detection/SIMUL-dataset/data/by-person/test','cpu',0.001,100)
    start_with_patches,end_with_patches = b.test()

    clean_start = utils.remove_small_brusts(start_with_patches[0],15)
    clean_end = utils.remove_small_brusts(end_with_patches[0],15)

    start = utils.remove_patches(clean_start)
    end = utils.remove_patches(clean_end)
    print(len(start))
    df = {'start':start,'end':end}
    df = pd.DataFrame(df)

    df.to_csv('pred.csv')

    df_label = {'start':b.ground_labels[:,0],'end':b.ground_labels[:,1],'activity':b.activity,'sensor':b.sensor_labels}

    df_label = pd.DataFrame(df_label)
    df_label.to_csv('label.csv')
    #sys.stdout.log.close()

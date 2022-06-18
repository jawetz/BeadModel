import numpy as np
import pandas as pd

def k_calc(args):
    data=pd.read_csv(r"string_data").to_numpy()
    data.astype('float64')
    if args.tens==0:
        delt=0
        slope=1000*(data[20,2]-data[0,2])/(data[20,1]-data[0,1])
    for i in range(data.shape[0]):
        if args.tens<=data[i,2]:
            delt=data[i,1]
            slope=1000*(data[i+20,2]-data[i-20,2])/(data[i+20,1]-data[i-20,1])
            return delt,slope

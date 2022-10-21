import pandas as pd
import numpy as np

def normalise(array):
    M = np.max(array)
    m = np.min(array)
    
    return ((array - m)/(M-m))


data = pd.read_csv("CleanData.csv")
ip = np.array(data.iloc[:,1:-1])
op = np.array(data.iloc[:,-1:])

ip = ip.astype(np.float64)
op = op.astype(np.float64)

ip = ip.T
ip[0] = normalise(ip[0])
ip[1] = normalise(ip[1])
ip[2] = normalise(ip[2])

ip = ip.T

MAX = np.max(op)
MIN = np.min(op)

op = normalise(op)

ip = ip[:10]
op = op[:10]

print(ip.shape, op.shape)
#%%

from NeuralNetwork import *

net = nn(ip,op)
net.train() 
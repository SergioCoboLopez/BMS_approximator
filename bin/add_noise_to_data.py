#19/11/2024. This code takes the generative data and adds gaussian noise to them

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import matplotlib.gridspec as gridspec

np.random.seed(seed=1111)

function='tanh' #tanh, leaky_ReLU, ReLU
mean=0;sigma=0.2;realization=0
sigmas=[i for i in np.arange(0,0.22,0.02)]

#Read double resolution file
file_data='NN_function_' + function + '_NREP_10_data' + '.csv'
data='../data/generative_data/' + file_data
d=pd.read_csv(data)
d=d.drop(columns='Unnamed: 0')
print(d)

sample=d.index.stop
print(d.index.stop)


for sigma in sigmas:
    for r in range(3):
        
        noise = np.random.normal(mean,sigma,sample)            

        #Add Gaussian noise to data
        d['noise']=noise
        d['y_noise']= d['y'] + d['noise']
        
        #Save data
        d.to_csv('../data/nns/1x_resolution/' + 'NN_' + function + '_sigma_' + str(sigma) + '_r_' + str(r) +  '.csv')


#24/1/25. Following up on similar codes, this one adds noise to the nguyen datasets

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import matplotlib.gridspec as gridspec

np.random.seed(seed=1111)


#Read nguyen data
input_path='../../data/generative_data/'
filename='nguyen_data_all.csv'
data=input_path + filename
d=pd.read_csv(data)
d=d.drop(columns='Unnamed: 0')


realizations=3

sigma_max=0.2
sigma_step=0.02
sigmas=[i for i in np.arange(0,sigma_max + sigma_step,sigma_step)]

mean=0;sample=d.index.stop
print(d.index.stop)


for sigma in sigmas:
    for r in range(realizations):

        noise = np.random.normal(mean,sigma,sample)

        #Add noise to high resolution data
        d['noise']=noise
        d['z_noise']= d['z'] + d['noise']

        #Save data
        output_path='../../data/noisy_data/nguyen/'
        d.to_csv(output_path + 'NN_nguyen'+ '_sigma_' + str(sigma) + '_r_' + str(r) +  '.csv')

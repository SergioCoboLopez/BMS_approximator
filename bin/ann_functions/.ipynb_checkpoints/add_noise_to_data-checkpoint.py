#19/11/2024. This code takes the generative data and adds gaussian noise to them

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import matplotlib.gridspec as gridspec

np.random.seed(seed=1111)

resolutions={'0.5x':'0.1', '1x':'0.05' , '2x': '0.025' , '4e-3x':'0.004' }


def add_noise_to_data(activation_function, resolution_var, sigma_v, realizations):

    NREP=10

    #read data
    input_path='../../data/generative_data/'
    filename='NN_function_' + activation_function + '_NREP_' + str(NREP) + '_res_'+ resolutions[resolution_var] + '_data' + '.csv'
    data=input_path + filename

    d=pd.read_csv(data)
    d=d.drop(columns='Unnamed: 0')

    #add noise
    mean=0;sample=d.index.stop

    for sigma in sigma_v:
        for r in range(realizations):

            noise = np.random.normal(mean,sigma,sample)

            #Add Gaussian noise to data
            d['noise']=noise
            d['y_noise']= d['y'] + d['noise']

            #Save data
            output_path='../../data/noisy_data/' + resolution_var + '_resolution/'
            d.to_csv(output_path + 'NN_' + activation_function + '_sigma_' + str(sigma) + '_r_' + str(r) + '_res_' + resolutions[resolution_var] +   '.csv')

    return None



resolution='1x' #'0.5x', '1x', '2x', '4e-3'

function='leaky_ReLU' #tanh, leaky_ReLU, ReLU
r=3

sigma_max=0.2
sigma_step=0.02
sigmas=[i for i in np.arange(0,sigma_max + sigma_step,sigma_step)]

add_noise_to_data(function, resolution, sigmas, r)



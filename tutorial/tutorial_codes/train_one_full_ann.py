#26/5/2025
# This code trains a single neural network with all data points. The only function of this code is to generate one of the figures in the tutorial notebook. Although it is inspired on the code train_anns.py, it seemed more pragmatic to have it stored here since it has a very specific function.

import pyrenn
import numpy as np
import copy
import pandas as pd
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import sys
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error
from random import sample
import random

def build_validation(all_training_points,len_validation,dataframe):

    pre_train_df=dataframe.loc[:]    
    validation_df=dataframe.iloc[validation_points]
    train_df=pre_train_df.drop(labels=validation_points)

    return train_df, validation_df

#Read data
#-----------------------------------------------------
random.seed(a=1111)

function='tanh' #tanh, leaky_ReLU or others
sigma=0.1 #0.0 to 0.2 in steps of 0.02
n=8
realization=0
train_size=80

resolution='1x' #1x, 2x, 0.5x, 4e-3x
resolutions={'1x': '0.05', '0.5x':'0.1','2x': '0.025' , '4e-3x':'0.004' }

output_path='../../data/tutorial/'
input_path= '../../data/noisy_data/' + resolution + '_resolution/'

filename=input_path + 'NN_' + function + '_sigma_' + str(sigma) + '_r_' + str(realization) + '_res_' + resolutions[resolution] + '.csv'


d=pd.read_csv(filename)
d=d.drop(columns='Unnamed: 0')
#Take subset of data
d=d[(d['x1'] >= -2.0) & (d['x1']<=2.0)]
d=d.reset_index(drop=True)

#Build ANN
ILS = 1;OLS=1
NL, LS = 5, 10
arch=[ILS] + NL*[LS] + [OLS]
nn=pyrenn.CreateNN(arch)

#Cross validations
train_border=d[d['rep']==0].loc[train_size-1]['x1']
iterations=300

dn=d[d['rep']==n]
dn.index.name = None
dn=dn.reset_index(drop=True)

#Train NN with all points
xtrain = dn.loc[0:train_size-1]['x1']
ytrain = dn.loc[0:train_size-1]['y_noise']

net=pyrenn.train_LM(xtrain,ytrain,nn,verbose=True,k_max=100,E_stop=1e-5)

#Test NN on all points                                                                               
xtest = dn.loc[train_size:]['x1']
ytest = pyrenn.NNOut(xtrain,net)
ypred = pyrenn.NNOut(xtest,net)

print(ytest)
print(len(ytest))

#Add predictions and save data 
dn['ymodel']=ytest

#Save updated data with model
dn.to_csv( output_path + 'NN_no_overfit_test_full_train_' + function + '_sigma_' + str(sigma) + '_r_' + str(realization) + '.csv')

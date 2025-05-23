#10/9/2024
# This code trains neural networks with a validation set.
# The validation set consists of randomly chosen (with a seed) points among the training set.
# The train set consists of 3/4 of the data points comprising the data.

# The train set is split in another training set (50 points) and a validation set (10 points). The number of points may change in future versions of this code.
# The training is done in sets of two epochs and there are n iterations of these sets. After each set, the errors (MAE and RMSE) are evaluated.
#The code saves the best neural network. Best, being the neural network with the minimum RMSE on the validation set, meaining that there is no overfitting.
#The code plots the errors (RMSE and MAE) as a function of the iterations. It also plots the first nn, the last nn, and the best nn together with the original signal and the signal with noise.
#In this version, MAE is purely informative and the best model is based on the minimum RMSE.

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

    pre_train_df=dataframe.loc[0:all_training_points-1]    
    validation_df=dataframe.iloc[validation_points]
    train_df=pre_train_df.drop(labels=validation_points)

    return train_df, validation_df

#Read data
#-----------------------------------------------------
random.seed(a=1111)

function='tanh' #tanh, leaky_ReLU, ReLU
sigma=0.0 #0.0 to 0.2 in steps of 0.02
realization=0


resolution='4e-3x' #1x, 2x, 0.5x, 4e-3x
resolutions={'1x': '0.05', '0.5x':'0.1','2x': '0.025' , '4e-3x':'0.004' }

output_path='../data/' + resolution + '_resolution/trained_nns/'
input_path= '../data/' + resolution + '_resolution/'
filename=input_path + 'NN_' + function + '_sigma_' + str(sigma) + '_r_' + str(realization) + \
    '_res_' + resolutions[resolution] + '.csv'


d=pd.read_csv(filename)
d=d.drop(columns='Unnamed: 0')

#Take subset of data
d=d[(d['x1'] >= -2.0) & (d['x1']<=2.0)]
d=d.reset_index(drop=True)

#train/validation size
n_points=int(len(d.index)/10)
# train_fraction=5/8;valid_fraction=1/8;test_fraction=0.25
# train_size=int(n_points*train_fraction)
# validation_size=train_size + int(n_points*valid_fraction)

#new validation scheme
#-----------------------------------------------------
pre_train_fraction=3/4;pre_train_size=int(n_points*pre_train_fraction)
validation_fraction=1/8;validation_size=int(n_points*validation_fraction)
validation_points=sample(range(pre_train_size), k=validation_size)#sample points from uniform sample
validation_points=np.sort(validation_points)
print(validation_size)
print(validation_points)
#-----------------------------------------------------


#Build ANN
ILS = 1;OLS=1
NL, LS = 5, 10
arch=[ILS] + NL*[LS] + [OLS]
nn=pyrenn.CreateNN(arch)

#Cross validations
train_border=d[d['rep']==0].loc[train_size-1]['x1']

n_functions=int(d['rep'].max()) #Number of functions in dataset
iterations=300

for n in range(n_functions + 1):
    #Read data
    dn=d[d['rep']==n]
    dn.index.name = None
    dn=dn.reset_index(drop=True)

    #Get training and validation points
    train_set, validation_set=build_validation(pre_train_size, validation_points, dn)

    xtrain=train_set['x1']
    ytrain=train_set['y_noise']

    xvalid=validation_set['x1']
    yvalid=validation_set['y_noise']

    print(train_set[:40])
    print(validation_set)

    #Error and neural network vectors
    MAE=[];MSE=[];RMSE=[]        #Lists of validation errors
    MAE_t=[];MSE_t=[]; RMSE_t=[] #List of training errors
    nn_dict={} #Dictionary of neural network models

#--------------------------------------------------
    for i in range(iterations):
        #Two iterations of training a neural network (k_max=1)
        net=pyrenn.train_LM(xtrain,ytrain,nn,verbose=True,k_max=1,E_stop=1e-200)
        
        #Test NN on validation set
        yvalid_test = pyrenn.NNOut(xtrain,net) #Prediction on train
        yvalid_pred = pyrenn.NNOut(xvalid,net) #Prediction on validation set

        #Validation errors
        #--------------------------------------------------
        # MSE_i=mean_squared_error(yvalid,yvalid_pred)
        # MSE.append(MSE_i)

        # MAE_i=mean_absolute_error(yvalid,yvalid_pred)
        # MAE.append(MAE_i)

        RMSE_i=root_mean_squared_error(yvalid,yvalid_pred)
        RMSE.append(RMSE_i)
        #--------------------------------------------------

        #Training errors
        #--------------------------------------------------
        # MSE_t_i=mean_squared_error(ytrain,yvalid_test)
        # MSE_t.append(MSE_t_i)

        # MAE_t_i=mean_absolute_error(ytrain,yvalid_test)
        # MAE_t.append(MAE_t_i)
        
        RMSE_t_i=root_mean_squared_error(ytrain,yvalid_test)
        RMSE_t.append(RMSE_t_i)
        #--------------------------------------------------

        #deepcopy and save neural network to dictionary
        net_copy=copy.deepcopy(net)
        nn_dict[i]=net_copy
        
        #update neural network for next step of the loop
        nn=net
#--------------------------------------------------
        
    #Find the model with the minimum error
    #min_error_mse=min(MSE);
    min_error_rmse=min(RMSE)

    #Take indices of the elements with minimum error
    #min_err_mse_ind=MSE.index(min_error_mse);
    min_err_rmse_ind=RMSE.index(min_error_rmse)
    #--------------------------------------------------------

    #Plot errors
    #----------------------------------------------------

    #Figure settings
    #--------------------------------
    output_path_fig='../results/nn_w_validation/'
    name_fig='validation_errors_' + 'sigma_' + str(sigma) + '_' + str(function) + '_' + str(n) + '_r_' + str(realization) + '.png'
    
    #Define figure size
    cm = 1/2.54 #convert inch to cm                                  
    width = 10*cm; height=8*cm
    fig=figure(figsize=(width,height), dpi=300)

    #Fonts and sizes
    size_axis=7;size_ticks=6;size_title=5
    line_w=1;marker_s=3
    #--------------------------------
    #plt.plot(MAE, '.', markersize=6, color='blue', label='MAE validation')
    plt.plot(RMSE,'.',markersize=6,color='green',label='RMSE validation')

    #plt.plot(MAE_t, linewidth=1,linestyle='--',color='blue',label='MAE train')
    plt.plot(RMSE_t,linewidth=1,linestyle='--',color='green',label='RMSE train')
    plt.scatter(min_err_rmse_ind,min_error_rmse,s=80,marker='*',color='red',label='minimum rmse')
    #--------------------------------------------------------

    #Labels
    plt.legend(loc='best', fontsize=size_ticks)
    plt.xlabel('iterations',fontsize=size_axis);plt.ylabel('error',fontsize=size_axis)
    plt.title('%s, n=%d' % (function, n),fontsize=size_title)

    plt.tight_layout()
    plt.savefig(output_path_fig+name_fig,dpi=300)
    

    #Best nn found
    #------------------------------------------------------
    net_best=nn_dict[min_err_rmse_ind]
    xtest = dn.loc[pre_train_size:]['x1']
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    xtrain_valid=dn.loc[:pre_train_size-1]['x1']
    ytest_best = pyrenn.NNOut(xtrain_valid,net_best)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ypred_best = pyrenn.NNOut(xtest,net_best)
    
    ymodel_best=np.concatenate((ytest_best, ypred_best))
    #------------------------------------------------------

    #Save neural network
    pyrenn.saveNN(net_best, output_path + 'NN_weights_no_overfit_' + function + '_sigma_' + str(sigma) + '_rep_' + str(n) + '_r_' + str(realization) + '.csv')

    try:
        ymodel=np.append(ymodel,ymodel_best)
    except NameError:
        ymodel=ymodel_best


#Add predictions to data
d['ymodel']=ymodel

#Save updated data with model
d.to_csv( output_path + 'NN_no_overfit_' + function + '_sigma_' + str(sigma) + '_r_' + str(realization) + '.csv')

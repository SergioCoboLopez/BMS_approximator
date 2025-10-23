#10/9/2024
# This code trains neural networks in input data without overfitting.
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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error
from random import sample
import random

#Build validation set by randomly taking points from the training set 
#-------------------------------------------------------------------
def build_validation(all_training_points,len_validation,dataframe):

    pre_train_df=dataframe.loc[0:all_training_points-1]

    validation_df=dataframe.iloc[validation_points]
    train_df=pre_train_df.drop(labels=validation_points)

    return train_df, validation_df
#-------------------------------------------------------------------

#A function to train a neural network for one specific function                                   
#-------------------------------------------------------------------
def train_one_nn(iterations, neural_network, x_train, y_train, x_valid, y_valid):

    #Error and neural network vectors
    MAE_valid=[];MSE_valid=[];RMSE_valid=[]        #Lists of validation errors
    MAE_train=[];MSE_train=[]; RMSE_train=[] #List of training errors
    neural_network_dict={} #Dictionary of neural network models

    print(x_train)
#--------------------------------------------------
    for i in range(iterations):
        #Two iterations of training a neural network (k_max=1)
        net=pyrenn.train_LM(x_train,y_train,neural_network,verbose=False,k_max=1,E_stop=1e-200)
        
        #Test NN on validation set
        y_valid_test = pyrenn.NNOut(xtrain,net) #Prediction on train
        y_valid_pred = pyrenn.NNOut(xvalid,net) #Prediction on valid

        #Validation errors
        #--------------------------------------------------
        RMSE_valid_i=root_mean_squared_error(y_valid,y_valid_pred)
        RMSE_valid.append(RMSE_valid_i)
        #--------------------------------------------------

        #Training errors
        #--------------------------------------------------
        RMSE_train_i=root_mean_squared_error(y_train,y_valid_test)
        RMSE_train.append(RMSE_train_i)
        #--------------------------------------------------

        #deepcopy and save neural network to dictionary
        net_copy=copy.deepcopy(net)
        neural_network_dict[i]=net_copy
        
        #update neural network for next step of the loop
        neural_network=net

    return neural_network_dict, RMSE_valid, RMSE_train
#--------------------------------------------------

#A function to plot the validation rmse errors across iterations
#-----------------------------------------------------------------
def plot_validation_figure(RMSE_valid,RMSE_train,minimum_rmse_index,minimum_rmse,name_figure):

    #Figure settings
    #--------------------------------
    output_path_fig='../../results/nn_w_validation_nguyen/'
    name_figure=name_figure + '.png'
    print(name_figure)
    
    #Define figure size
    cm = 1/2.54 #convert inch to cm                                  
    width = 10*cm; height=8*cm
    fig=figure(figsize=(width,height), dpi=300)

    #Fonts and sizes
    size_axis=7;size_ticks=6;size_title=5
    line_w=1;marker_s=3
    
    #--------------------------------
    plt.plot(RMSE_valid,'.',markersize=6,color='green',label='RMSE validation')
    plt.plot(RMSE_train,linewidth=1,linestyle='--',color='green',label='RMSE train')
    plt.scatter(minimum_rmse_index,minimum_rmse,s=80,marker='*',color='red',label='minimum rmse')
    #--------------------------------------------------------

    #Labels
    plt.legend(loc='best', fontsize=size_ticks)
    plt.xlabel('iterations',fontsize=size_axis);plt.ylabel('error',fontsize=size_axis)
    plt.title('n=%d' %  n,fontsize=size_title)
    plt.savefig(output_path_fig+name_figure,dpi=300)
    
    return None
#-----------------------------------------------------------------


    
#Read data
#-----------------------------------------------------
random.seed(a=1111)

sigma=sys.argv[1]
realization=int(sys.argv[2])


input_path='../../data/noisy_data/nguyen/'
filename=input_path + 'NN_nguyen_sigma_' + str(sigma) + '_r_' + str(realization) +  '.csv'

d=pd.read_csv(filename)
d=d.drop(columns='Unnamed: 0')
d=d.reset_index(drop=True)

output_path='../../data/nns/nguyen/approximation/'
#-----------------------------------------------------


#train/validation size 
#----------------------------------------
n_nguyen=[1, 5, 7, 8, 10]
#n_points=int(len(d.index)/len(n_nguyen))
#----------------------------------------

#Define cross-validation
#------------------------
pre_train_fraction=3/4
validation_fraction=1/8
#------------------------

#---------------------------------

#Cross validations
iterations=300

for n in n_nguyen:
    #Read data
    dn=d[d['rep']==n]
    dn.index.name = None
    dn=dn.reset_index(drop=True)

    #Train/validation/test sets
    #---------------------------------------------------------------------------------
    n_points=int(len(dn.index))
    pre_train_size=int(n_points*pre_train_fraction)
    validation_size=int(n_points*validation_fraction)
    validation_points=sample(range(pre_train_size), k=validation_size)
    validation_points=np.sort(validation_points)

    print(n_points)
    
    train_set, validation_set=build_validation(pre_train_size, validation_points, dn)

    print(train_set)
    
    if n==10:
        ILS=2
        xtrain=np.array([train_set['x'].values,train_set['y'].values])
        ytrain=train_set['z_noise']
        xvalid=np.array([validation_set['x'].values,validation_set['y'].values])
        yvalid=validation_set['z_noise']


    else:
        ILS=1
        xtrain=train_set['x'];ytrain=train_set['z_noise']
        xvalid=validation_set['x'];yvalid=validation_set['z_noise']
    #---------------------------------------------------------------------------------
        
    #Build ANN
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    OLS=1
    NL, LS = 5, 10
    arch=[ILS] + NL*[LS] + [OLS]
    nn=pyrenn.CreateNN(arch)
    print(arch)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    #train nn 300 times and save nns 
    nn_dict, RMSE_v, RMSE_t =train_one_nn(iterations, nn, xtrain, ytrain, xvalid, yvalid)
#--------------------------------------------------
        
    #Find the model with the minimum error
    min_error_rmse=min(RMSE_v)
    print(min_error_rmse)

    #Take indices of the elements with minimum error
    min_err_rmse_ind=RMSE_v.index(min_error_rmse)
    #--------------------------------------------------------

    #Plot validation errors
    #---------------------------------------------------
    name_fig='validation_errors_sigma_' + str(sigma) + '_' + \
    str(n) + '_r_' + str(realization)

    plot_validation_figure(RMSE_v, RMSE_t, min_err_rmse_ind, min_error_rmse,name_fig)
    #----------------------------------------------------
    
    #Best nn found
    #------------------------------------------------------
    net_best=nn_dict[min_err_rmse_ind]

    if n==10:

        xtest=np.array([dn.loc[pre_train_size:]['x'].values, dn.loc[pre_train_size:]['y'].values])
        
        xtrain_valid=np.array([dn.loc[:pre_train_size-1]['x'].values, dn.loc[:pre_train_size-1]['y'].values])

    else:
        xtest = dn.loc[pre_train_size:]['x']
        xtrain_valid=dn.loc[:pre_train_size-1]['x']

    ytest_best = pyrenn.NNOut(xtrain_valid,net_best)
    ypred_best = pyrenn.NNOut(xtest,net_best)
    ymodel_best=np.concatenate((ytest_best, ypred_best))
    #------------------------------------------------------

    #Save neural network
    pyrenn.saveNN(net_best, output_path + 'NN_weights_no_overfit_sigma_' + str(sigma) + '_rep_' + str(n) + '_r_' + str(realization) + '.csv')

    try:
        ymodel=np.append(ymodel,ymodel_best)
    except NameError:
        ymodel=ymodel_best


#Add predictions to data
d['zmodel']=ymodel

#Save updated data with model
d.to_csv( output_path + 'NN_no_overfit_sigma_' + str(sigma) + '_r_' + str(realization) + '.csv')


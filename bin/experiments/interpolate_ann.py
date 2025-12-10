#9/12/2025

import pyrenn
import numpy as np
import copy
import pandas as pd
import sys
import os

#CAUTION!! Due to conflicting versions of python libraries, you need to create and activate a virtual environment to run this code.
#If you don't have a virtual environment in your computer, create one with numpy 1.20 and pandas 1.2.5 so that pyrenn can work without conflicting with numpy.
#To activate the local virtual environment, I type source ~/entorno/bin/activate, with "entorno" being the name of my local virtual environment. To deactivate it, type "deactivate"


#External arguments
function=sys.argv[1] #tanh, leaky_ReLU
realization=sys.argv[2]
sigma=sys.argv[3]
step=sys.argv[4]
network_layers=sys.argv[5]      #ILS-NL-LS-OLS

network_layers=network_layers.split('-')
network_layers=[int(i) for i in network_layers]

ILS=network_layers[0];NL=network_layers[1]
LS=network_layers[2];OLS=network_layers[3]

#ANN architecture                                                                                  
#--------------------------------------------------                                                
architecture=[ILS] + NL*[LS] + [OLS]
net_architecture='ILS%s_NL%s_LS%s' % (ILS, NL, LS)
#--------------------------------------------------


input_path = '../../data/alternative_experiments/%s/' % net_architecture
                                                         
trained_nn_path = '../../data/alternative_experiments/%s/nns/%s/approximations/' % (net_architecture,str(step))
output_path= '../../data/alternative_experiments/%s/nns/%s/interpolations/' % (net_architecture,str(step))

try:
    os.makedirs(output_path)
except FileExistsError:
    # directory already exists                                                                     
    pass


#Read high resolution data to interpolate
#----------------------------------
interpolation_step=float(step)/5
print(interpolation_step)
file_name_d=input_path + 'NN_%s_NREP_10_%s_step_%s_interpolation_data.csv' % (function, net_architecture, interpolation_step)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

d=pd.read_csv(file_name_d)
d=d.drop(columns='Unnamed: 0')
d=d[(d['x1'] >= -2.0) & (d['x1']<=2.0)]
d=d.reset_index(drop=True)

n_functions=int(d['rep'].max()) #Number of functions in dataset

for n in range(n_functions + 1):

    dn=d[d['rep']==n]
    dn=dn.reset_index(drop=True)
    #----------------------------------

    #Read neural network
    #----------------------------------
    file_name_nn=trained_nn_path + 'NN_weights_no_overfit_' + function + '_sigma_' + str(sigma) +\
        '_rep_' + str(n) + '_r_' + str(realization) + '.csv'

    nn=pyrenn.loadNN(file_name_nn)
    #----------------------------------

    #predictions of nn
    x_tot=dn['x1']
    ymodel=pyrenn.NNOut(x_tot, nn)
    
    #Save results
    #----------------------------------
    try:
        ymodels_all=np.append(ymodels_all,ymodel)
    except NameError:
        ymodels_all=ymodel
    #----------------------------------

#Save updated data with model
d['ymodel']=ymodels_all

d.to_csv(output_path + 'NN_no_overfit_%s_sigma_%s_r_%s_interpolation.csv' %(function, sigma, realization))



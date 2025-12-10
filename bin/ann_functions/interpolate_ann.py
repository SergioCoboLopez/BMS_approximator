import pyrenn
import numpy as np
import copy
import pandas as pd
import sys

#CAUTION!! Due to conflicting versions of python libraries, you need to create and activate a virtual environment to run this code.                                                                     
#If you don't have a virtual environment in your computer, create one with numpy 1.20 and pandas 1.2.5 so that pyrenn can work without conflicting with numpy.  
#To activate the local virtual environment, I type source ~/entorno/bin/activate, with "entorno" being the name of my local virtual environment. To deactivate it, type "deactivate"

resolution='0.5x' #1x, 2x, 0.5x, 4e-3x
resolutions={'0.5x':'0.02', '1x':'0.01', '2x': '0.005', '4e-3x':'0.0008' }

function=sys.argv[1] #tanh, leaky_ReLU
realization=sys.argv[2]
sigma=sys.argv[3]

#Read high resolution data to interpolate
#----------------------------------
input_path= '../../data/generative_data/'
file_name_d=input_path + 'NN_function_'+function + '_NREP_10_res_' + str(resolutions[resolution]) + '_interpolation_data.csv'

output_path='../../data/nns/' + resolution + '_resolution/inter_extrapolation/'


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
    nn_path='../../data/nns/' + str(resolution) + '_resolution/approximation/'
    file_name_nn=nn_path + 'NN_weights_no_overfit_' + function + '_sigma_' + str(sigma) +\
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
d.to_csv( output_path + 'NN_no_overfit_' + function + '_sigma_' + str(sigma) + '_r_' + str(realization) + '_res_'  + resolutions[resolution] + '.csv')
# d.to_csv('../data/inter_extrapolate_nns/'+ 'NN_no_overfit_' + function + '_sigma_' + str(sigma) + '_r_' + str(realization) + '_res_0.01.csv')

#2025-05-30. This code builds BMS traces from ANN-generated data and trains with all points in the dataset. The purpose of this code is to generate a specific figure for the tutorial notebook.

import sys
import pandas as pd
import numpy as np
from datetime import date
from copy import deepcopy
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
import os

#Set the current working directory in Python
os.chdir('/export/home/shared/Projects/ANN/Sergio/BMS_approximator/bin')

# Command-line arguments
# -------------------------------------------------------------------
n = int(sys.argv[1])    #function number (0 to 9)
function=sys.argv[2]    #tanh, leaky_ReLU
sigma=sys.argv[3]       #mean of gaussian noise
realization=sys.argv[4] #Gaussian noise realization
runid=sys.argv[5]
resolution='1x' #0.5x, 1x, 2x, 4e-3

NPAR = 10 #number of parameters of prior
# -------------------------------------------------------------------
# Read the ANN-generated data
resolution='1x'
resolutions={'0.5x':'0.1', '1x':'0.05' , '2x': '0.025' , '4e-3x':'0.004' }

d=pd.read_csv('../data/noisy_data/' + resolution + '_resolution/NN_' + function + '_sigma_' + str(sigma) + '_r_' + str(realization)  + '_res_' + resolutions[resolution] + '.csv')

filename = 'BMS_test_full_train_' + function + '_n_' + str(n) + '_sigma_' + str(sigma) + '_r_' + str(realization) 
#-------------------------------------------------------------------
    
#Take specific function and -2/+2 interval
#---------------------------------------
d=d[d['rep']==n]
d=d[(d['x1'] >= -2.0) & (d['x1']<=2.0)];d=d.reset_index(drop=True)
#---------------------------------------

#Train set
#---------------------------------------
n_points=int(len(d.index))
#train_fraction=3/4
train_fraction=1
train_size=int(n_points*train_fraction)
d=d.loc[0:train_size-1]
#---------------------------------------


# Instantiate a Tree from the desired string
#---------------------------------------
X, y = d[['x1']], d['y_noise']
XLABS = list(X.columns)
print(d)
print(y)
#---------------------------------------

# BMS
# -------------------------------------------------------------------
import sys
sys.path.append('./no_degeneracy')
sys.path.append('./no_degeneracy/Prior')
from mcmc import *
from parallel import *
from fit_prior import read_prior_par


# Choose and initialize priors and temperatures
if NPAR==10:
    prior_par = read_prior_par('no_degeneracy/Prior/final_prior_param_sq.named_equations.nv1.np10.2017-10-18 18:07:35.089658.dat')

elif NPAR==20:
    prior_par = read_prior_par('no_degeneracy/Prior/final_prior_param_sq.named_equations.nv1.np20.maxs200.2024-05-10 162907.551306.dat')

# Set the temperatures for the parallel tempering
Ts = [1] + [1.04**k for k in range(1, 20)]

# REPEAT NREP TIMES
NSTEP = 50000

# Initialize the parallel machine scientist
pms = Parallel(
    Ts,
    variables=XLABS,
    parameters=['a%d' % i for i in range(8)],
    x=X, y=y,
    prior_par=prior_par,
)

description_lengths, mdl, mdl_model = [], np.inf, None


outf = open('../data/tutorial/'+ filename + '_trace_' + str(NSTEP) + '_prior_' + str(NPAR) + '.csv', 'w')

outf.close()

# MCMC
for i in tqdm(range(NSTEP)):
    # MCMC update
    pms.mcmc_step() # MCMC step within each T
    pms.tree_swap() # Attempt to swap two randomly selected consecutive temps
    # Add the description length to the trace
    description_lengths.append(pms.t1.E)

    outf = open('../data/tutorial/'+ filename + '_trace_' + str(NSTEP) + '_prior_' + str(NPAR) +  '.csv', 'a')
    print(
        ';'.join([
        str(kk) for kk in [i, pms.t1.E, pms.t1.pr(show_pow=True),
                               pms.t1.par_values, pms.t1.sse, pms.t1.bic, pms.t1.EP]
              ]),file=outf)
    
    outf.close()
    # Check if this is the MDL expression so far
    if pms.t1.E < mdl:
        mdl, mdl_model = pms.t1.E, deepcopy(pms.t1)
        

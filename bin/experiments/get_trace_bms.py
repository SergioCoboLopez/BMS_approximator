#2025-05-30. This code builds BMS traces from ANN-generated data
#something's wrong with this code. It generates traces with a 'None' factor in the equations.

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
#os.chdir('/home/sees/cobo/simulations_ANN_BMS/experiments') #llac cluster
os.chdir('/export/home/shared/Projects/ANN/Sergio/BMS_approximator/bin/experiments') #local

# Command-line arguments
# -------------------------------------------------------------------
n = int(sys.argv[1])    #function number (0 to 9)
function=sys.argv[2]    #tanh, leaky_ReLU
sigma=sys.argv[3]       #mean of gaussian noise
realization=sys.argv[4] #Gaussian noise realization
step=sys.argv[5] #0.1, 0.05,0.025,0.004
network_layers=sys.argv[6] #ILS-NL-LS-OLS
# -------------------------------------------------------------------

print(step)

#ANN architecture
#--------------------------------------------------
network_layers=network_layers.split('-')
network_layers=[int(i) for i in network_layers]

ILS=network_layers[0];NL=network_layers[1]
LS=network_layers[2];OLS=network_layers[3]
net_architecture='ILS%s_NL%s_LS%s' % (ILS, NL, LS)
#--------------------------------------------------

# Read the ANN-generated data
input_path='../../data/alternative_experiments/%s/noisy_data/%s/' % (net_architecture, step)
filename='NN_%s_%s_sigma_%s_r_%s_step_%s.csv' % (function, net_architecture, sigma, realization, step )
d=pd.read_csv(input_path + filename)
#-------------------------------------------------------------------
    
#Take specific function and -2/+2 interval
#---------------------------------------
d=d[d['rep']==n]
d=d[(d['x1'] >= -2.0) & (d['x1']<=2.0)];d=d.reset_index(drop=True)
#---------------------------------------

#Train set
#---------------------------------------
n_points=int(len(d.index))
train_fraction=3/4
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
sys.path.append('../no_degeneracy')
sys.path.append('../no_degeneracy/Prior')
from mcmc import *
from parallel import *
from fit_prior import read_prior_par

NPAR = 10 #number of parameters of prior
# Choose and initialize priors and temperatures
if NPAR==10:
    prior_par = read_prior_par('../no_degeneracy/Prior/final_prior_param_sq.named_equations.nv1.np10.2017-10-18 18:07:35.089658.dat')

elif NPAR==20:
    prior_par = read_prior_par('../no_degeneracy/Prior/final_prior_param_sq.named_equations.nv1.np20.maxs200.2024-05-10 162907.551306.dat')

# Set the temperatures for the parallel tempering
Ts = [1] + [1.04**k for k in range(1, 20)]

# REPEAT NSTEP times
NSTEP = 50000

# Initialize the parallel machine scientist
pms = Parallel(
    Ts,
    variables=XLABS,
    parameters=['a%d' % i for i in range(NPAR)],
    x=X, y=y,
    prior_par=prior_par,
)


description_lengths, mdl, mdl_model = [], np.inf, None

output_path='../../data/alternative_experiments/%s/MStraces/%s/' % (net_architecture, step)

try:
    os.makedirs(output_path)
except FileExistsError:
    # directory already exists                                       
    pass


filename='BMS_%s_%s_n_%s_sigma_%s_r_%s_trace_%s_prior_%s.csv' % \
    (function,net_architecture, n, sigma, realization, str(NSTEP), str(NPAR))

outf = open(output_path+filename, 'w')
outf.close()

# MCMC
for i in tqdm(range(NSTEP)):
    # MCMC update
    pms.mcmc_step() # MCMC step within each T
    pms.tree_swap() # Attempt to swap two randomly selected consecutive temps
    # Add the description length to the trace
    description_lengths.append(pms.t1.E)

    outf = open(output_path+filename, 'a')

    print(
        ';'.join([
        str(kk) for kk in [i, pms.t1.E, pms.t1.pr(show_pow=True),
                               pms.t1.par_values, pms.t1.sse, pms.t1.bic, pms.t1.EP]
              ]),file=outf)
    
    outf.close()
    # Check if this is the MDL expression so far
    if pms.t1.E < mdl:
        mdl, mdl_model = pms.t1.E, deepcopy(pms.t1)
        

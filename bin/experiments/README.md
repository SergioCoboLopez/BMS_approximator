28/11/2025

This folder contains all necessary codes to run experiments with different neural network architectures.


## Step 1: `generate_ANN_datasets.ipynb`

This notebook is intended to be executed **only once**, as it generates all datasets required for the experiments

In the **second cell**, the user can select the following parameters:

-Network architecture:

 `ILS`: input layer size, NL: number of layers and LS: number of nodes per layer or layer size

 `NREP`: Number of functions generated per experiment and activation function

 `xmax`: limits of x in the functions. By default, symmetric limits are assumed.

 `steps`: a vector with the resolutions (steps) considered. Each step indicates the distance between one point and the next.

 `activation functions`: `tanh` and `leaky_ReLU` are considered by default. The `ReLU` function can be also selected. Additional functions need to be explicitly programmed by the user.

Upon execution, the **third cell** generates and saves the dataframes with the functions in the corresponding resolutions. The code creates a new subfolder at `data/alternative_experiments/` with the new data. The subfolder will be named after the ANN architecture: `ILS_<X>_NL_<Y>_LS_<Z>/`

In the **fourth cell** the user can add Gaussian noise to the original functions and generate the corresponding dataframes. The cell allows for selecting the following parameters:

   `sigma_max`: maximum standard deviation of the Gaussian noise.
   `sigma_step`: increment in values of sigma.
   `r`: number of realizations of noise (3 is selected by default)

The dataframes with the noisy data are saved in the subfolder `noisy_data/`. With these configuration, we generate `activation_functions*NREP*N_sigma_steps*realizations` = $2 \cdot 10 \cdot 11 \cdot3=660$ datasets per each resolution..


The last two cells are intended to generate figures of the functions, but are currently being developed.


## Step 2: `train_anns.py` and `train_multiple_anns_script.sh`

The next step is to train ANNs using the observational data generated in Step 1. This is accomplished using the two codes mentioned above. While `train_anns.py` represents the core of this step, `train_multiple_anns_script.sh` is a shell script used to train multiple dataframes sequentially.

### `train_anns.py`

This code creates and trains `NREP` (10, by default) neural networks using the Levenberg-Marquadt algorithm from the [pyrenn](https://pyrenn.readthedocs.io/en/latest/index.html) Python library.
The network is trained with the observed data generated in the previous step. 
The user needs to pass four external arguments corresponding to the observed data: activation function, sigma (level of noise), realization (of noise), and step (resolution of the data). These arguments are used to select the proper input file.
In addition, the user needs to input the parameters of the neural network architecture. (future versions of this repository aim at automatically setting the architecture from the observational data)
The observed data is constrained to x=[-2,2], the training set is 3/4 of the total data, the validation is 1/8 of the total data, and the training takes 300 iterations. However, the user may change any of these parameters. The validation data are randomly sampled (from a uniform distribution) from the training set.
After each iteration, the code calculates the validation and training errors with the Root Mean Squared Error (RMSE). The RMSE is calculated using the [scikit-learn](http://scikit-learn.org/stable/) Python library. The errors from all iterations are stored in a dictionary.
After the training is finished, the code selects the weight distribution (the iteration) with the minimum validation error. This is done to prevent [overfitting](https://en.wikipedia.org/wiki/Overfitting) .
After that, the code might plot the validation and training errors and save the figure.
Finally, the code generates the predictions of the optimal weight distribution (the best neural network, so to speak) on the test set. Finally, it saves these data on csv file that includes the observed data as well. The data is stored in `data/alternative_experiments/ILS_<X>_NL_<Y>_LS_<Z>/`

### `train_multiple_anns_script.sh`

On its default version, this script calls `train_anns.py` for values of $\sigma$ between $0.0$  and $0.2$ with $\Delta \sigma=0.02$, for both activation functions $\tanh$ and $Leaky_ReLU$ and for a given value of the step function. The user might change the default parameters at will.


### Step 3: `get_trace.py`

This codes trains the Bayesian Machine Scientist (BMS) to get the most plausible equation that explains the observed data generated in Step 1. Although this code can perfectly run on a laptop or desktop computer, we have used supercomputers for our experiments, because each resolution value involves 660 simulations. On top of that, the BMS can get stuck on very long equations, implying that the code can get stuck and the user needs to re-run it again.

The user needs to pass five external arguments that define a file with observed data. These arguments are:


 `n`: function number, by default ranging from 0 to 9.
 `function`: by default `tanh` or `leaky_ReLu`
 `sigma`: level of Gaussian noise. By default, 0.0 to 0.2 in steps of $\Delta \sigma = 0.02$
 `realization`: the realization of Gaussian noise for each value of `sigma`. By default 0,1, or 2.
 `step`: resolution of the dataset. By default `0.1`, `0.05`, `0.025`, and `0.004`

The user might define, in the initial step, any other values of these parameters, but they will need to modify the codes accordingly.
After that, the user needs to manually introduce the network architecture for which they want to train the BMS:


 `ILS` : Input layer size
 `NL` : Number of (hidden) layers
 `LS` : (hidden) layer size
 `OLS`: output layer size

(future versions of this repository aim at automatically setting the architecture from the observational data)

Next, the code reads the corresponding file with the data generated on step 1 and the prediction of the ANN generated on step 2 and takes only data constrained to x=[-2,2]. It then defines a training set of 3/4 of the boserved data.

After that, the code defines the name of the variables x and y and the number of parameters used. It then reads the corresponding prior file.

The following steps are setting the temperatures for the parallel tempering, setting the number of MCMC steps ( 50000, by default), and initialize the parallel machine scientist.

Then the code performes the MCMC steps.
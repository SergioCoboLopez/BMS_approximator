28/11/2025

This folder contains all necessary codes to run experiments with different neural network architectures.


### Step 1: `generate_ANN_datasets.ipynb`

This notebook is intended to be executed **only once**, as it generates all datasets required for the experiments

In the **second cell**, the user can select the following parameters:

-Network architecture:

\* `ILS`: input layer size, NL: number of layers and LS: number of nodes per layer or layer size
\* `NREP`: Number of functions generated per experiment and activation function
\
\* `xmax`: limits of x in the functions. By default, symmetric limits are assumed.
\* `steps`: a vector with the resolutions (steps) considered. Each step indicates the distance between one point and the next.
\* `activation functions`: `tanh` and `leaky_ReLU` are considered by default. The `ReLU` function can be also selected. Additional functions need to be explicitly programmed by the user.

Upon execution, the **third cell** generates and saves the dataframes with the functions in the corresponding resolutions. The code creates a new subfolder at `data/alternative_experiments/` with the new data. The subfolder will be named after the ANN architecture: `ILS_<X>_NL_<Y>_LS_<Z>/`

In the **fourth cell** the user can add Gaussian noise to the original functions and generate the corresponding dataframes. The cell allows for selecting the following parameters:

   `sigma_max`: maximum standard deviation of the Gaussian noise.
   `sigma_step`: increment in values of sigma.
   `r`: number of realizations of noise (3 is selected by default)

The dataframes with the noisy data are saved in the subfolder `noisy_data/`

The last two cells are intended to generate figures of the functions, but are currently being developed.


### Step 2: `train_anns.py` and `train_multiple_anns_script.sh`

The next step is to train ANNs using the observational data generated in Step 1. This is accomplished using the two codes mentioned above. While `train_anns.py` represents the core of this step, `train_multiple_anns_script.sh` is shell script used to train multiple dataframes sequentially.

`train_anns.py`: This code creates and trains NREP neural networks using the Levenberg-Marquadt algorithm from the [pyrenn](https://pyrenn.readthedocs.io/en/latest/index.html) Python library.
		 The network is trained with the observed data generated in the previous step. 
		 The user needs to pass four external arguments corresponding to the observed data: activation function, sigma (level of noise), realization (of noise), and step (resolution of the data). These arguments are used to select the proper input file.
		 In addition, the user needs to input the parameters of the neural network architecture. (future versions of this repository aim at automatically setting the architecture from the observational data)
		 The observed data is constrained to x=[-2,2], the training set is 3/4 of the total data, the validation is 1/8 of the total data, and the training takes 300 iterations. However, the user may change any of these parameters.
		 
		 
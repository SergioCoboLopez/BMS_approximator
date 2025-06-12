This folder contains data files with the prediction of neural networks on the two different problems tackled in the paper:

1. Approximate, interpolate, and extrapolate nguyen (analytical) functions. Interpolation and extrapolation are in the same subfolder.
2. Approximate, interpolate, and extrapolate ann-generated functions. Interpolation and extrapolation are in the same subfolder.


The predictions were made with neural networks with the following architecture:

Input Layer Size: 1
Number of layers: 5
Layer Size: 10
Output Layer Size:1
Activation function: 'tanh'

The neural networks were trained in 3/4 of the data, validated on 1/8 of the points and tested on 2/10 of the points.

Besides those predictions, the data files are identical to the correspoinding files on the folder 'noisy_data/'

We used the library [pyrenn](https://pyrenn.readthedocs.io/en/latest/) and the [Levenberg–Marquardt algorithm](https://pyrenn.readthedocs.io/en/latest/train.html#pyrenn.pyrenn.train_LM). Note that the latest version of this library is from 2018.
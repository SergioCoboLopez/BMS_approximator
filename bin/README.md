4/12/2025

This folder contains all the codes needed to generate the results in the paper and to run experiments with alternative network architectures.
Currently, there are four different folders.

`ann_functions/` contains all the codes needed to generate the results to approximate ann-generated functions. For codes that reproduce (partially or totally) the figures in the manuscript or a tutorial on the project workflow, please refer to `../notebooks/`

`nguyen_functions/` contains all the codes needed to generate the results to approximate analytical nguyen functions.

`experiments/` contains all the codes needed to approximate ann-generated functions with architectures that are different from the results of the paper.

`no_degeneracy/` contains the Bayesian Machine Scientist, which is a repository in itself. The original repo is [here](https://bitbucket.org/rguimera/machine-scientist/src/no_degeneracy/) and the paper is [here](https://www.science.org/doi/10.1126/sciadv.aav6971).

`slurm_scripts/` contains three slurm scripts to run the code `get_trace.py` on a cluster environment using the [slurm](https://slurm.schedmd.com/documentation.html) protocol.






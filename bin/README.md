4/12/2025


This folder contains all the code needed to generate the results in the paper and to run experiments with alternative network architectures.
Currently, there are four main subfolders.

`ann_functions/` contains all the code needed to generate the results for approximating ANN‑generated functions. For code that (partially or fully) reproduces the figures in the manuscript, or a tutorial on the project workflow, please refer to `../notebooks/`

`nguyen_functions/` contains all the code needed to generate the results for approximating analytical Nguyen functions.

`experiments/`  contains all the code needed to approximate ANN‑generated functions with architectures that differ from those used in the paper.

`no_degeneracy/` contains the Bayesian Machine Scientist, which is a standalone repository. The original repository is [here](https://bitbucket.org/rguimera/machine-scientist/src/no_degeneracy/) and the paper is [here](https://www.science.org/doi/10.1126/sciadv.aav6971).

`slurm_scripts/` contains three scripts to run the code `get_trace.py` on a cluster environment using the [Slurm](https://slurm.schedmd.com/documentation.html) protocol.






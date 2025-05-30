#!/bin/bash
#This script is intended to run a whole set of ann-generated functions. I associate the function number to the task id variable.

#SBATCH --job-name=test              #Name of the job you are submitting   
#SBATCH --output=logs/n_%A_%a.out    # %A=job ID, %a=array index. Print script outputs to file
#SBATCH --error=logs/array_%A_%a.err # Print errors to a file
#SBATCH --array=0-9                  # Run 10 jobs with indices 0-9. These match names of variable
#SBATCH --ntasks=1                   # Each job runs 1 task. ?? Maybe I could run two variables?
#SBATCH --cpus-per-task=1                                                                          
#SBATCH --mem=1GB                    #Memory allowed per job (or task?)   
#SBATCH --time=1-12:00:00            #Max time allowed

echo "This is $SLURM_ARRAY_TASK_ID running"


# Load required modules (if needed)                                                                
# module load python/3.8                                                                           

#variables and parameters
fun="tanh" #tanh, leaky_ReLU
sigma=0.0  #0.0 to 0.2
r=0        #0,1,2

#Your job commands go here. The first external argument matches the job id.
python3 get_trace.py ${SLURM_ARRAY_TASK_ID} tanh 0.02 0 0 

#!/bin/bash

#I use this script when some of the runs in the slurm_script.sh did not finish and get stuck. Using the script check_traces in data/MSTraces I can check which runs made it to 50k steps and which did not, generating a txt file with the unfinished runs. I can copy paste the unfinished runs in the associative array below and run the script again.


#SBATCH --job-name=test
#SBATCH --output=logs/n_%A_%a.out    # %A=job ID, %a=array index
#SBATCH --error=logs/array_%A_%a.err
#SBATCH --array=11-28                # Run 10 jobs with indices 0-9
#SBATCH --ntasks=1                  # Each job runs 1 task
#SBATCH --cpus-per-task=1                                                                          
#SBATCH --mem=1GB                                                                                  
#SBATCH --time=1-12:00:00

#declare an associative array
declare -A job_array=([11]="2 tanh 0.0 1"
[12]="3 tanh 0.0 1"
[13]="4 tanh 0.0 1"
[14]="5 tanh 0.0 1"
[15]="6 tanh 0.0 1"
[16]="7 tanh 0.0 1"
[17]="8 tanh 0.0 1"
[18]="9 tanh 0.0 1"
[19]="0 tanh 0.0 2"
[20]="1 tanh 0.0 2"
[21]="2 tanh 0.0 2"
[22]="3 tanh 0.0 2"
[23]="4 tanh 0.0 2"
[24]="5 tanh 0.0 2"
[25]="6 tanh 0.0 2"
[26]="7 tanh 0.0 2"
[27]="8 tanh 0.0 2"
[28]="9 tanh 0.0 2")

echo "This is running ${job_array[$SLURM_ARRAY_TASK_ID]}"


# Load required modules (if needed)
# module load python/3.8

#Your job commands go here
python3 get_trace.py ${job_array[$SLURM_ARRAY_TASK_ID]} 0

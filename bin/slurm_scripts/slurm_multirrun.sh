#!/bin/bash
#Some runs do have a tendency to get stuck over and over again. In that case, I use this script to run the same job multiple times and I get whatever jobs made it to 50000 steps. This would be the last script I would use.

#SBATCH --job-name=test_runs
#SBATCH --output=logs/n_%A_%a.out    # %A=job ID, %a=array index
#SBATCH --error=logs/array_%A_%a.err                                   
#SBATCH --array=1-5                  # Run 10 jobs with indices 1-5
#SBATCH --ntasks=1                   # Each job runs 1 task
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB                    #More memory in case the job gets stuck
#SBATCH --time=1-12:00:00                                                              

# Load required modules (if needed)
# module load python/3.8

n=0
fun="tanh"
sigma=0.0
r=0

#Your job commands go here
python3 get_trace.py ${n} ${fun} ${sigma} ${r} ${SLURM_ARRAY_TASK_ID}

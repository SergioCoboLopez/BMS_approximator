#!/bin/python3
import subprocess
import sys
import numpy as np

BASE_PATH = '/export/home/shared/Projects/ANN/Sergio'
NODES_PER_TASK = 1
PROC_PER_TASK = 1
USER_MAIL = 'alejandro.horrillo@urv.cat'
JOB_NAME = 'nn_00'  #function_sigmanr
OUTPUT_PATH = BASE_PATH + '/logs_python_cluster.txt'
COMMAND_PATH = BASE_PATH + '/venv/bin/python3'

SCRIPT_PATH = BASE_PATH + '/BMS_approximator/bin/nguyen_functions/get_trace_nguyen.py'

# Genera una lista de strings que contiene los argumentos para el proceso.

n=10             #1,5,7,8,10
sigma=0.0

#jobs=[ [n, sigma, r, 15] for r in range(0,3)]  #run different rs

jobs=[ [n, sigma, 2, i] for i in range(16,21)] #run different ids

print(jobs)


def generate_arguments():
        
        args= jobs
        return args

# Construye el srun.
def build_command(arg):
        print(arg)
        base_command = f'srun --oversubscribe --ntasks={NODES_PER_TASK} --cpus-per-task=1 --mem=3G --mail-user {USER_MAIL} -J {JOB_NAME}_{arg[0]}_{arg[1]} --mail-type=ALL --error={OUTPUT_PATH} --output={OUTPUT_PATH} '

        script_command = f'{COMMAND_PATH} {SCRIPT_PATH} {arg[0]} {arg[1]} {arg[2]} {arg[3]}'
        
        #multiple runs
#        script_command = f'{COMMAND_PATH} {SCRIPT_PATH} {arg[0]} {arg[1]} {arg[2]} {arg[3]} {arg[4]}'
        return base_command + script_command

def main():
	args = generate_arguments()
	if len(args) == 0:
		print("no arguments passed")
		command = build_command('')
		process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
		#output, error = process.communicate()
	else:
		for arg in args:
			command = build_command(arg)
			process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
			#output, error = process.communicate()

if __name__ == "__main__":
	main()


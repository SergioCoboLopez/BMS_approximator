#!/bin/bash
                                                                    
#Show number of iterations of a batch of traces                     

n="$1"
function="$2" #tanh, leaky_ReLU
sigma="$3"
r="$4"
#id=2

for i in BMS_full_train_${function}_n_${n}_sigma_${sigma}_r_${r}_id_{0..10}_trace_50000_prior_10.csv;
    do echo $i; tail -n1 $i | cut -d';' -f1; stat -c=%y $i;
done

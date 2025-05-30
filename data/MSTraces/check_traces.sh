#!/bin/bash                                                                                        

#This script tells you how many traces on a set (30 simulations) made it to 50 k steps, one was the last time a change was observed, and it generates a list of unfinished traces that you can use to relaunch process to the cluster using the script bin/slurm_scripts/slurm_array.sh .

function="$1" #tanh, leaky_ReLU
sigma="$2"
res='1x'

declare -A resolutions=(['0.5x']='0.1'  ['1x']='0.05'  ['2x']='0.0025'  ['0.5x']='0.004')  



steps=49999

counter=1
echo "FINISHED"
echo "index, n, function, sigma, r"
for r in {0..2}
do for n in {0..9}
   do file=${res}_resolution/BMS_${function}_n_${n}_sigma_${sigma}_r_${r}_id_0_res_${resolutions[$res]}_trace_50000_prior_10.csv
#   do file=BMS_${function}_n_${n}_sigma_${sigma}_r_${r}_id_0_res_${resolutions[$res]}_trace_50000_prior_10.csv
   len_trace="$(tail -n1 $file | cut -d';' -f1)";
   if (( $len_trace == $steps))
       then
       echo $counter $n ${function}, ${sigma}, ${r}
       counter=$((counter+1))
   fi

done
printf "\n"
done


counter_not=1
echo "NOT FINISHED"
echo "index, n, function, sigma, r"
for r in {0..2}
do for n in {0..9}
   do file=BMS_${function}_n_${n}_sigma_${sigma}_r_${r}_id_0_res_${resolutions[$res]}_trace_50000_prior_10.csv
   len_trace="$(tail -n1 $file | cut -d';' -f1)";
   if (( $len_trace != $steps))
       then
       echo ,[${n}, ${function}, ${sigma}, ${r}]; stat -c=%y $file
       echo $len_trace
       echo [${n}, \'${function}\', ${sigma}, ${r}, 0] >> missing_traces.txt
       echo $file
       counter_not=$((counter_not+1))
   fi

done
done

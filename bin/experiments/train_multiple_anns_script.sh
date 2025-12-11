#!/bin/bash

declare -a sigmas=('0.0' '0.02' '0.04'  '0.06' '0.08' '0.1' '0.12' '0.14' '0.16' '0.18' '0.2')
#declare -a sigmas=('0.18' '0.2')

declare -a functions=('tanh' 'leaky_ReLU')
#declare -a functions=('leaky_ReLU')
layers='1-7-12-1'

step=0.1

for fun in "${functions[@]}";
do for sigma in  "${sigmas[@]}";
      do for realization in {0..2};
            do echo $s $r $fun;
            python3 train_anns.py $fun $sigma $realization $step $layers;
         done
done
done

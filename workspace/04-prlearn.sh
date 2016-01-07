#!/bin/bash

##
## This script will choose best pooling regions
##

for ds in 'liberty' 'notredame' 'yosemite'
  do

  for mu in 0.0025 0.005 0.010 0.015 0.020 0.025 0.030 0.035 0.040 0.045
    do

    for gamma in 0.010 0.015 0.020 0.025 0.050 0.075 0.100 0.125 0.150 0.175
      do

      result=$(cat pr-learn/$ds-$mu-$gamma-pr.log | grep "Step\: 50000000")

      if [ ${#result} -eq 0 ]
      then
        rm -f pr-learn/$ds-$mu-$gamma-pr.h5
        rm -f pr-learn/$ds-$mu-$gamma-pr.log
        echo "Compute dataset [$ds] mu: $mu gamma: $gamma"
        ../bin/pr-learn -iters 50000000 -mu $mu -gamma $gamma \
            filters.h5 \
            distances/$ds-dist.h5 \
            pr-learn/$ds-$mu-$gamma-pr.h5 \
        | tee pr-learn/$ds-$mu-$gamma-pr.log
      else
        echo "Learning PR for [pr-learn/$ds-$mu-$gamma-pr.h5] already done."
      fi

    done

  done

done

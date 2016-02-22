#!/bin/bash

##
## This script will learn pooling regions
##

for ds in "yosemite" "notredame" "liberty"
  do

  for mu in 0.001 0.002 0.003 0.004 0.005 0.010 0.015 0.020 0.025 0.030 0.035 0.040 0.045 0.050 0.055 0.060
    do

    for gamma in 0.005 0.010 0.015 0.020 0.025 0.030 0.040 0.050 0.075 0.100 0.125 0.150 0.175 0.200 0.225 0.250
      do

      result=$(cat pr-learn/logging/$ds-$mu-$gamma-pr.log | grep "\: 5000000  Loss\:")

      if [ ${#result} -eq 0 ]
      then
        rm -f pr-learn/$ds-$mu-$gamma-pr.h5
        rm -f pr-learn/logging/$ds-$mu-$gamma-pr.log
        echo "Compute dataset [$ds] mu: $mu gamma: $gamma"
        ../bin/pr-learn -iters 50000000 -mu $mu -gamma $gamma \
            filters.h5 \
            distances/$ds-dist.h5 \
            pr-learn/$ds-$mu-$gamma-pr.h5 \
        | tee pr-learn/logging/$ds-$mu-$gamma-pr.log
      else
        echo "Learning PR for [pr-learn/$ds-$mu-$gamma-pr.h5] already done."
      fi

    done

  done

done

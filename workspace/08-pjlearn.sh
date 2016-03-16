#!/bin/bash

##
## This script will learn projection matrix
##

for f in `find . -name *-unproj.h5 | sed -e 's|\.\/||g' -e 's|-unproj.h5||g' | grep -v rank`
  do

  for mu in 0.0001 0.0005 0.0010 0.0020 0.0030
    do

    for gamma in 0.025 0.050 0.100 0.150 0.200 0.250 0.500 0.750 1.000
      do

      ds=${f##*/}
      result=$(cat pj-learn/logging/$ds-$mu-$gamma-pj.log | grep "\: 50000  Loss\:")

      if [ ${#result} -eq 0 ]
      then
        rm -f pj-learn/$ds-$mu-$gamma-pj.h5
        rm -f pj-learn/logging/$ds-$mu-$gamma-pj.log
        echo "Compute dataset [$ds] mu: $mu gamma: $gamma"
        ../bin/pj-learn -iters 50000 -mu $mu -gamma $gamma \
            distances/$ds-unproj.h5 \
            pj-learn/$ds-$mu-$gamma-pj.h5 \
        | tee pj-learn/logging/$ds-$mu-$gamma-pj.log
      else
        echo "Learning PJ for [pj-learn/$ds-$mu-$gamma-pj.h5] already done."
      fi

    done

  done

done

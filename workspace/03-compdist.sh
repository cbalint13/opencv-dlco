#!/bin/bash

##
## This script will compute patch-pairs distance
##

for ds in "yosemite" "notredame" "liberty"
do

  if [ ! -f distances/$ds-dist.h5 ]
  then
    echo "Compute dataset [$ds]"
    ../bin/comp-fulldists filters.h5 dataset/$ds.h5 distances/$ds-dist.h5
  else
    echo "Distances for [$ds] already done."
  fi

done

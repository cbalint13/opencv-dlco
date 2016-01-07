#!/bin/bash

##
## This script will prepare patch datasets:
## http://www.cs.ubc.ca/~mbrown/patchdata/patchdata.html
##

for ds in 'liberty' 'notredame' 'yosemite'
do

  if [ ! -f dataset/$ds.h5 ]
  then
    echo "Export [$ds]"
    ../bin/conv-impatches dataset/$ds/ dataset/$ds.h5
  else
    echo "Dataset [$ds] already done."
  fi

done

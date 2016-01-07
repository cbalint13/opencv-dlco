#!/bin/bash

##
## This script will generate pooling regions
##

if [ ! -f filters.h5 ]
then
  echo "Generate Pooling Regions filters"
  ../bin/gen-poolregion filters.h5
else
  echo "PR Filters already done."
fi

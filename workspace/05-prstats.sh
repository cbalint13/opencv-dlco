#!/bin/bash

##
## This script will select best pr filters
##

for ds in 'liberty' 'notredame' 'yosemite'
do

  list=''
  for p in `ls pr-learn/$ds-*.h5`;
   do
   list="$list -prj $p"
  done

  echo "$list"

  rm -f pr-select-$ds.log

  ulimit -S -c 0
  ../bin/pr-stats filters.h5 \
         -dst distances/yosemite-dist.h5 \
         -dst distances/liberty-dist.h5 \
         -dst distances/notredame-dist.h5 \
         $list | tee pr-select-$ds.log

done

#!/bin/bash

##
## This script will compute statistics with output sorted by AUC
##

for ds in "yosemite" "notredame" "liberty"
do

  list=''
  for p in `find . -name $ds-*-pr.h5 | sed -e 's|\.\/||g'`;
   do
   list="$list -prj $p"
  done

  result=$(cat pr-select-$ds.log | grep Best | wc -l)

  if [ $result -ne 3 ]
  then

    echo "$list"

    rm -f pr-select-$ds.log

    ulimit -S -c 0
    ../bin/pr-stats filters.h5 \
           -dst distances/yosemite-dist.h5 \
           -dst distances/liberty-dist.h5 \
           -dst distances/notredame-dist.h5 \
           $list | tee pr-select-$ds.log
  else
  echo "$ds already done."
  fi

done

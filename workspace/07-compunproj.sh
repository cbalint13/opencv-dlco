#!/bin/bash

##
## This script will compute unprojected descriptors
##

for ds in 'liberty' 'notredame' 'yosemite'
do

  for pr in 'yosemite-0.025-0.075-pr.h5#7' \
            'notredame-0.015-0.075-pr.h5#7' \
            'liberty-0.025-0.075-pr.h5#5'
  do

    prj_id=$(echo "$pr" | cut -d'#' -f2)
    prj_file=$(echo "$pr" | cut -d'#' -f1)

    img=`ls dataset/$ds*.h5`
    prj=`ls pr-learn/$prj_file`

    if [ ! -f descs/$ds-$pr-unproj.h5 ]
    then
      ulimit -S -c 0
      ../bin/comp-uprjdists filters.h5 $img -prj pr-learn/$prj_file -id $prj_id -out descs/$ds-$pr-unproj.h5
    fi

  done

done

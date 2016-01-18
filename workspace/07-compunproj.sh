#!/bin/bash

##
## This script will compute unprojected distances
##

for ds in "yosemite" "notredame" "liberty"
do

  for pr in 'pr-learn/olderbest/yosemite-0.025-0.075-pr.h5#7' \
            'pr-learn/notredame-0.003-0.040-pr.h5#7' \
            'pr-learn/liberty-0.035-0.250-pr.h5#7'
  do

    pr_fl=$(echo $pr | cut -d'#' -f1)
    pr_id=$(echo ${pr##*/} | cut -d'#' -f2)
    pr_lb=$(echo ${pr##*/} | sed 's|.h5||')

    img=`ls dataset/$ds*.h5`

    echo "Compute $ds-$pr_lb-unproj.h5"

    if [ ! -f distances/$ds-$pr_lb-unproj.h5 ]
    then
      ulimit -S -c 0
      ../bin/comp-uprjdists filters.h5 $img -prj $pr_fl -id $pr_id -out distances/$ds-$pr_lb-unproj.h5
    else
      echo "$ds-$pr_lb-unproj.h5 already done."
    fi

  done

done

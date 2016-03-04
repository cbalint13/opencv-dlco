#!/bin/bash

##
## This script will compute statistics with output sorted by FPR
##

echo

rm -f pj-select-*.log

for ds in "yosemite" "notredame" "liberty"
do

  echo
  echo "Models learnt on DS: [$ds]"
  echo

  echo >> pj-select-$ds.log
  echo "Models learnt on DS: [$ds]" >> pj-select-$ds.log
  echo >> pj-select-$ds.log

  for pr in `find . -name $ds-*-pj.log | sed -e 's|\.\/||g'`
  do

    prf=$(echo $pr | sed -e 's|\.log|\.h5|g' -e 's|\/logging||g')

    # last [saved] entry from log
    l=`tac $pr | grep -e 'saved' -m 1 | sed -e 's|:| |g' -e 's|(||g' -e 's|)||g' | awk '{print $6"|"$9"|"$3}'`

    fpr=$(echo $l | cut -d'|' -f2)
    auc=$(echo $l | cut -d'|' -f1)
    dim=$(echo $l | cut -d'|' -f3)
    echo "  ModelStat: FPR95: $fpr AUC #$auc DIM: $dim [$prf]"
    echo "  ModelStat: FPR95: $fpr AUC #$auc DIM: $dim [$prf]" >> pj-select-$ds.log

  done

  best=$(cat pj-select-$ds.log | grep "ModelStat" | sort -n | sed -n '1p' | sed -e 's|  ModelStat:||g')

  echo
  echo "  BestModel:$best"

  echo >> pj-select-$ds.log
  echo "  BestModel:$best" >> pj-select-$ds.log

done

#!/bin/bash

##
## This script will show top 10 best learnt PR based on logs
##



rm -f pj-top10.log

for maxdim in 48 64 80 120
do

  echo
  echo "TOP maxdim #$maxdim minimisers:"

  echo >> pj-top10.log
  echo "TOP maxdim #$maxdim minimisers:" >> pj-top10.log

  entry=''
  for m in `cat pj-select*.log | grep ModelStat | awk '{print $8}'`
  do

    pr=$(echo ${m##*/} | sed -e 's|.h5]||g' -e 's|-|_|'| cut -d'_' -f2)
    entry="$entry|$pr"

  done

  list=''
  for l in `echo "$entry" | sed -e 's|\||\n|g' | sed -e '/^$/d' | sort -u`
  do

    #echo $l
    a1=$(cat pj-select*.log | grep ModelStat | grep /yosemite-$l | awk '{print $3}')
    a2=$(cat pj-select*.log | grep ModelStat | grep /notredame-$l | awk '{print $3}')
    a3=$(cat pj-select*.log | grep ModelStat | grep /liberty-$l | awk '{print $3}')
    dim1=$(cat pj-select*.log | grep ModelStat | grep /yosemite-$l | awk '{print $7}' | cut -d'[' -f2 | sed -e 's|\]||')
    dim2=$(cat pj-select*.log | grep ModelStat | grep /notredame-$l | awk '{print $7}' | cut -d'[' -f2 | sed -e 's|\]||')
    dim3=$(cat pj-select*.log | grep ModelStat | grep /liberty-$l | awk '{print $7}' | cut -d'[' -f2 | sed -e 's|\]||')
    auc1=$(cat pj-select*.log | grep ModelStat | grep /yosemite-$l | awk '{print $5}' | cut -d'#' -f2)
    auc2=$(cat pj-select*.log | grep ModelStat | grep /notredame-$l | awk '{print $5}' | cut -d'#' -f2)
    auc3=$(cat pj-select*.log | grep ModelStat | grep /liberty-$l | awk '{print $5}' | cut -d'#' -f2)

    if [ $dim1 -le $maxdim ] || [ $dim2 -le $maxdim ] || [ $dim3 -le $maxdim ]
    then
      # sort by average FPR95 over datasets
      med=`echo "scale=4;($a1+$a2+$a3)/3" | bc`
      list="$list mean: $med $nd YO:$a1 ND:$a2 LY:$a3 dim #[$dim1/$dim2/$dim3] AUC[$auc1/$auc2/$auc3] $l|";
    fi

  done

  echo "$list" | sed -e 's|\||\n|g' -e 's|\\\[||g'  -e 's|\\\]||g' -e "s|(||g" -e "s|)-||g" | sort -n
  echo "$list" | sed -e 's|\||\n|g' -e 's|\\\[||g'  -e 's|\\\]||g' -e "s|(||g" -e "s|)-||g" | sort -n >> pj-top10.log

done
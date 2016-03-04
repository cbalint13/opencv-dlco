#!/bin/bash

##
## This script will show top 10 best learnt PR based on logs
##


echo
echo "TOP 10 minimiser:"

rm -f pj-top10.log
echo >> pj-top10.log
echo "TOP 10 minimiesr:" >> pj-top10.log

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
  dim=$(cat pj-select*.log | grep ModelStat | grep /liberty-$l | awk '{print $7}' | cut -d'/' -f1 | sed -e 's|\[||')
  # sort by average FPR95 over datasets
  med=`echo "scale=4;($a1+$a2+$a3)/3" | bc`
  list="$list mean: $med $nd YO:$a1 ND:$a2 LY:$a3 dim #$dim $l|";

done

echo "$list" | sed -e 's|\||\n|g' -e 's|\\\[||g'  -e 's|\\\]||g' -e "s|(||g" -e "s|)-||g" | sort -n | sed -n -e '1,11p'
echo "$list" | sed -e 's|\||\n|g' -e 's|\\\[||g'  -e 's|\\\]||g' -e "s|(||g" -e "s|)-||g" | sort -n | sed -n -e '1,11p' >> pj-top10.log

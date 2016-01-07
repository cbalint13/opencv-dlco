#!/bin/bash

##
## This script will show top 5 best learnt PR based on logs
##

echo

rm -f pr-top5.log

for ds in "liberty" "notredame" "yosemite"
do

  echo "TOP 5 learnt over [$ds]:"
  echo >> pr-top5.log
  echo "TOP 5 learnt over [$ds]:" >> pr-top5.log

  list=''
  for l in `cat pr-select-$ds.log | grep notredame-dist | awk '{print $11}' | cut -d'>' -f1 | sed -e 's|\[|\\\[|g' -e 's|\]|\\\]|g'`
  do
    result=$(cat pr-select-$ds.log | grep yosemite-dist | grep "$l")

    if [ ${#result} -eq 0 ];
    then
     continue
    fi

    result=$(cat pr-select-$ds.log | grep liberty-dist | grep "$l")

    if [ ${#result} -eq 0 ];
    then
     continue
    fi

    nd=$(cat pr-select-$ds.log | grep $l | sed '1q' | awk '{ print $7}')
    a1=$(cat pr-select-$ds.log | grep yosemite-dist | grep $l | awk '{print $5}')
    a2=$(cat pr-select-$ds.log | grep notredame-dist | grep $l | awk '{print $5}')
    a3=$(cat pr-select-$ds.log | grep liberty-dist | grep $l | awk '{print $5}')

    med=`echo "scale=4;($a1+$a2+$a3)/3" | bc`
    list="$list mean: $med $nd YO:$a1 ND:$a2 LY:$a3 $l|";
  done

  echo "$list" | sed -e 's|\||\n|g' -e 's|\\\[||g'  -e 's|\\\]||g' -e "s|(||g" -e "s|)-||g" | sort -n | sed -n -e '1,5p'
  echo "$list" | sed -e 's|\||\n|g' -e 's|\\\[||g'  -e 's|\\\]||g' -e "s|(||g" -e "s|)-||g" | sort -n | sed -n -e '1,5p' >> pr-top5.log

  echo

done

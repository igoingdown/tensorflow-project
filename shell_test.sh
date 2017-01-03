#!/usr/bin/env bash

for file in `ls `
do
    cat $file
done

s_1=`date +%F`
s_2=`date +%T`
echo $s_1
echo $s_2
echo "time is: $s_1 $s_2"


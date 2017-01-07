#!/usr/bin/env bash

for file in `ls -R`
do
    cat $file
done

s_1=`date +%F`
s_2=`date +%T`
echo $s_1
echo $s_2
echo "time is: $s_1 $s_2"

if [ $1 == "start" ]; then
    echo "$1"
else
    echo -e "\nwe need a github repository as first arg!"
    exit 1
fi

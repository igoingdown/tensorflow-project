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

remote_origin=`git remote`
echo "$remote_origin"

if [ "$1"x == "start"x ] ; then
    echo "$1"
else
    echo -e "\nwe need a github repository as first arg!"
fi
echo `git remote`

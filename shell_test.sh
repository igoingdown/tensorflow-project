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

arr=("dd" "ee" "ff")
for s in ${arr[@]}
do
    echo "$s"
done

path_arr=(
    "/Users/zhaomingxing/PycharmProjects/test_tensor"
	"/Users/zhaomingxing/IdeaProjects/hello_world/src/package_1"
	"/Users/zhaomingxing/PycharmProjects/python_demo_and_tool"
	"/Users/zhaomingxing/Desktop/zmx/leetcode/leetcode_duplicate"
	"/Users/zhaomingxing/Desktop/zmx/Interesting-Github-Repositories/PythonDataScienceHandbook"
	"/Users/zhaomingxing/Desktop/zmx/Interesting-Github-Repositories/vczh_toys"
	"/Users/zhaomingxing/Desktop/zmx/Interesting-Github-Repositories/google-maps-services-java"
	"/Users/zhaomingxing/Desktop/zmx/MyResume"
	"/Users/zhaomingxing/PycharmProjects/models"
	"/Users/zhaomingxing/PycharmProjects/copyNet")
for path in ${path_arr[@]}
do
    echo $path
done

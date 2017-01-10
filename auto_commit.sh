#!/usr/bin/env bash

cd /Users/zhaomingxing/PycharmProjects/test_tensor
git status >> /dev/null 2>&1
git add . >> /dev/null 2>&1
s_1=`date +%F`
s_2=`date +%T`
remoteOrigin=`git remote`
git commit -m "commit time: $s_1 $s_2" >> /dev/null 2>&1
git push -u "$remoteOrigin" master >> /dev/null 2>&1
#!/usr/bin/env bash

git status >> ../git.out 2>&1
git add . >> ../git.out 2>&1
git commit -m "2017.1.3" >> ../git.out 2>&1
git push -u origin master >> ../git.out 2>&1
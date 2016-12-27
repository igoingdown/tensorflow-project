#!/bin/sh

git status &> ../git.out
git add . &> ../git.out
git commit -a &> ../git.out
git push -u origin master &> ../git.out
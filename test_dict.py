#!/usr/bin/python
# -*- coding:utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from time_wrapper import *

@time_recorder
def fool(m, n):
    print m
    print n
    return n


@time_recorder
def useful_func():
    y = fool(1, 3)
    print y

useful_func()


for i in xrange(100):
    if i % 10 == 0:
        print i

d = {"z": 1, "y": 2, "x": 3}
sorted_list = sorted(d.iteritems(), key=lambda x: x[1], reverse=True)
print sorted_list


"""
Longest Common Subsequences
Created on 2015/7/2  15:11
@author: Wang Xu
"""


def lcs(input_x, input_y):
    # input_y as column, input_x as row
    dp = [([0] * len(input_y)) for _ in range(len(input_x))]
    max_len = max_index = 0
    for i in range(0, len(input_x)):
        for j in range(0, len(input_y)):
            if input_x[i] == input_y[j]:
                if i != 0 and j != 0:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                if i == 0 or j == 0:
                    dp[i][j] = 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    max_index = i + 1 - max_len
    return input_x[max_index:max_index + max_len]

overlap_string = lcs("who's the president of the USA?", "the duck is president")
print len(overlap_string)
print overlap_string

l_1 = [1, 2, 3]
s_1 = set(l_1)

l_2 = [4, 5, 6]
s_2 = set(l_2)

s = s_1 | s_2
print s

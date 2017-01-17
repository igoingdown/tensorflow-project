#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
===============================================================================
author: 赵明星
desc:   测试dict的基本功能。
        实现LCS算法，即最长连续子串算法。
===============================================================================
"""

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from time_wrapper import *
import tensorflow as tf
import six
import re

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

overlap_string = lcs("who's the president of the USA?",
                     "the duck is president")
print len(overlap_string)
print overlap_string

l_1 = [1, 2, 3]
s_1 = set(l_1)

l_2 = [4, 5, 6]
s_2 = set(l_2)

s = s_1 | s_2
print s

s = 1
e = 2
print "aaa{0}bbb{1}".format(s, e)

_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
s = _WORD_SPLIT.split("a.b.c!ddd?ccc\"DDDD:mmmm)xxxxx")
print s

_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
print _buckets
for i, (source_size, target_size) in enumerate(_buckets):
    print i, "\t", source_size, "\t", target_size

bucket_sizes = [2, 2, 2, 2]
buckets_scale = [sum(bucket_sizes[:b + 1]) / float(sum(bucket_sizes)) for b in xrange(len(bucket_sizes))]
print buckets_scale
print bucket_sizes[0:1]

with open("test_dict.txt", "rb") as f:
    line_list = f.readlines()

print line_list
print type(line_list[0])
if isinstance (line_list[0], six.text_type):
    print "\"{0}\" is text type".format(line_list[0])
else:
    print "\"{0}\" is not text type".format(line_list[0])

if isinstance(line_list[0], bytes):
    print "the line is byte type"
else:
    print "the line is not byte type"

byte_lines = [tf.compat.as_bytes(line.strip()) for line in line_list]
print type(byte_lines[0])
print byte_lines

a_list = [[2, 3, 4], [5, 6, 7]]
print "dddd"
print a_list

x = 1 if 1 > 0 else 0
print "x is: {0}".format(x)

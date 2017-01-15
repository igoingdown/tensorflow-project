#!/usr/bin/python
# -*- coding:utf-8 -*-


"""
===============================================================================
author: 赵明星
desc:   学习tensorflow和numpy的简单用法并进行简单的测试。
===============================================================================
"""

import numpy as np
import tensorflow as tf

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

x = tf.constant([1.0])
y = tf.constant([2.0])
z = x/y

with tf.Session() as sess:
    z_res = sess.run(z)
    print z_res

if __name__ == "__main__":
    print "hello world!"
    s = ["3", "4", "6"]
    with open("test_file.txt") as f:
        lines = f.readlines()
        print lines



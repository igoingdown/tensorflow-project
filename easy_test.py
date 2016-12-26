#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
	@auth	赵明星
	@date	2016.12.26
	@desc	numpy and tensorflow demos
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
    a = np.core.defchararray.isnumeric(s)
    print a


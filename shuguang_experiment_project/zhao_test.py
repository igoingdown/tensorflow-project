#!/usr/bin/python
# -*- coding:utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import numpy as np
import tensorflow as tf

with tf.name_scope("test"):
    checkers = []
    x1 = tf.constant([[20.1, 20.1, 20.1],[100.1, 100.1, 100.1]], name = "x1")
    x2 = tf.constant([[1.0,1.0,2.0],[1.0,1.0,1.0]], name = "x2")
    x3 = x1/x2
    #m1 = tf.constant([x3 , [100.1, 100.1, 100.1] ])
    #checkers.append(tf.check_numerics(m1, "nan"))
    #checkers.append(tf.check_numerics(m2, "nan"))
    #checkers.append(tf.check_numerics(t, "nan"))


with tf.Session() as sess:
    writer = tf.train.SummaryWriter("/home/deeplearning/zsg/code/Integration/logs/", sess.graph)
    t_res = sess.run(x3)
    print t_res

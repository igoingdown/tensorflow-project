#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
===============================================================================
author: 赵明星
desc:   测试tensorflow的基本功能。
===============================================================================
"""


import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import tensorflow as tf
import numpy as np

xs = np.random.rand(10, 10)
ys = np.random.rand(10, 10)
indices = np.array([[3, 2, 0], [4, 5, 1]], dtype=np.int64)
data = np.array([3.499, 58493.32], dtype=np.float32)
shape = np.array([7, 9, 2], dtype=np.int64)

graph = tf.Graph()
with graph.as_default():
    x_data = tf.placeholder(tf.float32, [10, 10])
    y_data = tf.placeholder(tf.float32, [10, 10])

    test_np_sparse_data = tf.placeholder(tf.float32)
    y = tf.reduce_sum(test_np_sparse_data)
    res_1 = tf.reduce_sum(x_data, 1)
    res_2 = tf.reduce_sum(x_data, reduction_indices=[1])
    x_max = tf.argmax(x_data, 1)
    y_max = tf.argmax(y_data, 1)

    x_and_y_equal = tf.equal(x_max, y_max)
    accuracy = tf.reduce_mean(tf.cast(x_and_y_equal, tf.float32))

with tf.Session(graph=graph) as sess:
    print "xs:\n", xs
    print "ys:\n", ys
    d = {x_data: xs, y_data: ys}
    print sess.run(res_1, feed_dict=d)

    print "\n" * 3

    print sess.run(res_2, feed_dict=d)
    print sess.run(x_max, feed_dict=d)
    print sess.run(y_max, feed_dict=d)
    print sess.run(x_and_y_equal, feed_dict=d)
    print sess.run(accuracy, feed_dict=d)
    print "\n\nwtf??"

    for i in xrange(10):
        print i

    print [[1, 2], [3, 4]] * 3

    tf.compat.as_bytes("your are idiot")

    # This line won't work.
    # And i don't know why that happens?
    # print sess.run(y, feed_dict={test_np_sparse_data: (indices, data, shape)})

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
pre_data = np.random.rand(3, 3)
indices = np.array([[3, 2, 0], [4, 5, 1]], dtype=np.int64)
data = np.array([3.499, 58493.32], dtype=np.float32)
shape = np.array([7, 9, 2], dtype=np.int64)

graph = tf.Graph()
with graph.as_default():
    x_data = tf.placeholder(tf.float32, [10, 10], name="x")
    y_data = tf.placeholder(tf.float32, [10, 10])
    pre_holder = tf.placeholder(tf.float32, [3, 3], name="pre")
    pre_topk = tf.nn.top_k(pre_holder, 2)
    log_probs = tf.nn.log_softmax(pre_holder)
    class_shape = tf.shape(log_probs)
    max_idx = tf.arg_max(x_data, 1)
    print "shape: {0}".format(pre_holder.get_shape())
    print "name: {0}".format(pre_holder.name)

    test_np_sparse_data = tf.placeholder(tf.float32)
    y = tf.reduce_sum(test_np_sparse_data)
    res_1 = tf.reduce_sum(x_data, 1)
    res_2 = tf.reduce_sum(x_data, reduction_indices=[1])
    x_max = tf.argmax(x_data, 1)
    y_max = tf.argmax(y_data, 1)

    x_and_y_equal = tf.equal(x_max, y_max)
    accuracy = tf.reduce_mean(tf.cast(x_and_y_equal, tf.float32))

    my_range = tf.range(20, name="range")

    # pre_holder.set_shape([None, 1])
    print "after shape set: {0}".format(pre_holder.get_shape())
    print "dtype: {0}".format(pre_holder.dtype)

with tf.Session(graph=graph) as sess:
    print "xs:\n", xs
    print "ys:\n", ys
    d = {x_data: xs, y_data: ys, pre_holder: pre_data}
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
    range_list = sess.run(my_range, feed_dict=d)
    print "test \"//\": {0}".format(range_list // 2)
    print 10//2
    print 10.3 // 3
    print sess.run(max_idx, feed_dict=d)

    # This line won't work.
    # And i don't know why that happens?
    # print sess.run(y,
    #                feed_dict={test_np_sparse_data: (indices, data, shape)})

    print "logsoftmax test res:\n{0}".format(sess.run(log_probs, feed_dict= d))
    print "pre data:\n{0}".format(pre_data)
    print "class shape:\n{0}".format(sess.run(class_shape, feed_dict=d))
    print "top_k of pre_data:\n{0}".format(sess.run(pre_topk, feed_dict=d))
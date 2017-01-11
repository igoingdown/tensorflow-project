#!/usr/bin/python
# -*- coding:utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')


import tensorflow as tf
import numpy as np

x_data = np.random.rand(100)
y_data = 0.87 * x_data + 0.05

graph = tf.Graph()
with graph.as_default():
    xs = tf.placeholder(tf.float32, [100])
    ys = tf.placeholder(tf.float32, [100])

    with tf.name_scope("train"):

        weight = tf.Variable(tf.random_uniform([1], -1, 1), name="Weight")
        bias = tf.Variable(tf.zeros([1]), name="Bias")

        Wx_add_b = xs * weight + bias

        loss = tf.reduce_mean(tf.square(Wx_add_b - ys), name="loss")

        optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    with tf.name_scope("init"):
        init = tf.initialize_all_variables()

    saver = tf.train.Saver()


def train():
    with tf.Session(graph=graph) as sess:
        sess.run(init)
        writer = tf.train.SummaryWriter("logs", sess.graph)
        d = {xs: x_data,
             ys: y_data}

        for i in range(2000):
            _, loss_val = sess.run([optimizer, loss], feed_dict=d)
            print loss_val

        saver.save(sess, "variables_saved/variables")

        print sess.run(weight)
        print sess.run(bias)


def test():
    with tf.Session(graph=graph) as sess:
        d = {xs: x_data}
        saver.restore(sess, "variables_saved/variables")

        print sess.run(weight, feed_dict=d)
        print sess.run(bias, feed_dict=d)

        print sess.run(Wx_add_b, feed_dict=d)
        print y_data


if __name__ == '__main__':
    test()


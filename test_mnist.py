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
# import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
graph = tf.Graph()


def add_layer(inputs, in_size, out_size, layer_name, activation_func=None):
    with tf.name_scope(layer_name):
        with tf.name_scope("weight"):
            weight = tf.Variable(tf.random_normal([in_size, out_size]), name="W")
            tf.summary.histogram(layer_name + "weights", weight)
        with tf.name_scope("bias"):
            bias = tf.Variable(tf.zeros([1, out_size]) + 0.1, name="b")
            tf.summary.histogram(layer_name + "bias", bias)

        with tf.name_scope("wx_add_b"):
            wx_add_b = tf.matmul(inputs, weight) + bias

        if activation_func is None:
            output = wx_add_b
        else:
            output = activation_func(wx_add_b)
            tf.summary.histogram(layer_name + "output", output)
        # dropout half of the edges
        return output


with graph.as_default():
    with tf.name_scope("feed_place"):
        xs = tf.placeholder(tf.float32, [None, 784])
        ys = tf.placeholder(tf.float32, [None, 10])

    with tf.name_scope("nn"):
        prediction = add_layer(xs, 784, 10, "layer_1", tf.nn.softmax)

    with tf.name_scope("loss"):
        cross_entropy = tf.reduce_mean(tf.reduce_sum(-ys * tf.log(prediction), 1))
        tf.summary.scalar("loss", cross_entropy)
        # loss = tf.reduce_mean(tf.reduce_sum(tf.square(prediction - ys), 1), name="loss")

    with tf.name_scope("train"):
        opt = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

    with tf.name_scope("init"):
        init = tf.initialize_all_variables()

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(ys, 1), name="get_correct_prediction_ratio")
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    merged = tf.summary.merge_all()
    saver = tf.train.Saver()


def train():
    with tf.Session(graph=graph) as sess:
        writer = tf.summary.FileWriter("mnist_logs", sess.graph)
        sess.run(init)
        for i in range(2000):
            x_data, y_data = mnist.train.next_batch(100)
            d = {xs: x_data, ys: y_data}
            sess.run(opt, feed_dict=d)
            if i % 50 == 0:
                res = sess.run(merged, feed_dict=d)
                writer.add_summary(res, i)
        saver.save(sess, save_path="variables_saved/mnist_variables")


def test():
    with tf.Session(graph=graph) as sess:
        saver.restore(sess, "variables_saved/mnist_variables")
        x_data, y_data = mnist.test.next_batch(100)
        d = {xs: x_data, ys: y_data}
        print sess.run(accuracy, feed_dict=d)

if __name__ == '__main__':
    train()
    test()

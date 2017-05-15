#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
===============================================================================
author: 赵明星
desc:   使用tensorflow实现CNN。
===============================================================================
"""

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

data_input = input_data.read_data_sets("MNIST_data", one_hot=True)
graph = tf.Graph()


def test():
    pass


def weight_variable(shape, layer):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1),
                       name="weight_" + layer)

def bias_variable(shape, layer):
    return tf.Variable(tf.zeros(shape), name="bias_" + layer)

def convolution2d(x, w, layer):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME",
                        name="convolution_" + layer)

def max_pooling(x, layer):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding="SAME", name="pooling_" + layer)

with graph.as_default():
    with tf.name_scope("place_holders"):
        xs = tf.placeholder(tf.float32, [None, 784], name="x")
        ys = tf.placeholder(tf.float32, [None, 10], name="y")

    with tf.name_scope("variables_layer_1"):
        weight_1 = weight_variable([5, 5, 1, 32], "layer_1")
        bias_1 = bias_variable([32], "layer_1")

    x_image = tf.reshape(xs, [-1, 28, 28, 1])
    convolution_layer_1 = tf.nn.relu(convolution2d(x_image, weight_1,
                                                   "layer_1"))
    pooling_1 = max_pooling(convolution_layer_1, "layer_1")

    with tf.name_scope("variable_layer_2"):
        weight_2 = weight_variable([5, 5, 32, 64], "layer_2")
        bias_2 = bias_variable([1024], "layer_2")

    convolution_layer_2 = tf.nn.relu(convolution2d(pooling_1, weight_2,
                                                   "layer_2"))
    pooling_2 = max_pooling(convolution_layer_2, "layer_2")

    with tf.name_scope("variable_layer_3"):
        weight_3 = weight_variable([7*7*64, 1024], "layer_3")
        bias_3 = bias_variable([1024], "layer_3")

    pooling_2_flat = tf.reshape(pooling_2, [-1, 7 * 7 * 64])
    h_fact = tf.nn.relu(tf.matmul(pooling_2_flat, weight_3) + bias_3)

    keep_prob = tf.placeholder(tf.float32)
    fact_drop = tf.nn.dropout(h_fact, keep_prob)

    with tf.name_scope("variable_layer_4"):
        weight_4 = weight_variable([1024, 10], "output_layer")
        bias_4 = bias_variable([10], "output_layer")
    y_prediction = tf.nn.softmax(tf.matmul(fact_drop, weight_4) + bias_4)

    with tf.name_scope("cross_entropy"):
        cross_entropy = -tf.reduce_sum(ys * tf.log(y_prediction))
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    with tf.name_scope("correct_prediction"):
        correct_prediction = tf.equal(tf.arg_max(y_prediction, 1),
                                      tf.arg_max(ys, 1))
    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.name_scope("initial"):
        initializer = tf.initialize_all_variables()

if __name__ == "__main__":
    with tf.Session(graph=graph) as sess:
        sess.run(initializer)
        for i in range(20000):
            batch_data = data_input.train.next_batch(50)
            if (i % 100 == 0):
                train_accuracy = accuracy.eval(feed_dict={xs:batch_data[0],
                                                          ys:batch_data[1],
                                                          keep_prob:1.0})
                print "step {0} accuracy {1}".format(i, train_accuracy)
            sess.run(train_step, feed_dict={xs:batch_data[0],
                                            ys:batch_data[1],
                                            keep_prob:0.5})
        print "test.accuracy: {0}".format(accuracy.eval(
            feed_dict={xs:data_input.test.images, ys:data_input.test.labels,
                       keep_prob:1.0}))
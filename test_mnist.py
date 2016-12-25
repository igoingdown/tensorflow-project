# coding=utf-8

import tensorflow as tf
# import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
# the following code is designed for the competition
# x_data = np.random.rand(20000, 128)
# y_data = np.random.randint(0, 1, 20000)
# x_data = np.random.rand(20000, 1)
# noise = np.random.normal(0, 0.05, x_data.shape)
# y_data = np.square(x_data) - 0.5 + noise

graph = tf.Graph()


def add_layer(inputs, in_size, out_size, layer_name, activation_func=None):
    with tf.name_scope(layer_name):
        with tf.name_scope("weight"):
            weight = tf.Variable(tf.random_normal([in_size, out_size]), name="W")
            tf.histogram_summary(layer_name + "weights", weight)
        with tf.name_scope("bias"):
            bias = tf.Variable(tf.zeros([1, out_size]) + 0.1, name="b")
            tf.histogram_summary(layer_name + "bias", bias)

        with tf.name_scope("wx_add_b"):
            wx_add_b = tf.matmul(inputs, weight) + bias

        if activation_func is None:
            output = wx_add_b
        else:
            output = activation_func(wx_add_b)
            tf.histogram_summary(layer_name + "output", output)

        # dropout half of the edges
        return output


with graph.as_default():
    with tf.name_scope("feed_place"):
        xs = tf.placeholder(tf.float32, [None, 784])
        ys = tf.placeholder(tf.float32, [None, 10])
        # 我们比赛的代码中,也可以这么弄,可以分批,也可以不分批,形式为[19999, 600] 和 [19999, 6]

    with tf.name_scope("nn"):
        prediction = add_layer(xs, 784, 10, "layer_1", tf.nn.softmax)

    with tf.name_scope("loss"):
        # in the competition, we use entropy and soft_max
        # the question is how many layers in the neural network we should set!
        cross_entropy = tf.reduce_mean(tf.reduce_sum(-ys * tf.log(prediction), 1))
        tf.scalar_summary("loss", cross_entropy)
        # loss = tf.reduce_mean(tf.reduce_sum(tf.square(prediction - ys), 1), name="loss")

    with tf.name_scope("train"):
        opt = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

    with tf.name_scope("init"):
        init = tf.initialize_all_variables()

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(ys, 1), name="get_correct_prediction_ratio")
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary("accuracy", accuracy)

    merged = tf.merge_all_summaries()
    saver = tf.train.Saver()


def train():
    with tf.Session(graph=graph) as sess:
        writer = tf.train.SummaryWriter("mnist_logs", sess.graph)
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

import tensorflow as tf
import numpy as np

# the following code is designed for the competition
# x_data = np.random.rand(20000, 128)
# y_data = np.random.randint(0, 1, 20000)
x_data = np.random.rand(20000, 1)
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise
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
        xs = tf.placeholder(tf.float32, [20000, 1])
        ys = tf.placeholder(tf.float32, [20000, 1])

    with tf.name_scope("nn"):
        mid_res = add_layer(xs, 1, 10, "layer_1", tf.nn.relu)
        res = add_layer(mid_res, 10, 1, "layer_2")

    with tf.name_scope("loss"):
        # in the competition, we use entropy and softmax
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(res - ys), 1), name="loss")

    with tf.name_scope("train"):
        opt = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
        tf.scalar_summary("loss", loss)

    with tf.name_scope("init"):
        init = tf.initialize_all_variables()
    merged = tf.merge_all_summaries()


with tf.Session(graph=graph) as sess:
    writer = tf.train.SummaryWriter("logs", sess.graph)
    d = {xs: x_data, ys: y_data}
    sess.run(init)
    for i in range(1000):
        sess.run(opt, feed_dict=d)
        if i % 50 == 0:
            res = sess.run(merged, feed_dict=d)
            writer.add_summary(res, i)

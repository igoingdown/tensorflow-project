import tensorflow as tf
import numpy as np

xs = np.random.rand(10, 10)
ys = np.random.rand(10, 10)
graph = tf.Graph()

with graph.as_default():
    x_data = tf.placeholder(tf.float32, [10, 10])
    y_data = tf.placeholder(tf.float32, [10, 10])

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

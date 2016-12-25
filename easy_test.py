import tensorflow as tf

x = tf.constant([1.0])
y = tf.constant([2.0])
z = x/y


with tf.Session() as sess:
    z_res = sess.run(z)
    print z_res

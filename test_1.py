import numpy as np
import requests
import lxml
import re
import tensorflow as tf
import cPickle


graph = tf.Graph()
with graph.as_default():
    xs = tf.Variable(tf.zeros([200, 100]))
    ys = tf.transpose(xs)
    space = tf.placeholder(tf.float32, [None, 2])

    num = tf.Variable([[1, 3, 4], [3, 5, 9]], dtype=tf.float32)
    max_num = tf.reduce_max(num, 1, keep_dims=True)
    equal_flag = tf.equal(num, max_num)

    divide_num = tf.Variable([[3, 5, 0], [0, 12, 0]], dtype=tf.float32)
    mask = tf.cast(tf.logical_and(tf.cast(divide_num, tf.bool), tf.cast(tf.ones_like(divide_num), tf.bool)), tf.float32)
    temp_divide_num = tf.cast(tf.equal(divide_num, tf.zeros_like(divide_num)), tf.float32) + divide_num
    res = num / temp_divide_num * mask

    init = tf.initialize_all_variables()

with tf.Session(graph=graph) as sess:
    sess.run(init)
    space_data = np.random.randint(0, 1, [10, 2])
    d = {space: space_data}
    print sess.run(tf.shape(xs), feed_dict=d)
    print sess.run(tf.shape(ys))
    print "space shape: ", sess.run(tf.shape(space), feed_dict=d)
    print sess.run(tf.cast(equal_flag, tf.int32))
    print sess.run(mask)
    print sess.run(res)
    print 0.02510075/0.03319494
    print np.zeros((2, 3), dtype=int)
    print np.zeros((3, 2), dtype=int)
    s1 = set([1, 2])
    s2 = set({3, 2})
    s1 |= s2
    print s1

    str = """<table class="sparql" border="1">
  <tr>
    <th>name</th>
  </tr>
  <tr>
    <td><pre>"Fearless"@en</pre></td>
  </tr>
</table>"""
    name_pattern = re.compile("<td><pre>(.*?)@(.*?)</pre></td>", re.S)
    alias_language_list = name_pattern.findall(str)
    print alias_language_list
    with open("test.txt", "a") as f:
        print  >>f, "hello world"



#!/usr/bin/python
# -*- coding:utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import tensorflow as tf
import numpy as np
import melt_dataset
import sys
from sklearn.metrics import roc_auc_score


def init_weights(shape):
	return tf.Variable(tf.random_normal(shape, stddev=0.01))

# 工具函数
def length(data):
	used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
	max_length = tf.reduce_sum(used, reduction_indices=1)
	max_length = tf.cast(max_length, tf.int32)
	return max_length

def model(X, w_h, w_o):
	h = tf.nn.sigmoid(tf.matmul(X, w_h)) # this is a basic mlp, think 2 stacked logistic regressions
	return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us

X = tf.placeholder("float", [None, num_features]) # create symbolic variables
Y = tf.placeholder("float", [None, 1])
w_h = init_weights([num_features, hidden_size]) # create symbolic variables
w_o = init_weights([hidden_size, 1])
py_x = model(X, w_h, w_o)
predict_op = tf.nn.sigmoid(py_x)

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(py_x, Y)) # compute costs
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) # construct an optimizer
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)


for i in range(num_iters):
	predicts, cost_ = sess.run([predict_op, cost], feed_dict={X: teX, Y: teY})
	print i, 'auc:', roc_auc_score(teY, predicts), 'cost:', cost_

for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
	sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
predicts, cost_ = sess.run([predict_op, cost], feed_dict={X: teX, Y: teY})
print 'final ', 'auc:', roc_auc_score(teY, predicts),'cost:', cost_


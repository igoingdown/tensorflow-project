#!/usr/bin/python
# -*- coding:utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from helper import *
import tensorflow as tf
from batch_normalization import *

batch_size = 10
max_len = 20
hidder_size = 100


def train(d, s):
	with tf.Session(graph=graph) as sess:
		writer = tf.train.SummaryWriter("logs_" + s, sess.graph)
		sess.run(init)
		for i in xrange(100):
			print "running training epoch {0}......".format(i)
			raw_datas = get_raw_data(FILE_SAMPLE_DATA)
			batch_size = 10
			max_len = 20
			for batch_data in batches_producer(raw_datas, batch_size, max_len):
				shaped_batch_data = normalize_raw_batch(batch_data, batch_size, max_len)
				d = {}
			sess.run(opt, feed_dict=d)
			if i % 50 == 0:
				res = sess.run(merged, feed_dict=d)
				writer.add_summary(res, i)
		saver.save(sess, save_path="variables_saved/model_variables_" + s)


def test(d, s):
	with tf.Session(graph=graph) as sess:
		saver.restore(sess, "variables_saved/model_variables_" + s)
		a = sess.run(accuracy, feed_dict=d)
		print "the accuracy of the prediction model on this label is {0}".format(a), "\n" * 2


graph = tf.Graph()
with graph.as_default():
	# with tf.device('/gpu:1'):
	# use cpu only
	with tf.name_scope("feed_place"):
		labels = tf.placeholder(tf.float32, [batch_size])
		xs = tf.placeholder(tf.float32, [None, 600])
		ys = tf.placeholder(tf.float32, [None, 7])

	with tf.name_scope("nn"):
		prediction = add_layer(xs, 600, 7, "layer_1", tf.sigmoid)

	with tf.name_scope("loss"):
		# the question is how many layers in the neural network we should set!
		cross_entropy = tf.reduce_mean(tf.reduce_sum(-ys * tf.log(prediction), 1))
		tf.scalar_summary("loss", cross_entropy)

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


if __name__=='__main__':
	# take label A as an example first.
	# using cPickle! reconstruct the code!
	label_string = "a"
	train_feature, train_label, test_feature, test_label = seperate_train_and_test_set(label_string)

	train_feed_dict = {xs: train_feature, ys: train_label}
	test_feed_dict = {xs: test_feature, ys: test_label}
	train(train_feed_dict, label_string)
	test(test_feed_dict, label_string)


	# label B is different to label A and label C (only 2 option male and female)
	# when dealing with label B, placeholder should be changed (xs should be [None, 200], ys should be [None, 2])




























#!/usr/bin/python
# -*- coding:utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')


from helper import *
import os
import tensorflow as tf
from batch_normalization import *

batch_size = 10
max_len = 20
token_num = 2000000
hidden_size = 100

FILE_QUERY_AD = "data/trainset"
FILE_QUERY_AD_V = "data/validation"
"""
FILE_QUERY_AD = "data.sample"
FILE_QUERY_AD_V = "data.sample"
"""
 
# 工具函数
def length(data):
	used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
	max_length = tf.reduce_sum(used, reduction_indices=1)
	max_length = tf.cast(max_length, tf.int32)
	return max_length


def embeddings_initializer(dim_x, dim_y, minn, maxx):
	embeddings = tf.Variable(tf.random_uniform([dim_x, dim_y], minn, maxx))
	embeddings = tf.nn.l2_normalize(embeddings, 1)
	embeddings = tf.concat(0, [tf.zeros([1, dim_y], tf.float32), embeddings])
	return embeddings


def GRU_net(data, cell, scope, reused):
	with tf.variable_scope(scope, reuse=reused):
		output, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32, sequence_length=length(data))
	return output, state
	# output: [batch_size, max_size, hidden_size]
	# state: [batch_size, hidden_size]


graph = tf.Graph()
with graph.as_default():
	token_embeddings = embeddings_initializer(token_num, hidden_size, 0, 1)
	with tf.name_scope("feed_place"):
		print_checkers = []
		labels = tf.placeholder(tf.float32, [batch_size, 1])
		query_tokens = tf.placeholder(tf.int32, [batch_size, max_len])
		title_tokens = tf.placeholder(tf.int32, [batch_size, max_len])
		ad_loc_feature = tf.placeholder(tf.float32, [batch_size, 16])

	with tf.name_scope("look_up_embedding"):
		test_embedding_1 = tf.reshape(query_tokens, [-1])
		test_embedding_2 = tf.nn.embedding_lookup(token_embeddings, test_embedding_1)
		query_token_embeddings = tf.reshape(tf.nn.embedding_lookup(token_embeddings, tf.reshape(query_tokens, [-1])), [-1, max_len, hidden_size])
		title_token_embeddings = tf.reshape(tf.nn.embedding_lookup(token_embeddings, tf.reshape(title_tokens, [-1])), [-1, max_len, hidden_size])

	with tf.name_scope('generate_embedding'):
		with tf.variable_scope("cell_scope"):
			cell = tf.nn.rnn_cell.GRUCell(hidden_size)
		with tf.name_scope("query_output_state"):
			query_output, query_state = GRU_net(query_token_embeddings, cell, "cell_scope", False)
		with tf.name_scope("title_output_state"):
			title_output, title_state = GRU_net(title_token_embeddings, cell, "cell_scope", True)
		# output: [batch_size, max_len, hidden_size]
		# state: [batch_size, hidden_size]

	with tf.name_scope("generate_features"):
		query_state_mul_dims = tf.tile(tf.expand_dims(query_state, 1),[1,max_len, 1])
		# query_state_mul_dims: [batch_size, max_len, hidden_size]
		matmul_res = tf.reduce_sum(tf.mul(query_state_mul_dims, title_output), 2)
		print_checkers.append(tf.check_numerics(matmul_res, "nan error 1"))

		square_query_state = tf.reshape(tf.reduce_sum(tf.square(tf.reshape(query_state_mul_dims, [-1, hidden_size])), 1), [batch_size, max_len])
		square_title_output = tf.reshape(tf.reduce_sum(tf.square(tf.reshape(title_output, [-1, hidden_size])), 1), [batch_size, max_len])
		temp_devide_num = tf.mul(tf.sqrt(square_title_output), tf.sqrt(square_query_state))
		cons = tf.zeros([batch_size, max_len])
		devide_num = tf.cast(tf.equal(temp_devide_num, tf.zeros_like(temp_devide_num)), tf.float32) + temp_devide_num
		k_args = matmul_res / devide_num
		print_checkers.append(tf.check_numerics(k_args, "nan error 2"))

		#TODO: generalize k_args
		generalized_k_args = k_args / tf.tile(tf.reduce_sum(k_args, 1, keep_dims=True), [1, max_len])

		divide_num_after_change_0s_into_1s = tf.cast(tf.equal(tf.tile(tf.expand_dims(generalized_k_args, 2),[1, 1, hidden_size]), \
				tf.zeros_like(tf.tile(tf.expand_dims(generalized_k_args, 2),[1, 1, hidden_size]))), tf.float32) \
				+ tf.tile(tf.expand_dims(generalized_k_args, 2),[1, 1, hidden_size])
		mask = tf.cast(tf.logical_and(tf.cast(tf.tile(tf.expand_dims(generalized_k_args, 2),[1, 1, hidden_size]), tf.bool), \
				tf.cast(tf.ones_like(tf.tile(tf.expand_dims(generalized_k_args, 2),[1, 1, hidden_size])), tf.bool)), tf.float32)
		title_feature = tf.reduce_sum(title_output / divide_num_after_change_0s_into_1s * mask, 1)
		print_checkers.append(tf.check_numerics(title_feature, "nan error 3"))

		# ad_loc_feature = tf.Variable(tf.zeros([batch_size, 6]), name="ad_loc_feature")
		# should be modified in the later experiments!!!!
                
		batch_features = tf.concat(1, [query_state, title_feature, ad_loc_feature])

	with tf.name_scope("linear_network"):
		with tf.name_scope("weight"):
			weight = tf.Variable(tf.random_normal([hidden_size * 2 + 16, 1]), name="W")
		with tf.name_scope("bias"):
			bias = tf.Variable(tf.zeros([1, 1]) + 0.1, name="b")
		with tf.name_scope("wx_add_b"):
			wx_add_b = tf.matmul(batch_features, weight) + bias
			print_checkers.append(wx_add_b)
		with tf.name_scope("output"):
			prediction = tf.sigmoid(wx_add_b)

	with tf.name_scope("loss"):
		# the question is how many layers in the neural network we should set!
		# cross_entropy = tf.reduce_mean(tf.reduce_sum(-labels * tf.log(prediction), 1))
		cross_entropy =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(wx_add_b, labels))
		tf.scalar_summary("loss", cross_entropy)

	with tf.name_scope("train"):
		opt = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

	with tf.name_scope("accuracy"):
		correct_prediction = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(labels, 1), name="get_correct_prediction_ratio")
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		tf.scalar_summary("accuracy", accuracy)

	merged = tf.merge_all_summaries()
	saver = tf.train.Saver()


	with tf.name_scope("init"):
		init = tf.initialize_all_variables()

def train(s):
	with tf.Session(graph=graph) as sess:
		writer = tf.train.SummaryWriter("logs_" + s, sess.graph)
		sess.run(init)
                import cPickle as pickle
		for i in xrange(2):
			print "running training epoch {0}......".format(i)
			raw_datas = get_raw_data(FILE_QUERY_AD)
			batch_size = 10
			max_len = 20
                        count = 0
			for batch_data in batches_producer(raw_datas, batch_size, max_len):
				shaped_batch_data = normalize_raw_batch_using_binary_classifaction(batch_data, batch_size, max_len)
				d = {labels: shaped_batch_data[0],
					 query_tokens: shaped_batch_data[1],
					 title_tokens: shaped_batch_data[2],
					 ad_loc_feature: shaped_batch_data[3]}
				# print "matmul_res shape: ", sess.run(tf.shape(matmul_res), feed_dict=d)
				# print "temp_devide_num shape: ", sess.run(tf.shape(temp_devide_num), feed_dict=d)
				# print "square_title_output shape: ", sess.run(tf.shape(square_title_output), feed_dict=d)
				# print "k_args shape: ", sess.run(tf.shape(k_args), feed_dict=d)
				# print "batch_features shape: ", sess.run(tf.shape(batch_features), feed_dict=d)
                                
				sess.run(print_checkers, feed_dict=d)
				sess.run(opt, feed_dict=d)
                                q_emb = sess.run(query_state, feed_dict=d)
                                a_emb = sess.run(title_feature, feed_dict=d)
                                print "PRINTING....", count
                                pickle.dump(q_emb, open("query/q_emb.pkl"+str(count), "wb"))
                                pickle.dump(a_emb, open("title/a_emb.pkl"+str(count), "wb"))
                                count += 1


		if not os.path.exists("variables_saved"):
			os.makedirs("variables_saved")
		saver.save(sess, save_path="variables_saved/model_variables_" + s)



# this function needs to be modified!
def test(s):
	with tf.Session(graph=graph) as sess:
		saver.restore(sess, "variables_saved/model_variables_" + s)
		raw_datas = get_raw_data(FILE_QUERY_AD_V)
		batch_size = 10
		max_len = 20
		# print "the possibility prediction of being label 1 on every ad is:\n"
		i = 0
		for batch_data in batches_producer(raw_datas, batch_size, max_len):
			shaped_batch_data = normalize_raw_batch_using_binary_classifaction(batch_data, batch_size, max_len)
			i += 1
			d = {labels: shaped_batch_data[0],
				 query_tokens: shaped_batch_data[1],
				 title_tokens: shaped_batch_data[2],
				 ad_loc_feature: shaped_batch_data[3]}
			possibility_prediction = sess.run(prediction, feed_dict=d)

			"""
			if i == 1:
				# print "possibility_prediction:\n", possibility_prediction
				# print "labels:\n", shaped_batch_data[0]
				print "\n\n\n"
				print "检索词query的词嵌入Query_emb:"
				print "shape: ", sess.run(tf.shape(query_token_embeddings), feed_dict=d)
				print sess.run(query_token_embeddings, feed_dict=d), "\n\n\n"
				print "广告标题ad_title的词嵌入Title_emb:"
				print "shape: ", sess.run(tf.shape(title_token_embeddings), feed_dict=d)
				print sess.run(title_token_embeddings, feed_dict=d), "\n\n\n"
				print "经注意力模型改写后的ad_title的词嵌入Title_emb_attention:"
				print "shape:", sess.run(tf.shape(title_output / divide_num_after_change_0s_into_1s * mask), feed_dict=d)
				print sess.run(title_output / divide_num_after_change_0s_into_1s * mask, feed_dict=d), "\n\n\n"
				print "线性模型的融合特征batch_feature:"
				print "shape:", sess.run(tf.shape(batch_features), feed_dict=d)
				print sess.run(batch_features, feed_dict=d), "\n\n\n"
			"""

			with open("attention_based_prediction_6w_2p.txt", "a") as f:
				for x in possibility_prediction:
					print >> f, round(x[0], 6)
			# write test prediction into the file
			# print "accuracy: ", sess.run(accuracy, feed_dict = d)


if __name__=='__main__':
        import time
        start = time.time()
	train("helper_graph")
	test("helper_graph")
        print "COST TIME:", time.time() - start



























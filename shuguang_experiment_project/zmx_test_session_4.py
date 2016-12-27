#!/usr/bin/python
# -*- coding:utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import numpy as np
import tensorflow as tf
import zmx_params as params
from memory_profiler import profile
import time
import os
from memory_profiler import memory_usage
from time import sleep
from zmx_after_raw_data_3 import *
from zmx_create_test_batch_data import TestTextLoader

#工具函数
def length(data):
	used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
	max_length = tf.reduce_sum(used, reduction_indices=1)
	max_length = tf.cast(max_length, tf.int32)
	return max_length

#由于一批训练样例使用相同的负样例，因此负样例可直接用该biGRU而不用考虑升维
def biGRU(data, cell_fw, cell_bw, scope1, scope2, reused):
	with tf.variable_scope(scope1, reuse=reused):
		output_fw, state_fw = tf.nn.dynamic_rnn(cell_fw, data, dtype=tf.float32, sequence_length=length(data))
	reverse_data = tf.reverse_sequence(data, seq_lengths=length(data), seq_dim=1)
	with tf.variable_scope(scope2, reuse=reused):
		output_bw, state_bw = tf.nn.dynamic_rnn(cell_bw, reverse_data, dtype=tf.float32, sequence_length=length(reverse_data))
	
	#连接操作
	output = tf.concat(2, [output_fw, tf.reverse_sequence(output_bw, seq_lengths=length(output_bw), seq_dim=1)])
	state = tf.concat(1, [state_fw, state_bw])
	return output, state
	# output: [batch_size, max_size, 2 * hidden_size]
	# state: [batch_size, 2 * hidden_size]

def embeddings_initializer(dim_x, dim_y, minn, maxx):

	# 由于新版本的代码，oov_word, oov_phrase的开始序列均从1开始，因此这里的 dim_x-1 要改为 dim_x
	# embeddings = tf.Variable(tf.random_uniform([dim_x-1, dim_y], minn, maxx))
	embeddings = tf.Variable(tf.random_uniform([dim_x, dim_y], minn, maxx))

	embeddings = tf.nn.l2_normalize(embeddings, 1)
	embeddings = tf.concat(0, [tf.zeros([1, dim_y], tf.float32), embeddings])
	return embeddings

def check_zero_element_num(data):
	return tf.reduce_sum(tf.ones_like(data)-tf.sign(tf.abs(data)))



train_loader = Textloader()
graph = tf.Graph()
with graph.as_default(), tf.device('/gpu:1'):
	print_checkers = []

	"""
	# the shape of the following placeholders should be changed
	# because test data is far different from the train data!

	QA_NL_character_inputs = tf.placeholder(tf.int32, shape=[params.qa_batch_size, params.qa_nl_character_max_length])
	QA_NL_word_inputs = tf.placeholder(tf.int32, shape=[params.qa_batch_size, params.qa_nl_word_max_length])
	QA_NL_phrase_inputs = tf.placeholder(tf.int32, shape=[params.qa_batch_size, params.qa_nl_phrase_max_length])
	QA_KB_character_inputs = tf.placeholder(tf.int32, shape=[params.qa_batch_size, params.qa_kb_character_max_length])
	QA_KB_word_inputs = tf.placeholder(tf.int32, shape=[params.qa_batch_size, params.qa_kb_word_max_length])
	QA_KB_phrase_inputs = tf.placeholder(tf.int32, shape=[params.qa_batch_size, params.qa_kb_phrase_max_length])
	QA_KB_neg_character_inputs = tf.placeholder(tf.int32, shape=[params.qa_neg_size, params.qa_kb_character_max_length])
	QA_KB_neg_word_inputs = tf.placeholder(tf.int32, shape=[params.qa_neg_size, params.qa_kb_word_max_length])
	QA_KB_neg_phrase_inputs = tf.placeholder(tf.int32, shape=[params.qa_neg_size, params.qa_kb_phrase_max_length])
	QA_NL_oov_word_inputs = tf.placeholder(tf.int32, shape=[params.qa_batch_size, params.qa_nl_word_max_length])
	QA_NL_oov_phrase_inputs = tf.placeholder(tf.int32, shape=[params.qa_batch_size, params.qa_nl_phrase_max_length])
	QA_KB_oov_word_inputs = tf.placeholder(tf.int32, shape=[params.qa_batch_size, params.qa_kb_word_max_length])
	QA_KB_oov_phrase_inputs = tf.placeholder(tf.int32, shape=[params.qa_batch_size, params.qa_kb_phrase_max_length])
	"""

	QA_NL_character_inputs = tf.placeholder(tf.int32, shape=[None, params.qa_nl_character_max_length], name="question_character")
	QA_NL_word_inputs = tf.placeholder(tf.int32, shape=[None, params.qa_nl_word_max_length], name="question_word")
	QA_NL_phrase_inputs = tf.placeholder(tf.int32, shape=[None, params.qa_nl_phrase_max_length], name="question_phrase")
	QA_KB_character_inputs = tf.placeholder(tf.int32, shape=[None, params.qa_kb_character_max_length], name="answer_character")
	QA_KB_word_inputs = tf.placeholder(tf.int32, shape=[None, params.qa_kb_word_max_length], name="answer_word")
	QA_KB_phrase_inputs = tf.placeholder(tf.int32, shape=[None, params.qa_kb_phrase_max_length], name="answer_phrase")
	QA_KB_neg_character_inputs = tf.placeholder(tf.int32, shape=[params.qa_neg_size, params.qa_kb_character_max_length], name="neg_answer_character")
	QA_KB_neg_word_inputs = tf.placeholder(tf.int32, shape=[params.qa_neg_size, params.qa_kb_word_max_length], name="neg_answer_word")
	QA_KB_neg_phrase_inputs = tf.placeholder(tf.int32, shape=[params.qa_neg_size, params.qa_kb_phrase_max_length], name="neg_answer_phrase")
	QA_NL_oov_word_inputs = tf.placeholder(tf.int32, shape=[None, params.qa_nl_word_max_length], name="question_oov_word")
	QA_NL_oov_phrase_inputs = tf.placeholder(tf.int32, shape=[None, params.qa_nl_phrase_max_length], name="question_oov_phrase")
	QA_KB_oov_word_inputs = tf.placeholder(tf.int32, shape=[None, params.qa_kb_word_max_length], name="answer_oov_word")
	QA_KB_oov_phrase_inputs = tf.placeholder(tf.int32, shape=[None, params.qa_kb_phrase_max_length], name="answer_oov_phrase")


	QA_KB_neg_oov_word_inputs = tf.placeholder(tf.int32, shape=[params.qa_neg_size, params.qa_kb_word_max_length], name="neg_answer_oov_word")
	QA_KB_neg_oov_phrase_inputs = tf.placeholder(tf.int32, shape=[params.qa_neg_size, params.qa_kb_phrase_max_length], name="neg_answer_oov_phrase")
	NL_character_of_oov_word = tf.placeholder(tf.int32, name="question_character_of_oov_word")
	NL_word_of_oov_phrase = tf.placeholder(tf.int32, name="question_word_of_oov_phrase")
	NL_oov_word_of_oov_phrase = tf.placeholder(tf.int32, name="question_oov_word_of_oov_phrase")
	KB_character_of_oov_word = tf.placeholder(tf.int32, name="answer_character_of_oov_word")
	KB_word_of_oov_phrase = tf.placeholder(tf.int32, name="answer_word_of_oov_phrase")
	KB_oov_word_of_oov_phrase = tf.placeholder(tf.int32, name="answer_oov_word_of_oov_phrase")

	"""
	# the shape of the following placeholders should be changed
	# because test data is far different from the train data!

	# hierarchical summarization需要的树结构信息placeholder（哪个单词对应哪些字母，单词长度是多少...）
	QA_c2w_tree = tf.placeholder(tf.float32, shape=[params.sf_batch_size, params.qa_nl_word_max_length, params.qa_nl_character_max_length], name="c2w")
	# checkers .append(tf.check_numerics(QA_c2w_tree, "nan error 1"))
	QA_w2p_tree = tf.placeholder(tf.float32, shape=[params.sf_batch_size, params.qa_nl_phrase_max_length, params.qa_nl_word_max_length], name="w2p")
	# checkers .append(tf.check_numerics(QA_w2p_tree, "nan error 2"))
	"""

	# hierarchical summarization需要的树结构信息placeholder（哪个单词对应哪些字母，单词长度是多少...）
	QA_c2w_tree = tf.placeholder(tf.float32, shape=[None, params.qa_nl_word_max_length, params.qa_nl_character_max_length], name="c2w")
	# checkers .append(tf.check_numerics(QA_c2w_tree, "nan error 1"))
	QA_w2p_tree = tf.placeholder(tf.float32, shape=[None, params.qa_nl_phrase_max_length, params.qa_nl_word_max_length], name="w2p")
	# checkers .append(tf.check_numerics(QA_w2p_tree, "nan error 2"))

	"""
	# the shape of the following placeholders should be changed
	# because test data is far different from the train data!

	# hierarchical summarization需要nl端word和phrase的mask信息，然而在attention部分，需要nl和kb在所有level的mask信息
	QA_NL_character_mask = tf.placeholder(tf.float32, shape=[params.qa_batch_size, params.qa_nl_character_max_length])
	# checkers .append(tf.check_numerics(QA_NL_character_mask, "nan error 3"))
	# QA_NL_character_mask [10, 100]
	QA_NL_word_mask = tf.placeholder(tf.float32, shape=[params.qa_batch_size, params.qa_nl_word_max_length])
	# checkers .append(tf.check_numerics(QA_NL_word_mask, "nan error 4"))
	QA_NL_phrase_mask = tf.placeholder(tf.float32, shape=[params.qa_batch_size, params.qa_nl_phrase_max_length])
	# checkers .append(tf.check_numerics(QA_NL_phrase_mask, "nan error 5"))
	QA_KB_character_mask = tf.placeholder(tf.float32, shape=[params.qa_batch_size, params.qa_kb_character_max_length])
	# checkers .append(tf.check_numerics(QA_KB_character_mask, "nan error 6"))
	QA_KB_word_mask = tf.placeholder(tf.float32, shape=[params.qa_batch_size, params.qa_kb_word_max_length])
	# checkers .append(tf.check_numerics(QA_KB_word_mask, "nan error 7"))
	QA_KB_phrase_mask = tf.placeholder(tf.float32, shape=[params.qa_batch_size, params.qa_kb_phrase_max_length])
	# checkers .append(tf.check_numerics(QA_KB_phrase_mask, "nan error 8"))
	"""

	# hierarchical summarization需要nl端word和phrase的mask信息，然而在attention部分，需要nl和kb在所有level的mask信息
	QA_NL_character_mask = tf.placeholder(tf.float32, shape=[None, params.qa_nl_character_max_length], name="question_character_mask")
	# checkers .append(tf.check_numerics(QA_NL_character_mask, "nan error 3"))
	# QA_NL_character_mask [10, 100]
	QA_NL_word_mask = tf.placeholder(tf.float32, shape=[None, params.qa_nl_word_max_length], name="question_word_mask")
	# checkers .append(tf.check_numerics(QA_NL_word_mask, "nan error 4"))
	QA_NL_phrase_mask = tf.placeholder(tf.float32, shape=[None, params.qa_nl_phrase_max_length], name="question_phrase_mask")
	# checkers .append(tf.check_numerics(QA_NL_phrase_mask, "nan error 5"))
	QA_KB_character_mask = tf.placeholder(tf.float32, shape=[None, params.qa_kb_character_max_length], name="answer_character_mask")
	# checkers .append(tf.check_numerics(QA_KB_character_mask, "nan error 6"))
	QA_KB_word_mask = tf.placeholder(tf.float32, shape=[None, params.qa_kb_word_max_length], name="answer_word_mask")
	# checkers .append(tf.check_numerics(QA_KB_word_mask, "nan error 7"))
	QA_KB_phrase_mask = tf.placeholder(tf.float32, shape=[None, params.qa_kb_phrase_max_length], name="answer_phrase_mask")
	# checkers .append(tf.check_numerics(QA_KB_phrase_mask, "nan error 8"))


	QA_KB_neg_character_mask = tf.placeholder(tf.float32, shape=[params.qa_neg_size, params.qa_kb_character_max_length], name="neg_answer_character_mask")
	# checkers .append(tf.check_numerics(QA_KB_neg_character_mask, "nan error 9"))
	QA_KB_neg_word_mask = tf.placeholder(tf.float32, shape=[params.qa_neg_size, params.qa_kb_word_max_length], name="neg_answer_word_mask")
	# checkers .append(tf.check_numerics(QA_KB_neg_word_mask, "nan error 10"))
	QA_KB_neg_phrase_mask = tf.placeholder(tf.float32, shape=[params.qa_neg_size, params.qa_kb_phrase_max_length], name="neg_answer_phrase_mask")
	# checkers .append(tf.check_numerics(QA_KB_neg_phrase_mask, "nan error 11"))

	### variable
	# question-answer匹配，从character到phrase共三层，每层有NL和KB两侧，每侧需要两个不同方向的GRU
	with tf.variable_scope('QA_NL_character_fw'):
		QA_NL_character_cell_fw = tf.nn.rnn_cell.GRUCell(params.character_hidden_size)
	with tf.variable_scope('QA_NL_character_bw'):
		QA_NL_character_cell_bw = tf.nn.rnn_cell.GRUCell(params.character_hidden_size)
	with tf.variable_scope('QA_NL_word_fw'):
		QA_NL_word_cell_fw = tf.nn.rnn_cell.GRUCell(params.word_hidden_size)
	with tf.variable_scope('QA_NL_word_bw'):
		QA_NL_word_cell_bw = tf.nn.rnn_cell.GRUCell(params.word_hidden_size)
	with tf.variable_scope('QA_NL_phrase_fw'):	
		QA_NL_phrase_cell_fw = tf.nn.rnn_cell.GRUCell(params.phrase_hidden_size)
	with tf.variable_scope('QA_NL_phrase_bw'):
		QA_NL_phrase_cell_bw = tf.nn.rnn_cell.GRUCell(params.phrase_hidden_size)
	with tf.variable_scope('QA_KB_character_fw'):
		QA_KB_character_cell_fw = tf.nn.rnn_cell.GRUCell(params.character_hidden_size)
	with tf.variable_scope('QA_KB_character_bw'):
		QA_KB_character_cell_bw = tf.nn.rnn_cell.GRUCell(params.character_hidden_size)
	with tf.variable_scope('QA_KB_word_fw'):
		QA_KB_word_cell_fw = tf.nn.rnn_cell.GRUCell(params.word_hidden_size)
	with tf.variable_scope('QA_KB_word_bw'):
		QA_KB_word_cell_bw = tf.nn.rnn_cell.GRUCell(params.word_hidden_size)
	with tf.variable_scope('QA_KB_phrase_fw'):
		QA_KB_phrase_cell_fw = tf.nn.rnn_cell.GRUCell(params.phrase_hidden_size)
	with tf.variable_scope('QA_KB_phrase_bw'):
		QA_KB_phrase_cell_bw = tf.nn.rnn_cell.GRUCell(params.phrase_hidden_size)

	# statement-fact计算attention score，从character到phrase共三层，每层需要一个weight,一个bias和一个projection(注意江乐原来分开写了)
	with tf.variable_scope('QA_character'):
		# 将所有的weight设计为[4 * hiddeen_size, 1],这样就简单很多，也不再需要pro参数了。
		# QA_character_weight = tf.Variable(tf.truncated_normal([(params.character_hidden_size+params.character_hidden_size)*2, params.character_hidden_size], -1.0, 1.0))
		# QA_character_bias = tf.Variable(tf.zeros([params.character_hidden_size]))
		# QA_character_pro = tf.Variable(tf.truncated_normal([params.character_hidden_size, 1], -1.0, 1.0))#matmul要求y矩阵必须是二维
		QA_character_weight = tf.Variable(tf.truncated_normal([(params.character_hidden_size+params.character_hidden_size)*2, 1], -1.0, 1.0))
		QA_character_bias = tf.Variable(tf.zeros([1]))

	with tf.variable_scope('QA_word'):
		# QA_word_weight = tf.Variable(tf.truncated_normal([(params.word_hidden_size+params.word_hidden_size)*2, params.word_hidden_size], -1.0, 1.0))
		# QA_word_bias = tf.Variable(tf.zeros([params.word_hidden_size]))
		# QA_word_pro = tf.Variable(tf.truncated_normal([params.word_hidden_size, 1], -1.0, 1.0))
		QA_word_weight = tf.Variable(tf.truncated_normal([(params.word_hidden_size+params.word_hidden_size)*2, 1], -1.0, 1.0))
		QA_word_bias = tf.Variable(tf.zeros([1]))

	with tf.variable_scope('QA_phrase'):
		# qa_phrase_weight = tf.variable(tf.truncated_normal([(params.phrase_hidden_size+params.phrase_hidden_size)*2, params.phrase_hidden_size], -1.0, 1.0))
		# qa_phrase_bias = tf.variable(tf.zeros([params.phrase_hidden_size]))
		# QA_phrase_pro = tf.Variable(tf.truncated_normal([params.phrase_hidden_size, 1], -1.0, 1.0))
		QA_phrase_weight = tf.Variable(tf.truncated_normal([(params.phrase_hidden_size+params.phrase_hidden_size)*2, 1], -1.0, 1.0))
		QA_phrase_bias = tf.Variable(tf.zeros([1]))

	# 计算OOV词和词组的embedding，需要4个单向GRU
	with tf.variable_scope('NL' + 'c2w'):
		NL_c2w_cell = tf.nn.rnn_cell.GRUCell(params.character_hidden_size)
	with tf.variable_scope('NL' + 'w2p'):
		NL_w2p_cell = tf.nn.rnn_cell.GRUCell(params.word_hidden_size)
	with tf.variable_scope('KB' + 'c2w'):
		KB_c2w_cell = tf.nn.rnn_cell.GRUCell(params.character_hidden_size)
	with tf.variable_scope('KB' + 'w2p'):
		KB_w2p_cell = tf.nn.rnn_cell.GRUCell(params.word_hidden_size)

	# Multi-grained embedding由非监督学习获得，此处先直接随机生成
	with tf.name_scope("embeddings_initializer_NL_charater"):
		NL_character_embeddings = embeddings_initializer(train_loader.nl_character_oov_begin-1, params.character_hidden_size, -1.0, 1.0)
		# checkers .append(tf.check_numerics(NL_character_embeddings, "nan error 34"))

	with tf.name_scope("embeddings_initializer_NL_word"):
		NL_word_embeddings = embeddings_initializer(train_loader.nl_word_oov_begin-1, params.word_hidden_size, -1.0, 1.0)
		# checkers .append(tf.check_numerics(NL_word_embeddings, "nan error 12"))

	with tf.name_scope("embeddings_initializer_NL_phrase"):
		NL_phrase_embeddings = embeddings_initializer(train_loader.nl_phrase_oov_begin-1, params.phrase_hidden_size, -1.0, 1.0)
		# checkers .append(tf.check_numerics(NL_phrase_embeddings, "nan error 13"))

	with tf.name_scope("embeddings_initializer_KB_charater"):
		KB_character_embeddings = embeddings_initializer(train_loader.kb_character_oov_begin-1, params.character_hidden_size, -1.0, 1.0)
		# checkers .append(tf.check_numerics(KB_character_embeddings, "nan error 14"))

	with tf.name_scope("embeddings_initializer_KB_word"):
		KB_word_embeddings = embeddings_initializer(train_loader.kb_word_oov_begin-1, params.word_hidden_size, -1.0, 1.0)
		# checkers .append(tf.check_numerics(KB_word_embeddings, "nan error 15"))

	with tf.name_scope("embeddings_initializer_KB_phrase"):
		KB_phrase_embeddings = embeddings_initializer(train_loader.kb_phrase_oov_begin-1, params.phrase_hidden_size, -1.0, 1.0)
		# checkers .append(tf.check_numerics(KB_phrase_embeddings, "nan error 16"))

	### 开始loss计算
	### 0.预先获取OOV词和词组下层的字母和词序列embedding，并由rnn计算oov_embedding
	with tf.name_scope("get_oov_embedding"):
		NL_character_of_oov_word_embedding = tf.reshape(tf.nn.embedding_lookup(NL_character_embeddings,tf.reshape(NL_character_of_oov_word, [-1])),[-1, params.word_max_length, params.character_hidden_size])
		# print_checkers.append(NL_character_of_oov_word_embedding)
		with tf.variable_scope('NL' + 'c2w'):
			NL_oov_word_embedding = tf.concat(0, [tf.zeros([1, params.character_hidden_size], tf.float32), tf.nn.dynamic_rnn(NL_c2w_cell, NL_character_of_oov_word_embedding, dtype=tf.float32, sequence_length=length(NL_character_of_oov_word_embedding))[1]])
			# print_checkers.append(NL_oov_word_embedding)
			oov_embedding_checker = tf.check_numerics(NL_oov_word_embedding, "nan error 17")

		NL_word_of_oov_phrase_embedding = tf.reshape(tf.nn.embedding_lookup(NL_word_embeddings,tf.reshape(NL_word_of_oov_phrase, [-1])),[-1, params.phrase_max_length, params.word_hidden_size])
		NL_oov_word_of_oov_phrase_embedding = tf.reshape(tf.nn.embedding_lookup(NL_oov_word_embedding,tf.reshape(NL_oov_word_of_oov_phrase, [-1])),[-1, params.phrase_max_length, params.word_hidden_size])
		# print_checkers.append(NL_oov_word_of_oov_phrase_embedding)

		NL_word_of_oov_phrase_embedding_complete = NL_word_of_oov_phrase_embedding + NL_oov_word_of_oov_phrase_embedding
		# print_checkers.append(NL_word_of_oov_phrase_embedding_complete)
		with tf.variable_scope('NL' + 'w2p'):
			NL_oov_phrase_embedding = tf.concat(0, [tf.zeros([1, params.word_hidden_size], tf.float32), tf.nn.dynamic_rnn(NL_w2p_cell, NL_word_of_oov_phrase_embedding_complete, dtype=tf.float32, sequence_length=length(NL_word_of_oov_phrase_embedding_complete))[1]])
		
		KB_character_of_oov_word_embedding = tf.reshape(tf.nn.embedding_lookup(KB_character_embeddings,tf.reshape(KB_character_of_oov_word, [-1])),[-1, params.word_max_length, params.character_hidden_size])
		with tf.variable_scope('KB' + 'c2w'):
			KB_oov_word_embedding = tf.concat(0, [tf.zeros([1, params.character_hidden_size], tf.float32), tf.nn.dynamic_rnn(KB_c2w_cell, KB_character_of_oov_word_embedding, dtype=tf.float32, sequence_length=length(KB_character_of_oov_word_embedding))[1]])

		KB_word_of_oov_phrase_embedding = tf.reshape(tf.nn.embedding_lookup(KB_word_embeddings,tf.reshape(KB_word_of_oov_phrase, [-1])),[-1, params.phrase_max_length, params.word_hidden_size])
		KB_oov_word_of_oov_phrase_embedding = tf.reshape(tf.nn.embedding_lookup(KB_oov_word_embedding,tf.reshape(KB_oov_word_of_oov_phrase, [-1])),[-1, params.phrase_max_length, params.word_hidden_size])
		KB_word_of_oov_phrase_embedding_complete = KB_word_of_oov_phrase_embedding + KB_oov_word_of_oov_phrase_embedding
		with tf.variable_scope('KB' + 'w2p'):
			KB_oov_phrase_embedding = tf.concat(0, [tf.zeros([1, params.word_hidden_size], tf.float32), tf.nn.dynamic_rnn(KB_w2p_cell, KB_word_of_oov_phrase_embedding_complete, dtype=tf.float32, sequence_length=length(KB_word_of_oov_phrase_embedding_complete))[1]])

	### 1.获取匹配所需embedding
	# 1.1 查询字母、非OOV词和词组的embedding,注意kb_phrase_embeddings不分左右
	with tf.name_scope("look_up_embeddings"):
		with tf.name_scope("look_up_no_oov_embedding"):
			QA_NL_character_train_embedding = tf.reshape(tf.nn.embedding_lookup(NL_character_embeddings, tf.reshape(QA_NL_character_inputs, [-1])), [-1, params.qa_nl_character_max_length, params.character_hidden_size])
			# checkers .append(tf.check_numerics(QA_NL_character_train_embedding, "nan error 18"))
			QA_NL_word_train_embedding = tf.reshape(tf.nn.embedding_lookup(NL_word_embeddings, tf.reshape(QA_NL_word_inputs, [-1])), [-1, params.qa_nl_word_max_length, params.word_hidden_size])
			QA_NL_phrase_train_embedding_no_oov = tf.reshape(tf.nn.embedding_lookup(NL_phrase_embeddings, tf.reshape(QA_NL_phrase_inputs, [-1])), [-1, params.qa_nl_phrase_max_length, params.phrase_hidden_size])
			QA_KB_character_train_embedding = tf.reshape(tf.nn.embedding_lookup(KB_character_embeddings, tf.reshape(QA_KB_character_inputs, [-1])), [-1, params.qa_kb_character_max_length, params.character_hidden_size])
			QA_KB_word_train_embedding = tf.reshape(tf.nn.embedding_lookup(KB_word_embeddings, tf.reshape(QA_KB_word_inputs, [-1])), [-1, params.qa_kb_word_max_length, params.word_hidden_size])
			QA_KB_phrase_train_embedding = tf.reshape(tf.nn.embedding_lookup(KB_phrase_embeddings, tf.reshape(QA_KB_phrase_inputs, [-1])), [-1, params.qa_kb_phrase_max_length, params.phrase_hidden_size])
			QA_KB_neg_character_train_embedding = tf.reshape(tf.nn.embedding_lookup(KB_character_embeddings, tf.reshape(QA_KB_neg_character_inputs, [-1])), [-1, params.qa_kb_character_max_length, params.character_hidden_size])
			QA_KB_neg_word_train_embedding = tf.reshape(tf.nn.embedding_lookup(KB_word_embeddings, tf.reshape(QA_KB_neg_word_inputs, [-1])), [-1, params.qa_kb_word_max_length, params.word_hidden_size])
			QA_KB_neg_phrase_train_embedding = tf.reshape(tf.nn.embedding_lookup(KB_phrase_embeddings, tf.reshape(QA_KB_neg_phrase_inputs, [-1])), [-1, params.qa_kb_phrase_max_length, params.phrase_hidden_size])

		with tf.name_scope("look_up_oov_embedding"):
			# 1.2 查询OOV词和词组的embedding
			QA_NL_oov_word_train_embedding = tf.reshape(tf.nn.embedding_lookup(NL_oov_word_embedding, tf.reshape(QA_NL_oov_word_inputs, [-1])), [-1, params.qa_nl_word_max_length, params.word_hidden_size])
			QA_NL_oov_phrase_train_embedding = tf.reshape(tf.nn.embedding_lookup(NL_oov_phrase_embedding, tf.reshape(QA_NL_oov_phrase_inputs, [-1])), [-1, params.qa_nl_phrase_max_length, params.phrase_hidden_size])
			QA_KB_oov_word_train_embedding = tf.reshape(tf.nn.embedding_lookup(KB_oov_word_embedding, tf.reshape(QA_KB_oov_word_inputs, [-1])), [-1, params.qa_kb_word_max_length, params.word_hidden_size])
			QA_KB_oov_phrase_train_embedding = tf.reshape(tf.nn.embedding_lookup(KB_oov_phrase_embedding, tf.reshape(QA_KB_oov_phrase_inputs, [-1])), [-1, params.qa_kb_phrase_max_length, params.phrase_hidden_size])
			# checkers .append(tf.check_numerics(QA_KB_oov_phrase_train_embedding, "nan error 19"))

			QA_KB_neg_oov_word_train_embedding = tf.reshape(tf.nn.embedding_lookup(KB_oov_word_embedding, tf.reshape(QA_KB_neg_oov_word_inputs, [-1])), [-1, params.qa_kb_word_max_length, params.word_hidden_size])
			QA_KB_neg_oov_phrase_train_embedding = tf.reshape(tf.nn.embedding_lookup(KB_oov_phrase_embedding, tf.reshape(QA_KB_neg_oov_phrase_inputs, [-1])), [-1, params.qa_kb_phrase_max_length, params.phrase_hidden_size])

		with tf.name_scope("concat_oov_and_no_oov_embedding"):
			# 1.3 融合非OOV词、词组embedding和OOV词、词组embedding
			QA_NL_word_train_embedding = QA_NL_word_train_embedding + QA_NL_oov_word_train_embedding
			QA_NL_phrase_train_embedding = QA_NL_phrase_train_embedding_no_oov + QA_NL_oov_phrase_train_embedding
			QA_KB_word_train_embedding = QA_KB_word_train_embedding + QA_KB_oov_word_train_embedding
			QA_KB_phrase_train_embedding = QA_KB_phrase_train_embedding + QA_KB_oov_phrase_train_embedding
			QA_KB_neg_word_train_embedding = QA_KB_neg_word_train_embedding + QA_KB_neg_oov_word_train_embedding
			QA_KB_neg_phrase_train_embedding = QA_KB_neg_phrase_train_embedding + QA_KB_neg_oov_phrase_train_embedding

	# 2.计算biGRU的所有states
	with tf.name_scope("get_states"):
		with tf.name_scope("QA_NL_character_state"):
			QA_NL_character_state, _ = biGRU(QA_NL_character_train_embedding, QA_NL_character_cell_fw, QA_NL_character_cell_bw, 'QA_NL_character_fw', 'QA_NL_character_bw', False)
			# checkers .append(tf.check_numerics(QA_NL_character_state, "nan error 20"))
			# QA_NL_character_state [10, 100, 256]

		with tf.name_scope("QA_NL_word_state"):
			QA_NL_word_state, _ = biGRU(QA_NL_word_train_embedding, QA_NL_word_cell_fw, QA_NL_word_cell_bw, 'QA_NL_word_fw', 'QA_NL_word_bw', False)

		with tf.name_scope("QA_NL_phrase_state"):
			QA_NL_phrase_state, _ = biGRU(QA_NL_phrase_train_embedding, QA_NL_phrase_cell_fw, QA_NL_phrase_cell_bw, 'QA_NL_phrase_fw', 'QA_NL_phrase_bw', False)

		with tf.name_scope("QA_KB_character_state"):
			QA_KB_character_state, _ = biGRU(QA_KB_character_train_embedding, QA_KB_character_cell_fw, QA_KB_character_cell_bw, 'QA_KB_character_fw', 'QA_KB_character_bw', False)

		with tf.name_scope("QA_KB_word_state"):
			QA_KB_word_state, _ = biGRU(QA_KB_word_train_embedding, QA_KB_word_cell_fw, QA_KB_word_cell_bw, 'QA_KB_word_fw', 'QA_KB_word_bw', False)
			# QA_KB_character_state [10, 100, 256]

		with tf.name_scope("QA_KB_phrase_state"):
			QA_KB_phrase_state, _ = biGRU(QA_KB_phrase_train_embedding, QA_KB_phrase_cell_fw, QA_KB_phrase_cell_bw, 'QA_KB_phrase_fw', 'QA_KB_phrase_bw', False)

		with tf.name_scope("QA_KB_neg_character_state"):
			QA_KB_neg_character_state, _ = biGRU(QA_KB_neg_character_train_embedding, QA_KB_character_cell_fw, QA_KB_character_cell_bw, 'QA_KB_character_fw', 'QA_KB_character_bw', True)

		with tf.name_scope("QA_KB_neg_word_state"):
			QA_KB_neg_word_state, _ = biGRU(QA_KB_neg_word_train_embedding, QA_KB_word_cell_fw, QA_KB_word_cell_bw, 'QA_KB_word_fw', 'QA_KB_word_bw', True)

		with tf.name_scope("QA_KB_neg_phrase_state"):
			QA_KB_neg_phrase_state, _ = biGRU(QA_KB_neg_phrase_train_embedding, QA_KB_phrase_cell_fw, QA_KB_phrase_cell_bw, 'QA_KB_phrase_fw', 'QA_KB_phrase_bw', True)

	### 3.计算问句的每个字母、单词、词组位置的匹配分数
	with tf.name_scope("QA_pos_character_score"):
		# with tf.device('/gpu:1'):
		# 3.7计算问句的字母匹配分数（正样例）
		QA_NL_character_state_temp = tf.tile(tf.expand_dims(QA_NL_character_state,2),[1,1,params.qa_kb_character_max_length,1])
		# checkers .append(tf.check_numerics(QA_NL_character_state_temp, "nan error 21"))
		QA_KB_character_state_temp = tf.tile(tf.expand_dims(QA_KB_character_state,1),[1,params.qa_nl_character_max_length,1,1])
		# QA_NL_character_state_temp, QA_KB_character_state_tmp [10, 100, 100, 256]
		QA_concat_character_state = tf.concat(3,[QA_NL_character_state_temp,QA_KB_character_state_temp])
		# QA_concat_character_state [10, 100, 100, 512]
		QA_concat_character_state_temp = tf.reshape(QA_concat_character_state,[-1,params.character_hidden_size*4])
		with tf.variable_scope('QA_character'):
			# QA_character_unnormalized_score =tf.exp(tf.reshape(tf.matmul(tf.tanh(tf.matmul(QA_concat_character_state_temp, QA_character_weight)+QA_character_bias), QA_character_pro),[params.qa_batch_size, params.qa_nl_character_max_length, params.qa_kb_character_max_length]))
			# 将数据限制在较小的范围内！将tf.tanh()换成tf.sigmoid(),再于最外围套上一个tf.sigmoid()
			# QA_character_unnormalized_score = tf.sigmoid(tf.exp(tf.reshape(tf.matmul(tf.sigmoid(tf.matmul(QA_concat_character_state_temp, QA_character_weight)+QA_character_bias), QA_character_pro),[params.qa_batch_size, params.qa_nl_character_max_length, params.qa_kb_character_max_length])))
			QA_character_unnormalized_score = tf.exp(tf.reshape(tf.tanh(tf.matmul(QA_concat_character_state_temp, QA_character_weight)+QA_character_bias), [-1, params.qa_nl_character_max_length, params.qa_kb_character_max_length]))
			# QA_character_unnormalized_score = tf.exp(tf.reshape(tf.tanh(tf.matmul(QA_concat_character_state_temp, QA_character_weight)+QA_character_bias), [params.qa_batch_size, params.qa_nl_character_max_length, params.qa_kb_character_max_length]))
			# QA_character_unnormalized_score [10, 100, 100]

		QA_NL_character_mask_temp = tf.tile(tf.expand_dims(QA_NL_character_mask,2),[1,1,params.qa_kb_character_max_length])
		QA_KB_character_mask_temp = tf.tile(tf.expand_dims(QA_KB_character_mask,1),[1,params.qa_nl_character_max_length,1])
		# QA_NL_character_mask_temp, QA_KB_character_mask  [10, 100, 100]
		QA_character_mask = QA_NL_character_mask_temp*QA_KB_character_mask_temp
		QA_character_unnormalized_score = QA_character_unnormalized_score*QA_character_mask
		QA_NL_character_mask_rev = tf.ones_like(QA_NL_character_mask) - QA_NL_character_mask

		QA_NL_character_mask_rev_temp = tf.expand_dims(QA_NL_character_mask_rev,2)
		# QA_NL_character_mask_rev_temp [10, 100, 1]

		# 由于mask设计有缺陷，现在全部弃用mask，使用下面避免除0的方法。
		#QA_character_normalization = tf.reduce_sum(QA_character_unnormalized_score, 2, keep_dims=True)+QA_NL_character_mask_rev_temp
		QA_character_normalization_temp = tf.reduce_sum(QA_character_unnormalized_score, 2, keep_dims=True)
		QA_character_normalization = tf.ones_like(QA_character_normalization_temp) - tf.sign(tf.abs(QA_character_normalization_temp)) + QA_character_normalization_temp
		# QA_character_normalization [10, 100, 1]

		QA_character_normalized_score = QA_character_unnormalized_score/QA_character_normalization
		# checkers .append(tf.check_numerics(QA_character_normalized_score, "nan error 22"))
		# print_checkers.append(QA_character_unnormalized_score)
		# print_checkers.append(QA_character_normalization)

		# QA_character_normalized_score [10, 100, 100]
		QA_character_normalized_score_temp = tf.tile(tf.expand_dims(QA_character_normalized_score,3),[1, 1 ,1, params.character_hidden_size*2])
		# QA_character_normalized_score_temp [10, 100, 100, 256]
		QA_KB_character_attentioned_state = tf.reduce_sum(QA_character_normalized_score_temp*QA_KB_character_state_temp, 2)
		# QA_KB_character_attentioned_state [10, 100, 256]
		QA_character_matching_score = tf.reduce_sum(QA_NL_character_state*QA_KB_character_attentioned_state,2)
		# QA_character_matching_score [10, 100]
		# checkers .append(tf.check_numerics(QA_character_matching_score, "nan error 23"))

	# 3.8计算问句的字母匹配分数（负样例）
	with tf.name_scope("QA_neg_character_score"):
		QA_NL_character_state_temp2 = tf.tile(tf.expand_dims(QA_NL_character_state_temp,0),[params.qa_neg_size,1,1,1,1])
		# QA_NL_character_state_temp2 [10(neg_size), 10(batch_size), 100(ch_max), 100(ch_max), 256(2 * hidden)]

		QA_KB_neg_character_state_temp = tf.tile(tf.expand_dims(tf.tile(tf.expand_dims(QA_KB_neg_character_state,1),[1,params.qa_nl_character_max_length,1,1]),1),[1,params.qa_batch_size,1,1,1])
		QA_neg_concat_character_state = tf.concat(4,[QA_NL_character_state_temp2,QA_KB_neg_character_state_temp])
		QA_neg_concat_character_state_temp = tf.reshape(QA_neg_concat_character_state,[-1,params.character_hidden_size*4])
		with tf.variable_scope('QA_character', reuse = True):
			QA_neg_character_unnormalized_score = tf.exp(tf.reshape(tf.tanh(tf.matmul(QA_neg_concat_character_state_temp, QA_character_weight)+QA_character_bias), [params.qa_neg_size, params.qa_batch_size, params.qa_nl_character_max_length, params.qa_kb_character_max_length]))

		QA_NL_character_mask_temp2 = tf.tile(tf.expand_dims(QA_NL_character_mask_temp,0),[params.qa_neg_size,1,1,1])
		QA_KB_neg_character_mask_temp = tf.tile(tf.expand_dims(tf.tile(tf.expand_dims(QA_KB_neg_character_mask,1),[1,params.qa_nl_character_max_length,1]),1),[1,params.qa_batch_size,1,1])
		QA_neg_character_mask = QA_NL_character_mask_temp2*QA_KB_neg_character_mask_temp
		QA_neg_character_unnormalized_score = QA_neg_character_unnormalized_score*QA_neg_character_mask
		QA_NL_character_mask_rev2 = tf.tile(tf.expand_dims(QA_NL_character_mask_rev,0),[params.qa_neg_size,1,1])
		QA_NL_character_mask_rev2_temp = tf.expand_dims(QA_NL_character_mask_rev2,3)

		# 由于mask设计有缺陷，现在全部弃用mask，使用下面避免除0的方法。
		# QA_neg_character_normalization = tf.reduce_sum(QA_neg_character_unnormalized_score, 3, keep_dims=True)+QA_NL_character_mask_rev2_temp
		QA_neg_character_normalization_temp = tf.reduce_sum(QA_neg_character_unnormalized_score, 3, keep_dims=True)
		QA_neg_character_normalization = tf.ones_like(QA_neg_character_normalization_temp) - tf.sign(tf.abs(QA_neg_character_normalization_temp)) + QA_neg_character_normalization_temp

		QA_neg_character_normalized_score = QA_neg_character_unnormalized_score/QA_neg_character_normalization
		# checkers .append(tf.check_numerics(QA_neg_character_normalized_score, "nan error 24"))
		# print_checkers.append(QA_neg_character_unnormalized_score)
		# print_checkers.append(QA_neg_character_normalization)

		QA_neg_character_normalized_score_temp = tf.tile(tf.expand_dims(QA_neg_character_normalized_score,4),[1,1,1,1,params.character_hidden_size*2])
		QA_KB_neg_character_attentioned_state = tf.reduce_sum(QA_neg_character_normalized_score_temp*QA_KB_neg_character_state_temp, 3)
		QA_NL_character_state_temp3 = tf.tile(tf.expand_dims(QA_NL_character_state,0),[params.qa_neg_size,1,1,1])
		QA_neg_character_matching_score = tf.reduce_sum(QA_NL_character_state_temp3*QA_KB_neg_character_attentioned_state,3)

	#3.9计算问句的单词匹配分数（正样例）
	with tf.name_scope("QA_pos_word_score"):
		QA_NL_word_state_temp = tf.tile(tf.expand_dims(QA_NL_word_state,2),[1,1,params.qa_kb_word_max_length,1])
		QA_KB_word_state_temp = tf.tile(tf.expand_dims(QA_KB_word_state,1),[1,params.qa_nl_word_max_length,1,1])
		QA_concat_word_state = tf.concat(3,[QA_NL_word_state_temp,QA_KB_word_state_temp])
		QA_concat_word_state_temp = tf.reshape(QA_concat_word_state,[-1,params.word_hidden_size*4])
		with tf.variable_scope('QA_word'):
			QA_word_unnormalized_score = tf.exp(tf.reshape(tf.tanh(tf.matmul(QA_concat_word_state_temp, QA_word_weight)+QA_word_bias), [-1, params.qa_nl_word_max_length, params.qa_kb_word_max_length]))
			# QA_word_unnormalized_score = tf.exp(tf.reshape(tf.tanh(tf.matmul(QA_concat_word_state_temp, QA_word_weight)+QA_word_bias), [params.qa_batch_size, params.qa_nl_word_max_length, params.qa_kb_word_max_length]))

		QA_NL_word_mask_temp = tf.tile(tf.expand_dims(QA_NL_word_mask,2),[1,1,params.qa_kb_word_max_length])
		QA_KB_word_mask_temp = tf.tile(tf.expand_dims(QA_KB_word_mask,1),[1,params.qa_nl_word_max_length,1])
		QA_word_mask = QA_NL_word_mask_temp*QA_KB_word_mask_temp
		QA_word_unnormalized_score = QA_word_unnormalized_score*QA_word_mask
		QA_NL_word_mask_rev = tf.ones_like(QA_NL_word_mask) - QA_NL_word_mask
		QA_NL_word_mask_rev_temp = tf.expand_dims(QA_NL_word_mask_rev,2)

		# 由于mask设计有缺陷，现在全部弃用mask，使用下面避免除0的方法。
		# QA_word_normalization = tf.reduce_sum(QA_word_unnormalized_score, 2, keep_dims=True)+QA_NL_word_mask_rev_temp
		QA_word_normalization_temp = tf.reduce_sum(QA_word_unnormalized_score, 2, keep_dims=True)
		QA_word_normalization = tf.ones_like(QA_word_normalization_temp) - tf.sign(tf.abs(QA_word_normalization_temp)) + QA_word_normalization_temp

		QA_word_normalized_score = QA_word_unnormalized_score/QA_word_normalization
		# checkers .append(tf.check_numerics(QA_word_normalized_score, "nan error 25"))
		# print_checkers.append(QA_word_unnormalized_score)
		# print_checkers.append(QA_word_normalization)

		QA_word_normalized_score_temp = tf.tile(tf.expand_dims(QA_word_normalized_score,3),[1, 1 ,1, params.word_hidden_size*2])
		QA_KB_word_attentioned_state = tf.reduce_sum(QA_word_normalized_score_temp*QA_KB_word_state_temp, 2)
		QA_word_matching_score = tf.reduce_sum(QA_NL_word_state*QA_KB_word_attentioned_state,2)

	# 3.10计算问句的单词匹配分数（负样例）
	with tf.name_scope("QA_neg_word_score"):
		QA_NL_word_state_temp2 = tf.tile(tf.expand_dims(QA_NL_word_state_temp,0),[params.qa_neg_size,1,1,1,1])
		QA_KB_neg_word_state_temp = tf.tile(tf.expand_dims(tf.tile(tf.expand_dims(QA_KB_neg_word_state,1),[1,params.qa_nl_word_max_length,1,1]),1),[1,params.qa_batch_size,1,1,1])
		QA_neg_concat_word_state = tf.concat(4,[QA_NL_word_state_temp2,QA_KB_neg_word_state_temp])
		QA_neg_concat_word_state_temp = tf.reshape(QA_neg_concat_word_state,[-1,params.word_hidden_size*4])
		with tf.variable_scope('QA_word', reuse = True):
			QA_neg_word_unnormalized_score = tf.exp(tf.reshape(tf.tanh(tf.matmul(QA_neg_concat_word_state_temp, QA_word_weight) + QA_word_bias), [params.qa_neg_size, params.qa_batch_size, params.qa_nl_word_max_length, params.qa_kb_word_max_length]))

		QA_NL_word_mask_temp2 = tf.tile(tf.expand_dims(QA_NL_word_mask_temp,0),[params.qa_neg_size,1,1,1])
		QA_KB_neg_word_mask_temp = tf.tile(tf.expand_dims(tf.tile(tf.expand_dims(QA_KB_neg_word_mask,1),[1,params.qa_nl_word_max_length,1]),1),[1,params.qa_batch_size,1,1])
		QA_neg_word_mask = QA_NL_word_mask_temp2*QA_KB_neg_word_mask_temp
		QA_neg_word_unnormalized_score = QA_neg_word_unnormalized_score*QA_neg_word_mask
		QA_NL_word_mask_rev2 = tf.tile(tf.expand_dims(QA_NL_word_mask_rev,0),[params.qa_neg_size,1,1])
		QA_NL_word_mask_rev2_temp = tf.expand_dims(QA_NL_word_mask_rev2,3)

		# 由于mask设计有缺陷，现在全部弃用mask，使用下面避免除0的方法。
		# QA_neg_word_normalization = tf.reduce_sum(QA_neg_word_unnormalized_score, 3, keep_dims=True)+QA_NL_word_mask_rev2_temp
		QA_neg_word_normalization_temp = tf.reduce_sum(QA_neg_word_unnormalized_score, 3, keep_dims=True)
		QA_neg_word_normalization = tf.ones_like(QA_neg_word_normalization_temp) - tf.sign(tf.abs(QA_neg_word_normalization_temp)) + QA_neg_word_normalization_temp


		QA_neg_word_normalized_score = QA_neg_word_unnormalized_score/QA_neg_word_normalization
		# checkers .append(tf.check_numerics(QA_neg_word_normalized_score, "nan error 26"))
		# print_checkers.append(QA_neg_word_unnormalized_score)
		# print_checkers.append(QA_neg_word_normalization)

		QA_neg_word_normalized_score_temp = tf.tile(tf.expand_dims(QA_neg_word_normalized_score,4),[1,1,1,1,params.word_hidden_size*2])
		QA_KB_neg_word_attentioned_state = tf.reduce_sum(QA_neg_word_normalized_score_temp*QA_KB_neg_word_state_temp, 3)
		QA_NL_word_state_temp3 = tf.tile(tf.expand_dims(QA_NL_word_state,0),[params.qa_neg_size,1,1,1])
		QA_neg_word_matching_score = tf.reduce_sum(QA_NL_word_state_temp3*QA_KB_neg_word_attentioned_state,3)

	#3.11计算问句的词组匹配分数（正样例）
	with tf.name_scope("QA_pos_phrase_score"):
		QA_NL_phrase_state_temp = tf.tile(tf.expand_dims(QA_NL_phrase_state,2),[1,1,params.qa_kb_phrase_max_length,1])
		QA_KB_phrase_state_temp = tf.tile(tf.expand_dims(QA_KB_phrase_state,1),[1,params.qa_nl_phrase_max_length,1,1])
		QA_concat_phrase_state = tf.concat(3,[QA_NL_phrase_state_temp,QA_KB_phrase_state_temp])
		QA_concat_phrase_state_temp = tf.reshape(QA_concat_phrase_state,[-1,params.phrase_hidden_size*4])
		with tf.variable_scope('QA_phrase'):
			QA_phrase_unnormalized_score = tf.exp(tf.reshape(tf.tanh(tf.matmul(QA_concat_phrase_state_temp, QA_phrase_weight) + QA_phrase_bias), [-1, params.qa_nl_phrase_max_length, params.qa_kb_phrase_max_length]))
			# QA_phrase_unnormalized_score = tf.exp(tf.reshape(tf.tanh(tf.matmul(QA_concat_phrase_state_temp, QA_phrase_weight) + QA_phrase_bias), [params.qa_batch_size, params.qa_nl_phrase_max_length, params.qa_kb_phrase_max_length]))
			# checkers .append(tf.check_numerics(QA_phrase_unnormalized_score, "nan error 33"))

		QA_NL_phrase_mask_temp = tf.tile(tf.expand_dims(QA_NL_phrase_mask,2),[1,1,params.qa_kb_phrase_max_length])
		QA_KB_phrase_mask_temp = tf.tile(tf.expand_dims(QA_KB_phrase_mask,1),[1,params.qa_nl_phrase_max_length,1])
		QA_phrase_mask = QA_NL_phrase_mask_temp*QA_KB_phrase_mask_temp
		# checkers .append(tf.check_numerics(QA_phrase_mask, "nan error 32"))

		# rev 的作用就在于把tf.reduce_sum(QA_phrase_unnormalized_score, 2, keep_dims=True)的结果中的0换成1，做法麻烦了点
		# 可以效仿下面利用sign的方法
		QA_phrase_unnormalized_score = QA_phrase_unnormalized_score*QA_phrase_mask
		QA_NL_phrase_mask_rev = tf.ones_like(QA_NL_phrase_mask) - QA_NL_phrase_mask
		QA_NL_phrase_mask_rev_temp = tf.expand_dims(QA_NL_phrase_mask_rev,2)

		# 通过mask的方法仍然不能确保消除除零操作，改用下面的避免除零的方法
		# QA_phrase_normalization = tf.reduce_sum(QA_phrase_unnormalized_score, 2, keep_dims=True)+QA_NL_phrase_mask_rev_temp
		QA_phrase_normalization_temp = tf.reduce_sum(QA_phrase_unnormalized_score, 2, keep_dims=True)
		QA_phrase_normalization = tf.ones_like(QA_phrase_normalization_temp) - tf.sign(tf.abs(QA_phrase_normalization_temp)) + QA_phrase_normalization_temp
		# checkers .append(tf.check_numerics(QA_phrase_unnormalized_score, "nan error 31"))


		QA_phrase_normalized_score = QA_phrase_unnormalized_score/QA_phrase_normalization
		# checkers .append(tf.check_numerics(QA_phrase_normalized_score, "nan error 27"))
		# print_checkers.append(QA_phrase_unnormalized_score)
		# print_checkers.append(QA_phrase_normalization)

		QA_phrase_normalized_score_temp = tf.tile(tf.expand_dims(QA_phrase_normalized_score,3),[1, 1 ,1, params.phrase_hidden_size*2])
		QA_KB_phrase_attentioned_state = tf.reduce_sum(QA_phrase_normalized_score_temp*QA_KB_phrase_state_temp, 2)
		QA_phrase_matching_score = tf.reduce_sum(QA_NL_phrase_state*QA_KB_phrase_attentioned_state,2)

	# 3.12计算问句的词组匹配分数（负样例）
	with tf.name_scope("QA_neg_phrase_score"):
		QA_NL_phrase_state_temp2 = tf.tile(tf.expand_dims(QA_NL_phrase_state_temp,0),[params.qa_neg_size,1,1,1,1])
		QA_KB_neg_phrase_state_temp = tf.tile(tf.expand_dims(tf.tile(tf.expand_dims(QA_KB_neg_phrase_state,1),[1,params.qa_nl_phrase_max_length,1,1]),1),[1,params.qa_batch_size,1,1,1])
		QA_neg_concat_phrase_state = tf.concat(4,[QA_NL_phrase_state_temp2,QA_KB_neg_phrase_state_temp])
		QA_neg_concat_phrase_state_temp = tf.reshape(QA_neg_concat_phrase_state,[-1,params.phrase_hidden_size*4])
		with tf.variable_scope('QA_phrase', reuse = True):
			QA_neg_phrase_unnormalized_score = tf.exp(tf.reshape(tf.tanh(tf.matmul(QA_neg_concat_phrase_state_temp, QA_phrase_weight) + QA_phrase_bias), [params.qa_neg_size, params.qa_batch_size, params.qa_nl_phrase_max_length, params.qa_kb_phrase_max_length]))

		QA_NL_phrase_mask_temp2 = tf.tile(tf.expand_dims(QA_NL_phrase_mask_temp,0),[params.qa_neg_size,1,1,1])
		QA_KB_neg_phrase_mask_temp = tf.tile(tf.expand_dims(tf.tile(tf.expand_dims(QA_KB_neg_phrase_mask,1),[1,params.qa_nl_phrase_max_length,1]),1),[1,params.qa_batch_size,1,1])
		QA_neg_phrase_mask = QA_NL_phrase_mask_temp2*QA_KB_neg_phrase_mask_temp
		QA_neg_phrase_unnormalized_score = QA_neg_phrase_unnormalized_score*QA_neg_phrase_mask
		QA_NL_phrase_mask_rev2 = tf.tile(tf.expand_dims(QA_NL_phrase_mask_rev,0),[params.qa_neg_size,1,1])
		QA_NL_phrase_mask_rev2_temp = tf.expand_dims(QA_NL_phrase_mask_rev2,3)

		# 由于mask设计有缺陷，现在全部弃用mask，使用下面避免除0的方法。
		# QA_neg_phrase_normalization = tf.reduce_sum(QA_neg_phrase_unnormalized_score, 3, keep_dims=True)+QA_NL_phrase_mask_rev2_temp
		QA_neg_phrase_normalization_temp = tf.reduce_sum(QA_neg_phrase_unnormalized_score, 3, keep_dims=True)
		QA_neg_phrase_normalization = tf.ones_like(QA_neg_phrase_normalization_temp) - tf.sign(tf.abs(QA_neg_phrase_normalization_temp)) + QA_neg_phrase_normalization_temp

		QA_neg_phrase_normalized_score = QA_neg_phrase_unnormalized_score/QA_neg_phrase_normalization
		# checkers .append(tf.check_numerics(QA_neg_phrase_normalized_score, "nan error 28"))
		# print_checkers.append(QA_neg_phrase_unnormalized_score)
		# print_checkers.append(QA_neg_phrase_normalization)

		QA_neg_phrase_normalized_score_temp = tf.tile(tf.expand_dims(QA_neg_phrase_normalized_score,4),[1,1,1,1,params.phrase_hidden_size*2])
		QA_KB_neg_phrase_attentioned_state = tf.reduce_sum(QA_neg_phrase_normalized_score_temp*QA_KB_neg_phrase_state_temp, 3)
		QA_NL_phrase_state_temp3 = tf.tile(tf.expand_dims(QA_NL_phrase_state,0),[params.qa_neg_size,1,1,1])
		QA_neg_phrase_matching_score = tf.reduce_sum(QA_NL_phrase_state_temp3*QA_KB_neg_phrase_attentioned_state,3)

	# 4.3计算问句的最终得分（正样例）
	with tf.name_scope("QA_pos_final_score"):
		QA_character_matching_score = QA_character_matching_score*QA_NL_character_mask
		QA_word_matching_score = QA_word_matching_score*QA_NL_word_mask
		QA_phrase_matching_score = QA_phrase_matching_score*QA_NL_phrase_mask

		QA_character_matching_score_temp = tf.tile(tf.expand_dims(QA_character_matching_score,1),[1,params.qa_nl_word_max_length,1])

		# 避免除0问题
		sum1 = tf.reduce_sum(QA_c2w_tree,2)
		div1 = tf.ones_like(sum1)-tf.sign(sum1)+sum1
		QA_word_matching_score_from_lower = tf.reduce_sum(QA_character_matching_score_temp*QA_c2w_tree,2)/div1
		QA_word_matching_score = tf.maximum(QA_word_matching_score,QA_word_matching_score_from_lower)

		QA_word_matching_score_temp = tf.tile(tf.expand_dims(QA_word_matching_score,1),[1,params.qa_nl_phrase_max_length,1])
		sum2 = tf.reduce_sum(QA_w2p_tree,2)
		div2 = tf.ones_like(sum2)-tf.sign(sum2)+sum2
		QA_phrase_matching_score_from_lower = tf.reduce_sum(QA_word_matching_score_temp*QA_w2p_tree,2)/div2
		QA_phrase_matching_score = tf.maximum(QA_phrase_matching_score,QA_phrase_matching_score_from_lower)
		QA_final_score = tf.reduce_sum(QA_phrase_matching_score,1)/tf.reduce_sum(QA_NL_phrase_mask,1)

	with tf.name_scope("most_high_score_pos"):
		max_num = tf.reduce_max(QA_final_score, keep_dims=True)
		equal_flag = tf.equal(QA_final_score, max_num)
		batch_answer_pos = tf.cast(equal_flag, tf.int32)

	# 4.4计算问句的最终得分（负样例）
	with tf.name_scope("QA_neg_final_score"):
		QA_neg_character_matching_score = QA_neg_character_matching_score*tf.tile(tf.expand_dims(QA_NL_character_mask,0),[params.qa_neg_size,1,1])
		QA_neg_word_matching_score = QA_neg_word_matching_score*tf.tile(tf.expand_dims(QA_NL_word_mask,0),[params.qa_neg_size,1,1])
		QA_NL_phrase_mask_temp = tf.tile(tf.expand_dims(QA_NL_phrase_mask,0),[params.qa_neg_size,1,1])
		QA_neg_phrase_matching_score = QA_neg_phrase_matching_score*QA_NL_phrase_mask_temp

		QA_neg_character_matching_score_temp = tf.tile(tf.expand_dims(QA_neg_character_matching_score,2),[1,1,params.qa_nl_word_max_length,1])
		QA_c2w_tree_temp = tf.tile(tf.expand_dims(QA_c2w_tree,0),[params.qa_neg_size,1,1,1])
		sum3 = tf.reduce_sum(QA_c2w_tree_temp,3)
		div3 = tf.ones_like(sum3)-tf.sign(sum3)+sum3
		QA_neg_word_matching_score_from_lower = tf.reduce_sum(QA_neg_character_matching_score_temp*QA_c2w_tree_temp,3)/div3
		QA_neg_word_matching_score = tf.maximum(QA_neg_word_matching_score,QA_neg_word_matching_score_from_lower)

		QA_neg_word_matching_score_temp = tf.tile(tf.expand_dims(QA_neg_word_matching_score,2),[1,1,params.qa_nl_phrase_max_length,1])
		QA_w2p_tree_temp = tf.tile(tf.expand_dims(QA_w2p_tree,0),[params.qa_neg_size,1,1,1])
		sum4 = tf.reduce_sum(QA_w2p_tree_temp,3)
		div4 = tf.ones_like(sum4)-tf.sign(sum4)+sum4
		QA_neg_phrase_matching_score_from_lower = tf.reduce_sum(QA_neg_word_matching_score_temp*QA_w2p_tree_temp,3)/div4
		QA_neg_phrase_matching_score = tf.maximum(QA_neg_phrase_matching_score,QA_neg_phrase_matching_score_from_lower)
		QA_neg_final_score = tf.reduce_sum(QA_neg_phrase_matching_score,2)/tf.reduce_sum(QA_NL_phrase_mask_temp,2)
		# checkers .append(tf.check_numerics(QA_neg_final_score, "nan error 29"))

	with tf.name_scope("train"):
		QA_loss = tf.reduce_sum(tf.nn.relu(tf.tile(tf.expand_dims(params.gamma_qa - QA_final_score, 0), [params.qa_neg_size, 1])+QA_neg_final_score))
		# checkers .append(tf.check_numerics(QA_loss, "loss nan error 30"))
		optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(QA_loss)

	with tf.name_scope("save_variables"):
		saver = tf.train.Saver()

	with tf.name_scope("init"):
		init = tf.initialize_all_variables()


def train_op():
	with tf.Session(graph=graph, config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
		writer = tf.train.SummaryWriter("/home/deeplearning/zsg/code/Integration/logs/", sess.graph)
		sess.run(init)
		for e in range(params.epoch_num):
			train_loader.reset_batch_pointer()
			loss_all = 0
			for b in range(train_loader.qa_batch_num):
				now_batch = train_loader.next_batch()
				d = {QA_NL_character_inputs: now_batch[0],
						QA_NL_word_inputs: now_batch[1],
						QA_NL_phrase_inputs: now_batch[2],
						QA_KB_character_inputs: now_batch[3],
						QA_KB_word_inputs: now_batch[4],
						QA_KB_phrase_inputs: now_batch[5],
						QA_KB_neg_character_inputs: now_batch[6],
						QA_KB_neg_word_inputs: now_batch[7],
						QA_KB_neg_phrase_inputs: now_batch[8],
						QA_NL_oov_word_inputs: now_batch[9],
						QA_NL_oov_phrase_inputs: now_batch[10],
						QA_KB_oov_word_inputs: now_batch[11],
						QA_KB_oov_phrase_inputs: now_batch[12],
						QA_KB_neg_oov_word_inputs: now_batch[13],
						QA_KB_neg_oov_phrase_inputs: now_batch[14],
						NL_character_of_oov_word: now_batch[15],
						NL_word_of_oov_phrase: now_batch[16],
						NL_oov_word_of_oov_phrase: now_batch[17],
						KB_character_of_oov_word: now_batch[18],
						KB_word_of_oov_phrase: now_batch[19],
						KB_oov_word_of_oov_phrase: now_batch[20],
						QA_c2w_tree: now_batch[21],
						QA_w2p_tree: now_batch[22],
						QA_NL_character_mask: now_batch[23],
						QA_NL_word_mask: now_batch[24],
						QA_NL_phrase_mask: now_batch[25],
						QA_KB_character_mask: now_batch[26],
						QA_KB_word_mask: now_batch[27],
						QA_KB_phrase_mask: now_batch[28],
						QA_KB_neg_character_mask: now_batch[29],
						QA_KB_neg_word_mask: now_batch[30],
						QA_KB_neg_phrase_mask: now_batch[31]}

				print "running batch {0} of epoch {1}......".format(b, e)
				"""
				res = sess.run(print_checkers, feed_dict=d)
				print res[0]
				print "*" * 50
				print res[1]
				print "res got!"

				loss_val = sess.run(QA_loss, feed_dict=d)
				print "loss got and loss_val is ", loss_val
				"""

				# print "before optimizer"
				_ = sess.run(optimizer, feed_dict=d)
				# print "after optimizer"

				"""
				res = sess.run(print_checkers, feed_dict=d)
				print res[0]
				print "*" * 50
				print res[1]
				print "res got!"
				"""

				# loss_val = sess.run(QA_loss, feed_dict = d)
				# print "loss_val got and loss_val is ", loss_val
		if not os.path.exists("variables_saved"):
			os.makedirs("variables_saved")
		saver.save(sess, "variables_saved/variables")


def train_for_test():
	with tf.Session(graph=graph, config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
		writer = tf.train.SummaryWriter("/home/deeplearning/zsg/code/Integration/logs/", sess.graph)
		sess.run(init)
		for e in range(params.epoch_num):
			train_loader.reset_batch_pointer()
			loss_all = 0
			for b in range(1):
				now_batch = train_loader.next_batch()
				d = {QA_NL_character_inputs: now_batch[0],
						QA_NL_word_inputs: now_batch[1],
						QA_NL_phrase_inputs: now_batch[2],
						QA_KB_character_inputs: now_batch[3],
						QA_KB_word_inputs: now_batch[4],
						QA_KB_phrase_inputs: now_batch[5],
						QA_KB_neg_character_inputs: now_batch[6],
						QA_KB_neg_word_inputs: now_batch[7],
						QA_KB_neg_phrase_inputs: now_batch[8],
						QA_NL_oov_word_inputs: now_batch[9],
						QA_NL_oov_phrase_inputs: now_batch[10],
						QA_KB_oov_word_inputs: now_batch[11],
						QA_KB_oov_phrase_inputs: now_batch[12],
						QA_KB_neg_oov_word_inputs: now_batch[13],
						QA_KB_neg_oov_phrase_inputs: now_batch[14],
						NL_character_of_oov_word: now_batch[15],
						NL_word_of_oov_phrase: now_batch[16],
						NL_oov_word_of_oov_phrase: now_batch[17],
						KB_character_of_oov_word: now_batch[18],
						KB_word_of_oov_phrase: now_batch[19],
						KB_oov_word_of_oov_phrase: now_batch[20],
						QA_c2w_tree: now_batch[21],
						QA_w2p_tree: now_batch[22],
						QA_NL_character_mask: now_batch[23],
						QA_NL_word_mask: now_batch[24],
						QA_NL_phrase_mask: now_batch[25],
						QA_KB_character_mask: now_batch[26],
						QA_KB_word_mask: now_batch[27],
						QA_KB_phrase_mask: now_batch[28],
						QA_KB_neg_character_mask: now_batch[29],
						QA_KB_neg_word_mask: now_batch[30],
						QA_KB_neg_phrase_mask: now_batch[31]}
				sess.run(optimizer, feed_dict=d)
				print "running batch {0}......".format(b)

		if not os.path.exists("variables_saved"):
			os.makedirs("variables_saved")
		saver.save(sess, "variables_saved/variables")


def test_op():
	test_loader = TestTextLoader()
	correct_ans_num = 0
	with tf.Session(graph=graph, config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
		saver.restore(sess, "variables_saved/variables")

		# 缩减时间，减少batch总数，尽快得到accuracy
		# for b in range(17):

		b = 1
		for _ in range(test_loader.qa_batch_num):
			b = _ + 1
			print "running test batch {0}......".format(b)
			data = test_loader.next_batch()

			data = list(data)
			if len(data[0]) == 0:
				print "test batch {0} is empty!!!!!".format(b)
				break

			# 填充空白数据
			for i in range(len(data)):
				if len(data[i]) == 0:
					if i in [0, 3, 15, 18, 26, 29]:
						data[i] = np.zeros((len(data[0]), params.qa_nl_character_max_length))
					if i in [24]:
						data[i] = np.zeros((len(data[0]), params.qa_nl_word_max_length, qa_nl_character_max_length))
					if i in [25]:
						data[i] = np.zeros((len(data[0]), params.qa_nl_phrase_max_length, params.qa_nl_word_max_length))
					if i in [1, 4, 9, 11, 16, 17, 19, 20, 27, 30]:
						data[i] = np.zeros((len(data[0]), params.qa_nl_word_max_length))
					if i in [2, 10, 28]:
						data[i] = np.zeros((len(data[0]), params.qa_nl_phrase_max_length))
					if i in [5, 12, 31]:
						data[i] = np.zeros((len(data[0]), params.qa_kb_phrase_max_length))
			"""
			# 查看test_batch中为空的数据，及时进行填充。
			for i in range(len(data)):
				if len(data[i]) == 0:
					print "data {0} of test batch {1} is empty".format(i, b)
			"""

			answer_pos = data[35]
			d = {QA_NL_character_inputs: data[0], 
					QA_NL_word_inputs: data[1], 
					QA_NL_phrase_inputs: data[2], 
					QA_KB_character_inputs: data[3], 
					QA_KB_word_inputs: data[4], 
					QA_KB_phrase_inputs: data[5], 
					# QA_KB_character_inputs_for_answers: data[6], 
					# QA_KB_word_inputs_for_answers: data[7], 
					# QA_KB_phrase_inputs_for_answers: data[8], 
					QA_NL_oov_word_inputs: data[9], 
					QA_NL_oov_phrase_inputs: data[10], 
					QA_KB_oov_word_inputs: data[11], 
					QA_KB_oov_phrase_inputs: data[12],
					# QA_KB_oov_word_inputs_for_answers: data[13], 
					# QA_KB_oov_phrase_inputs_for_answers: data[14],
					NL_character_of_oov_word: data[15], 
					NL_word_of_oov_phrase: data[16], 
					NL_oov_word_of_oov_phrase: data[17], 
					KB_character_of_oov_word: data[18], 
					KB_word_of_oov_phrase: data[19], 
					KB_oov_word_of_oov_phrase: data[20], 
					# KB_character_of_oov_word_for_answers: data[21], 
					# KB_word_of_oov_phrase_for_answers: data[22], 
					# KB_oov_word_of_oov_phrase_for_answers: data[23], 
					QA_c2w_tree: data[24], 
					QA_w2p_tree: data[25], 
					QA_NL_character_mask: data[26], 
					QA_NL_word_mask: data[27], 
					QA_NL_phrase_mask: data[28], 
					QA_KB_character_mask: data[29], 
					QA_KB_word_mask: data[30], 
					QA_KB_phrase_mask: data[31]
					# QA_KB_character_mask_for_answers: data[32], 
					# QA_KB_word_mask_for_answers: data[33], 
					# QA_KB_phrase_mask_for_answers: data[34]
					}

			pos = sess.run(batch_answer_pos, feed_dict=d)
			for i in xrange(len(pos)):
				if pos[i] == 1 and pos[i] == answer_pos[i]:
					correct_ans_num += 1

	# accuracy = correct_ans_num / float(test_loader.qa_batch_num)
	accuracy = correct_ans_num / float(b - 1)

	# 尽快得到accuracy！
	# accuracy = correct_ans_num / 17.0
	# print "accuracy: ", accuracy

if __name__ == '__main__':
	if not os.path.exists("variables_saved"):
		train_op()
	test_op()























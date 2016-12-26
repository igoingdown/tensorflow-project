#coding=utf-8
import numpy as np
import tensorflow as tf
import params
from after_raw_data import *

#工具函数
def length(data):
	used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
	max_length = tf.reduce_sum(used, reduction_indices=1)
	max_length = tf.cast(max_length, tf.int32)
	return max_length

#由于一批训练样例使用相同的负样例，因此负样例可直接用该biGRU而不用考虑升维
def biGRU(data, cell_fw, cell_bw):
	output_fw, state_fw = tf.nn.dynamic_rnn(cell_fw, data, dtype=tf.float32, sequence_length=length(data))
	reverse_data = tf.reverse_sequence(data, seq_lengths=length(data), seq_dim=1)
	output_bw, state_bw = tf.nn.dynamic_rnn(cell_bw, reverse_data, dtype=tf.float32, sequence_length=length(reverse_data))
	#连接操作
	output = tf.concat(2, [output_fw, output_bw])
	state = tf.concat(2, [state_fw, state_bw])
	return output, state

def embeddings_initializer(dim_x, dim_y, minn, maxx):
	embeddings = tf.Variable(tf.random_uniform([dim_x-1, dim_y], minn, maxx))
	print embeddings.eval()
	embeddings = tf.nn.l2_normalize(embeddings, 1)
	embeddings = tf.concat(0, [tf.zeros([1, dim_y], tf.float32), embeddings])
	return embeddings
	
def main():
#定义loader，其加载所有训练数据并分批
	loader = Textloader()
	print 'Successfully load data!'

#定义和初始化计算流图
	graph = tf.Graph()
	with graph.as_default():

###placeholder
#hierarchical summarization训练样例的placeholder
		'''
		SF_NL_character_train_inputs = tf.placeholder(tf.int32, shape=[params.sf_batch_size, params.sf_nl_character_max_length])
		SF_NL_word_train_inputs = tf.placeholder(tf.int32, shape=[params.sf_batch_size, params.sf_nl_word_max_length])
		SF_NL_phrase_train_inputs = tf.placeholder(tf.int32, shape=[params.sf_batch_size, params.sf_nl_phrase_max_length])

		SF_KB_character_train_inputs = tf.placeholder(tf.int32, shape=[params.sf_batch_size, params.sf_kb_character_max_length])
		SF_KB_word_train_inputs = tf.placeholder(tf.int32, shape=[params.sf_batch_size, params.sf_kb_word_max_length])
		SF_KB_phrase_train_inputs = tf.placeholder(tf.int32, shape=[params.sf_batch_size, params.sf_kb_phrase_max_length])

		SF_KB_neg_character_train_inputs = tf.placeholder(tf.int32, shape=[params.sf_neg_size, params.sf_kb_character_max_length])
		SF_KB_neg_word_train_inputs = tf.placeholder(tf.int32, shape=[params.sf_neg_size, params.sf_kb_word_max_length])
		SF_KB_neg_phrase_train_inputs = tf.placeholder(tf.int32, shape=[params.sf_neg_size, params.sf_kb_phrase_max_length])
		'''
		QA_NL_character_train_inputs = tf.placeholder(tf.int32, shape=[params.qa_batch_size, params.qa_nl_character_max_length])
		QA_NL_word_train_inputs = tf.placeholder(tf.int32, shape=[params.qa_batch_size, params.qa_nl_word_max_length])
		QA_NL_phrase_train_inputs = tf.placeholder(tf.int32, shape=[params.qa_batch_size, params.qa_nl_phrase_max_length])

		QA_KB_character_train_inputs = tf.placeholder(tf.int32, shape=[params.qa_batch_size, params.qa_kb_character_max_length])
		QA_KB_word_train_inputs = tf.placeholder(tf.int32, shape=[params.qa_batch_size, params.qa_kb_word_max_length])
		QA_KB_phrase_train_inputs = tf.placeholder(tf.int32, shape=[params.qa_batch_size, params.qa_kb_phrase_max_length])

		QA_KB_neg_character_train_inputs = tf.placeholder(tf.int32, shape=[params.qa_neg_size, params.qa_kb_character_max_length])
		QA_KB_neg_word_train_inputs = tf.placeholder(tf.int32, shape=[params.qa_neg_size, params.qa_kb_word_max_length])
		QA_KB_neg_phrase_train_inputs = tf.placeholder(tf.int32, shape=[params.qa_neg_size, params.qa_kb_phrase_max_length])

#hierarchical summarization训练样例的OOV词和词组placeholder
		'''
		SF_NL_oov_word_train_inputs = tf.placeholder(tf.int32, shape=[params.sf_batch_size, params.sf_nl_word_max_length])
		SF_NL_oov_phrase_train_inputs = tf.placeholder(tf.int32, shape=[params.sf_batch_size, params.sf_nl_phrase_max_length])

		SF_KB_oov_word_train_inputs = tf.placeholder(tf.int32, shape=[params.sf_batch_size, params.sf_kb_word_max_length])
		SF_KB_oov_phrase_train_inputs = tf.placeholder(tf.int32, shape=[params.sf_batch_size, params.sf_kb_phrase_max_length])

		SF_KB_neg_oov_word_train_inputs = tf.placeholder(tf.int32, shape=[params.sf_neg_size, params.sf_kb_word_max_length])
		SF_KB_neg_oov_phrase_train_inputs = tf.placeholder(tf.int32, shape=[params.sf_neg_size, params.sf_kb_phrase_max_length])
		'''    
		QA_NL_oov_word_train_inputs = tf.placeholder(tf.int32, shape=[params.qa_batch_size, params.qa_nl_word_max_length])
		QA_NL_oov_phrase_train_inputs = tf.placeholder(tf.int32, shape=[params.qa_batch_size, params.qa_nl_phrase_max_length])

		QA_KB_oov_word_train_inputs = tf.placeholder(tf.int32, shape=[params.qa_batch_size, params.qa_kb_word_max_length])
		QA_KB_oov_phrase_train_inputs = tf.placeholder(tf.int32, shape=[params.qa_batch_size, params.qa_kb_phrase_max_length])

		QA_KB_neg_oov_word_train_inputs = tf.placeholder(tf.int32, shape=[params.qa_neg_size, params.qa_kb_word_max_length])
		QA_KB_neg_oov_phrase_train_inputs = tf.placeholder(tf.int32, shape=[params.qa_neg_size, params.qa_kb_phrase_max_length])

#OOV词和词组已被编号，需传递它们对应的下一层字母和单词信息（词组对应的单词因为也存在OOV词，所以需要传递两项信息）
#可变长度，不指定shape
		NL_character_of_oov_word = tf.placeholder(tf.int32)
		NL_word_of_oov_phrase = tf.placeholder(tf.int32)
		NL_oov_word_of_oov_phrase = tf.placeholder(tf.int32)
		KB_character_of_oov_word = tf.placeholder(tf.int32)
		KB_word_of_oov_phrase = tf.placeholder(tf.int32)
		KB_oov_word_of_oov_phrase = tf.placeholder(tf.int32)

#hierarchical summarization需要的树结构信息placeholder（哪个单词对应哪些字母，单词长度是多少...）
		'''
		SF_c2w_tree = tf.placeholder(tf.int32, shape=[params.sf_batch_size, params.sf_nl_word_max_length, params.sf_nl_character_max_length])
		SF_w2p_tree = tf.placeholder(tf.int32, shape=[params.sf_batch_size, params.sf_nl_phrase_max_length, params.sf_nl_word_max_length])
		'''
		QA_c2w_tree = tf.placeholder(tf.int32, shape=[params.sf_batch_size, params.qa_nl_word_max_length, params.qa_nl_character_max_length])
		QA_w2p_tree = tf.placeholder(tf.int32, shape=[params.sf_batch_size, params.qa_nl_phrase_max_length, params.qa_nl_word_max_length])

#hierarchical summarization需要nl端word和phrase的mask信息，然而在attention部分，需要nl和kb在所有level的mask信息
		'''
		SF_NL_character_mask = tf.placeholder(tf.int32, shape=[params.sf_batch_size, params.sf_nl_character_max_length])
		SF_NL_word_mask = tf.placeholder(tf.int32, shape=[params.sf_batch_size, params.sf_nl_word_max_length])
		SF_NL_phrase_mask = tf.placeholder(tf.int32, shape=[params.sf_batch_size, params.sf_nl_phrase_max_length])
		SF_KB_character_mask = tf.placeholder(tf.int32, shape=[params.sf_batch_size, params.sf_kb_character_max_length])
		SF_KB_word_mask = tf.placeholder(tf.int32, shape=[params.sf_batch_size, params.sf_kb_word_max_length])
		SF_KB_phrase_mask = tf.placeholder(tf.int32, shape=[params.sf_batch_size, params.sf_kb_phrase_max_length])
		SF_KB_neg_character_mask = tf.placeholder(tf.int32, shape=[params.sf_neg_size, params.sf_kb_character_max_length])
		SF_KB_neg_word_mask = tf.placeholder(tf.int32, shape=[params.sf_neg_size, params.sf_kb_word_max_length])
		SF_KB_neg_phrase_mask = tf.placeholder(tf.int32, shape=[params.sf_neg_size, params.sf_kb_phrase_max_length])
		'''
		QA_NL_character_mask = tf.placeholder(tf.int32, shape=[params.qa_batch_size, params.qa_nl_character_max_length])
		QA_NL_word_mask = tf.placeholder(tf.int32, shape=[params.qa_batch_size, params.qa_nl_word_max_length])
		QA_NL_phrase_mask = tf.placeholder(tf.int32, shape=[params.qa_batch_size, params.qa_nl_phrase_max_length])
		QA_KB_character_mask = tf.placeholder(tf.int32, shape=[params.qa_batch_size, params.qa_kb_character_max_length])
		QA_KB_word_mask = tf.placeholder(tf.int32, shape=[params.qa_batch_size, params.qa_kb_word_max_length])
		QA_KB_phrase_mask = tf.placeholder(tf.int32, shape=[params.qa_batch_size, params.qa_kb_phrase_max_length])
		QA_KB_neg_character_mask = tf.placeholder(tf.int32, shape=[params.qa_neg_size, params.qa_kb_character_max_length])
		QA_KB_neg_word_mask = tf.placeholder(tf.int32, shape=[params.qa_neg_size, params.qa_kb_word_max_length])
		QA_KB_neg_phrase_mask = tf.placeholder(tf.int32, shape=[params.qa_neg_size, params.qa_kb_phrase_max_length])

###variable
#statement-fact匹配，从character到phrase共三层，每层有NL和KB两侧，每侧需要两个不同方向的GRU
		'''
		with tf.variable_scope('SF' + 'NL' + 'character'):
			SF_NL_character_cell_fw = tf.nn.rnn_cell.GRUCell(params.character_hidden_size)
			SF_NL_character_cell_bw = tf.nn.rnn_cell.GRUCell(params.character_hidden_size)
		with tf.variable_scope('SF' + 'NL' + 'word'):	
			SF_NL_word_cell_fw = tf.nn.rnn_cell.GRUCell(params.word_hidden_size)
			SF_NL_word_cell_bw = tf.nn.rnn_cell.GRUCell(params.word_hidden_size)
		with tf.variable_scope('SF' + 'NL' + 'phrase'):	
			SF_NL_phrase_cell_fw = tf.nn.rnn_cell.GRUCell(params.phrase_hidden_size)
			SF_NL_phrase_cell_bw = tf.nn.rnn_cell.GRUCell(params.phrase_hidden_size)
		with tf.variable_scope('SF' + 'KB' + 'character'):	
			SF_KB_character_cell_fw = tf.nn.rnn_cell.GRUCell(params.character_hidden_size)
			SF_KB_character_cell_bw = tf.nn.rnn_cell.GRUCell(params.character_hidden_size)
		with tf.variable_scope('SF' + 'KB' + 'word'):	
			SF_KB_word_cell_fw = tf.nn.rnn_cell.GRUCell(params.word_hidden_size)
			SF_KB_word_cell_bw = tf.nn.rnn_cell.GRUCell(params.word_hidden_size)
		with tf.variable_scope('SF' + 'KB' + 'phrase'):	
			SF_KB_phrase_cell_fw = tf.nn.rnn_cell.GRUCell(params.phrase_hidden_size)
			SF_KB_phrase_cell_bw = tf.nn.rnn_cell.GRUCell(params.phrase_hidden_size)
		'''
#question-answer匹配亦然
		with tf.variable_scope('QA' + 'NL' + 'character'):
			QA_NL_character_cell_fw = tf.nn.rnn_cell.GRUCell(params.character_hidden_size)
			QA_NL_character_cell_bw = tf.nn.rnn_cell.GRUCell(params.character_hidden_size)
		with tf.variable_scope('QA' + 'NL' + 'word'):
			QA_NL_word_cell_fw = tf.nn.rnn_cell.GRUCell(params.word_hidden_size)
			QA_NL_word_cell_bw = tf.nn.rnn_cell.GRUCell(params.word_hidden_size)
		with tf.variable_scope('QA' + 'NL' + 'phrase'):	
			QA_NL_phrase_cell_fw = tf.nn.rnn_cell.GRUCell(params.phrase_hidden_size)
			QA_NL_phrase_cell_bw = tf.nn.rnn_cell.GRUCell(params.phrase_hidden_size)
		with tf.variable_scope('QA' + 'KB' + 'character'):
			QA_KB_character_cell_fw = tf.nn.rnn_cell.GRUCell(params.character_hidden_size)
			QA_KB_character_cell_bw = tf.nn.rnn_cell.GRUCell(params.character_hidden_size)
		with tf.variable_scope('QA' + 'KB' + 'word'):
			QA_KB_word_cell_fw = tf.nn.rnn_cell.GRUCell(params.word_hidden_size)
			QA_KB_word_cell_bw = tf.nn.rnn_cell.GRUCell(params.word_hidden_size)
		with tf.variable_scope('QA' + 'KB' + 'phrase'):
			QA_KB_phrase_cell_fw = tf.nn.rnn_cell.GRUCell(params.phrase_hidden_size)
			QA_KB_phrase_cell_bw = tf.nn.rnn_cell.GRUCell(params.phrase_hidden_size)

#statement-fact计算attention score，从character到phrase共三层，每层需要一个weight,一个bias和一个projection(注意江乐原来分开写了)
		'''
		with tf.variable_scope('SF' + 'character'):
			SF_character_weight = tf.Variable(tf.truncated_normal([(params.character_hidden_size+params.character_hidden_size)*2, params.character_hidden_size], -1.0, 1.0))
			SF_character_bias = tf.Variable(tf.zeros([params.character_hidden_size]))
			SF_character_pro = tf.Variable(tf.zeros([params.character_hidden_size]))
		with tf.variable_scope('SF' + 'word'):
			SF_word_weight = tf.Variable(tf.truncated_normal([(params.word_hidden_size+params.word_hidden_size)*2, params.character_hidden_size], -1.0, 1.0))
			SF_word_bias = tf.Variable(tf.zeros([params.word_hidden_size]))
			SF_word_pro = tf.Variable(tf.zeros([params.word_hidden_size]))
		with tf.variable_scope('SF' + 'phrase'):
			SF_phrase_weight = tf.Variable(tf.truncated_normal([(params.phrase_hidden_size+params.phrase_hidden_size)*2, params.phrase_hidden_size], -1.0, 1.0))
			SF_phrase_bias = tf.Variable(tf.zeros([params.phrase_hidden_size]))
			SF_phrase_pro = tf.Variable(tf.zeros([params.phrase_hidden_size]))
		'''
		with tf.variable_scope('QA' + 'character'):
			QA_character_weight = tf.Variable(tf.truncated_normal([(params.character_hidden_size+params.character_hidden_size)*2, params.character_hidden_size], -1.0, 1.0))
			QA_character_bias = tf.Variable(tf.zeros([params.character_hidden_size]))
			QA_character_pro = tf.Variable(tf.zeros([params.character_hidden_size]))
		with tf.variable_scope('QA' + 'word'):
			QA_word_weight = tf.Variable(tf.truncated_normal([(params.word_hidden_size+params.word_hidden_size)*2, params.word_hidden_size], -1.0, 1.0))
			QA_word_bias = tf.Variable(tf.zeros([params.word_hidden_size]))
			QA_word_pro = tf.Variable(tf.zeros([params.word_hidden_size]))
		with tf.variable_scope('QA' + 'phrase'):
			QA_phrase_weight = tf.Variable(tf.truncated_normal([(params.phrase_hidden_size+params.phrase_hidden_size)*2, params.phrase_hidden_size], -1.0, 1.0))
			QA_phrase_bias = tf.Variable(tf.zeros([params.phrase_hidden_size]))
			QA_phrase_pro = tf.Variable(tf.zeros([params.phrase_hidden_size]))
#计算OOV词和词组的embedding，需要4个单向GRU
		with tf.variable_scope('NL' + 'c2w'):
			NL_c2w_cell = tf.nn.rnn_cell.GRUCell(params.character_hidden_size)
		with tf.variable_scope('NL' + 'w2p'):
			NL_w2p_cell = tf.nn.rnn_cell.GRUCell(params.word_hidden_size)
		with tf.variable_scope('KB' + 'c2w'):
			KB_c2w_cell = tf.nn.rnn_cell.GRUCell(params.character_hidden_size)
		with tf.variable_scope('KB' + 'w2p'):
			KB_w2p_cell = tf.nn.rnn_cell.GRUCell(params.word_hidden_size)

#Multi-grained embedding由非监督学习获得，此处先直接随机生成
		nl_character_embeddings = embeddings_initializer(loader.nl_character_oov_begin-1, params.character_hidden_size, -1.0, 1.0)
		nl_word_embeddings = embeddings_initializer(loader.nl_word_oov_begin-1, params.word_hidden_size, -1.0, 1.0)
		nl_phrase_embeddings = embeddings_initializer(loader.nl_phrase_oov_begin-1, params.phrase_hidden_size, -1.0, 1.0)
		kb_character_embeddings = embeddings_initializer(loader.kb_character_oov_begin-1, params.character_hidden_size, -1.0, 1.0)
		kb_word_embeddings = embeddings_initializer(loader.kb_word_oov_begin-1, params.word_hidden_size, -1.0, 1.0)
		kb_phrase_embeddings = embeddings_initializer(loader.kb_phrase_oov_begin-1, params.phrase_hidden_size, -1.0, 1.0)

###开始loss计算
###0.预先获取OOV词和词组下层的字母和词序列embedding，并由rnn计算oov_embedding
		NL_character_of_oov_word_embedding = tf.reshape(tf.nn.embedding_lookup(nl_character_embeddings,tf.reshape(NL_character_of_oov_word, [-1])),[-1, params.word_max_length, params.character_hidden_size])
		NL_oov_word_embedding = tf.concat(0, [tf.zeros([1, params.character_hidden_size], tf.float32), tf.nn.dynamic_rnn(NL_c2w_cell, NL_character_of_oov_word_embedding, dtype=tf.float32, sequence_length=length(NL_character_of_oov_word_embedding))])
		NL_word_of_oov_phrase_embedding = tf.reshape(tf.nn.embedding_lookup(nl_word_embeddings,tf.reshape(NL_word_of_oov_phrase, [-1])),[-1, params.phrase_max_length, params.word_hidden_size])

		NL_oov_word_of_oov_phrase_embedding = tf.reshape(tf.nn.embedding_lookup(NL_oov_word_embedding,tf.reshape(NL_oov_word_of_oov_phrase, [-1])),[-1, params.phrase_max_length, params.word_hidden_size])
		NL_word_of_oov_phrase_embedding_complete = NL_word_of_oov_phrase_embedding + NL_oov_word_of_oov_phrase_embedding
		NL_oov_phrase_embedding = tf.concat(0, [tf.zeros([1, params.word_hidden_size], tf.float32), tf.nn.dynamic_rnn(NL_w2p_cell, NL_word_of_oov_phrase_embedding_complete, dtype=tf.float32, sequence_length=length(NL_word_of_oov_phrase_embedding_complete))])
		
		KB_character_of_oov_word_embedding = tf.reshape(tf.nn.embedding_lookup(kb_character_embeddings,tf.reshape(KB_character_of_oov_word, [-1])),[-1, params.word_max_length, params.character_hidden_size])
		KB_oov_word_embedding = tf.concat(0, [tf.zeros([1, params.character_hidden_size], tf.float32), tf.nn.dynamic_rnn(KB_c2w_cell, KB_character_of_oov_word_embedding, dtype=tf.float32, sequence_length=length(KB_character_of_oov_word_embedding))])
		KB_word_of_oov_phrase_embedding = tf.reshape(tf.nn.embedding_lookup(kb_word_embeddings,tf.reshape(KB_word_of_oov_phrase, [-1])),[-1, params.phrase_max_length, params.word_hidden_size])
		KB_oov_word_of_oov_phrase_embedding = tf.reshape(tf.nn.embedding_lookup(KB_oov_word_embedding,tf.reshape(KB_oov_word_of_oov_phrase, [-1])),[-1, params.phrase_max_length, params.word_hidden_size])
		KB_word_of_oov_phrase_embedding_complete = KB_word_of_oov_phrase_embedding + KB_oov_word_of_oov_phrase_embedding
		KB_oov_phrase_embedding = tf.concat(0, [tf.zeros([1, params.word_hidden_size], tf.float32), tf.nn.dynamic_rnn(KB_w2p_cell, KB_word_of_oov_phrase_embedding_complete, dtype=tf.float32, sequence_length=length(KB_word_of_oov_phrase_embedding_complete))])


###1.获取匹配所需embedding
#1.1 查询字母、非OOV词和词组的embedding,注意kb_phrase_embeddings不分左右
	'''
		SF_NL_character_train_inputs = tf.reshape(SF_NL_character_train_inputs, [-1])
		SF_NL_word_train_inputs = tf.reshape(SF_NL_word_train_inputs, [-1])
		SF_NL_phrase_train_inputs = tf.reshape(SF_NL_phrase_train_inputs, [-1])
		SF_KB_character_train_inputs = tf.reshape(SF_KB_character_train_inputs, [-1])
		SF_KB_word_train_inputs = tf.reshape(SF_KB_word_train_inputs, [-1])
		SF_KB_phrase_train_inputs = tf.reshape(SF_KB_phrase_train_inputs, [-1])
		SF_KB_neg_character_train_inputs = tf.reshape(SF_KB_neg_character_train_inputs, [-1])
		SF_KB_neg_word_train_inputs = tf.reshape(SF_KB_neg_word_train_inputs, [-1])
		SF_KB_neg_phrase_train_inputs = tf.reshape(SF_KB_neg_phrase_train_inputs, [-1])

		SF_NL_character_train_embedding = tf.reshape(tf.nn.embedding_lookup(nl_character_embeddings_lf, SF_NL_character_train_inputs), [-1, params.sf_nl_character_max_length, params.character_hidden_size])
		SF_NL_word_train_embedding = tf.reshape(tf.nn.embedding_lookup(nl_word_embeddings_lf, SF_NL_word_train_inputs), [-1, params.sf_nl_word_max_length, params.word_hidden_size])
		SF_NL_phrase_train_embedding = tf.reshape(tf.nn.embedding_lookup(nl_phrase_embeddings_lf, SF_NL_phrase_train_inputs), [-1, params.sf_nl_phrase_max_length, params.phrase_hidden_size])
		SF_KB_character_train_embedding = tf.reshape(tf.nn.embedding_lookup(kb_character_embeddings_lf, SF_KB_character_train_inputs), [-1, params.sf_kb_character_max_length, params.character_hidden_size])
		SF_KB_word_train_embedding = tf.reshape(tf.nn.embedding_lookup(kb_word_embeddings_lf, SF_KB_word_train_inputs), [-1, params.sf_kb_word_max_length, params.word_hidden_size])
		SF_KB_phrase_train_embedding = tf.reshape(tf.nn.embedding_lookup(kb_phrase_embeddings, SF_KB_phrase_train_inputs), [-1, params.sf_kb_phrase_max_length, params.phrase_hidden_size])
		SF_KB_neg_character_train_embedding = tf.reshape(tf.nn.embedding_lookup(kb_character_embeddings_lf, SF_KB_neg_character_train_inputs), [-1, params.sf_neg_size, params.sf_kb_character_max_length, params.character_hidden_size])
		SF_KB_neg_word_train_embedding = tf.reshape(tf.nn.embedding_lookup(kb_word_embeddings_lf, SF_KB_neg_word_train_inputs), [-1, params.sf_neg_size, params.sf_kb_word_max_length, params.word_hidden_size])
		SF_KB_neg_phrase_train_embedding = tf.reshape(tf.nn.embedding_lookup(kb_phrase_embeddings, SF_KB_neg_phrase_train_inputs), [-1, params.sf_neg_size, params.sf_kb_phrase_max_lengthh, params.phrase_hidden_size])
	'''
		QA_NL_character_train_inputs = tf.reshape(QA_NL_character_train_inputs, [-1])
		QA_NL_word_train_inputs = tf.reshape(QA_NL_word_train_inputs, [-1])
		QA_NL_phrase_train_inputs = tf.reshape(QA_NL_phrase_train_inputs, [-1])
		QA_KB_character_train_inputs = tf.reshape(QA_KB_character_train_inputs, [-1])
		QA_KB_word_train_inputs = tf.reshape(QA_KB_word_train_inputs, [-1])
		QA_KB_phrase_train_inputs = tf.reshape(QA_KB_phrase_train_inputs, [-1])
		QA_KB_neg_character_train_inputs = tf.reshape(QA_KB_neg_character_train_inputs, [-1])
		QA_KB_neg_word_train_inputs = tf.reshape(QA_KB_neg_word_train_inputs, [-1])
		QA_KB_neg_phrase_train_inputs = tf.reshape(QA_KB_neg_phrase_train_inputs, [-1])

		QA_NL_character_train_embedding = tf.reshape(tf.nn.embedding_lookup(nl_character_embeddings_lf, QA_NL_character_train_inputs), [-1, params.qa_nl_character_max_length, params.character_hidden_size])
		QA_NL_word_train_embedding = tf.reshape(tf.nn.embedding_lookup(nl_word_embeddings_lf, QA_NL_word_train_inputs), [-1, params.qa_nl_word_max_length params.word_hidden_size])
		QA_NL_phrase_train_embedding = tf.reshape(tf.nn.embedding_lookup(nl_phrase_embeddings_lf, QA_NL_phrase_train_inputs), [-1, params.qa_nl_phrase_max_length, params.phrase_hidden_size])
		QA_KB_character_train_embedding = tf.reshape(tf.nn.embedding_lookup(kb_character_embeddings_lf, QA_KB_character_train_inputs), [-1, params.qa_kb_character_max_length, params.character_hidden_size])
		QA_KB_word_train_embedding = tf.reshape(tf.nn.embedding_lookup(kb_word_embeddings_lf, QA_KB_word_train_inputs), [-1, params.qa_kb_word_max_length, params.word_hidden_size])
		QA_KB_phrase_train_embedding = tf.reshape(tf.nn.embedding_lookup(kb_phrase_embeddings, QA_KB_phrase_train_inputs), [-1, params.qa_kb_phrase_max_length, params.phrase_hidden_size])
		QA_KB_neg_character_train_embedding = tf.reshape(tf.nn.embedding_lookup(kb_character_embeddings_lf, QA_KB_neg_character_train_inputs), [-1, params.qa_neg_size, params.qa_kb_character_max_length, params.character_hidden_size])
		QA_KB_neg_word_train_embedding = tf.reshape(tf.nn.embedding_lookup(kb_word_embeddings_lf, QA_KB_neg_word_train_inputs), [-1, params.qa_neg_size, params.qa_kb_word_max_length, params.word_hidden_size])
		QA_KB_neg_phrase_train_embedding = tf.reshape(tf.nn.embedding_lookup(kb_phrase_embeddings, QA_KB_neg_phrase_train_inputs), [-1, params.qa_neg_size, params.qa_kb_phrase_max_lengthh, params.phrase_hidden_size])
#1.2 查询OOV词和词组的embedding
	'''
		SF_NL_oov_word_train_inputs = tf.reshape(SF_NL_oov_word_train_inputs, [-1])
		SF_NL_oov_phrase_train_inputs = tf.reshape(SF_NL_oov_phrase_train_inputs, [-1])
		SF_KB_oov_word_train_inputs = tf.reshape(SF_KB_oov_word_train_inputs, [-1])
		SF_KB_oov_phrase_train_inputs = tf.reshape(SF_KB_oov_phrase_train_inputs, [-1])
		SF_KB_neg_oov_word_train_inputs = tf.reshape(SF_KB_oov_neg_word_train_inputs, [-1])
		SF_KB_neg_oov_phrase_train_inputs = tf.reshape(SF_KB_oov_neg_phrase_train_inputs, [-1])

		SF_NL_oov_word_train_embedding = tf.reshape(tf.nn.embedding_lookup(NL_oov_word_embedding, SF_NL_oov_word_train_inputs), [-1, params.sf_nl_word_max_length, params.word_hidden_size])
		SF_NL_oov_phrase_train_embedding = tf.reshape(tf.nn.embedding_lookup(NL_oov_phrase_embedding, SF_NL_oov_phrase_train_inputs), [-1, params.sf_nl_phrase_max_length, params.phrase_hidden_size])
		SF_KB_oov_word_train_embedding = tf.reshape(tf.nn.embedding_lookup(KB_oov_word_embedding, SF_KB_oov_word_train_inputs), [-1, params.sf_kb_word_max_length, params.word_hidden_size])
		SF_KB_oov_phrase_train_embedding = tf.reshape(tf.nn.embedding_lookup(KB_oov_phrase_embedding, SF_KB_oov_phrase_train_inputs), [-1, params.sf_kb_phrase_max_length, params.phrase_hidden_size])
		SF_KB_oov_neg_word_train_embedding = tf.reshape(tf.nn.embedding_lookup(KB_oov_word_embedding, SF_KB_neg_oov_word_train_inputs), [-1, params.sf_neg_size, params.sf_kb_word_max_length, params.word_hidden_size])
		SF_KB_oov_neg_phrase_train_embedding = tf.reshape(tf.nn.embedding_lookup(KB_oov_phrase_embedding, SF_KB_neg_oov_phrase_train_inputs), [-1, params.sf_neg_size, params.sf_kb_phrase_max_lengthh, params.phrase_hidden_size])
	'''
		QA_NL_oov_word_train_inputs = tf.reshape(QA_NL_oov_word_train_inputs, [-1])
		QA_NL_oov_phrase_train_inputs = tf.reshape(QA_NL_oov_phrase_train_inputs, [-1])
		QA_KB_oov_word_train_inputs = tf.reshape(QA_KB_oov_word_train_inputs, [-1])
		QA_KB_oov_phrase_train_inputs = tf.reshape(QA_KB_oov_phrase_train_inputs, [-1])
		QA_KB_neg_oov_word_train_inputs = tf.reshape(QA_KB_oov_neg_word_train_inputs, [-1])
		QA_KB_neg_oov_phrase_train_inputs = tf.reshape(QA_KB_oov_neg_phrase_train_inputs, [-1])

		QA_NL_oov_word_train_embedding = tf.reshape(tf.nn.embedding_lookup(NL_oov_word_embedding, QA_NL_oov_word_train_inputs), [-1, params.qa_nl_word_max_length, params.word_hidden_size])
		QA_NL_oov_phrase_train_embedding = tf.reshape(tf.nn.embedding_lookup(NL_oov_phrase_embedding, QA_NL_oov_phrase_train_inputs), [-1, params.qa_nl_phrase_max_length, params.phrase_hidden_size])
		QA_KB_oov_word_train_embedding = tf.reshape(tf.nn.embedding_lookup(KB_oov_word_embedding, QA_KB_oov_word_train_inputs), [-1, params.qa_kb_word_max_length, params.word_hidden_size])
		QA_KB_oov_phrase_train_embedding = tf.reshape(tf.nn.embedding_lookup(KB_oov_phrase_embedding, QA_KB_oov_phrase_train_inputs), [-1, params.qa_kb_phrase_max_length, params.phrase_hidden_size])
		QA_KB_oov_neg_word_train_embedding = tf.reshape(tf.nn.embedding_lookup(KB_oov_word_embedding, QA_KB_neg_oov_word_train_inputs), [-1, params.qa_neg_size, params.qa_kb_word_max_length, params.word_hidden_size])
		QA_KB_oov_neg_phrase_train_embedding = tf.reshape(tf.nn.embedding_lookup(KB_oov_phrase_embedding, QA_KB_neg_oov_phrase_train_inputs), [-1, params.qa_neg_size, params.qa_kb_phrase_max_lengthh, params.phrase_hidden_size])
#1.3 融合非OOV词、词组embedding和OOV词、词组embedding
	'''
		SF_NL_word_train_embedding = SF_NL_word_train_embedding + SF_NL_oov_word_train_embedding
		SF_NL_phrase_train_embedding = SF_NL_phrase_train_embedding + SF_NL_oov_phrase_train_embedding
		SF_KB_word_train_embedding = SF_KB_word_train_embedding + SF_KB_oov_word_train_embedding
		SF_KB_phrase_train_embedding = SF_KB_phrase_train_embedding + SF_KB_oov_phrase_train_embedding
		SF_KB_neg_word_train_embedding = SF_KB_neg_word_train_embedding + SF_KB_neg_oov_word_train_embedding
		SF_KB_neg_phrase_train_embedding = SF_KB_neg_phrase_train_embedding + SF_KB_neg_oov_phrase_train_embedding
	'''	
		QA_NL_word_train_embedding = QA_NL_word_train_embedding + QA_NL_oov_word_train_embedding
		QA_NL_phrase_train_embedding = QA_NL_phrase_train_embedding + QA_NL_oov_phrase_train_embedding
		QA_KB_word_train_embedding = QA_KB_word_train_embedding + QA_KB_oov_word_train_embedding
		QA_KB_phrase_train_embedding = QA_KB_phrase_train_embedding + QA_KB_oov_phrase_train_embedding
		QA_KB_neg_word_train_embedding = QA_KB_neg_word_train_embedding + QA_KB_neg_oov_word_train_embedding
		QA_KB_neg_phrase_train_embedding = QA_KB_neg_phrase_train_embedding + QA_KB_neg_oov_phrase_train_embedding

#2.计算biGRU的所有states
	'''
		SF_NL_character_state, _ = biGRU(SF_NL_character_train_embedding, SF_NL_character_cell_fw, SF_NL_character_cell_bw)
		SF_NL_word_state, _ = biGRU(SF_NL_word_train_embedding, SF_NL_word_cell_fw, SF_NL_word_cell_bw)
		SF_NL_phrase_state, _ = biGRU(SF_NL_phrase_train_embedding, SF_NL_phrase_cell_fw, SF_NL_phrase_cell_bw)
		SF_KB_character_state, _ = biGRU(SF_KB_character_train_embedding, SF_KB_character_cell_fw, SF_KB_character_cell_bw)
		SF_KB_word_state, _ = biGRU(SF_KB_word_train_embedding, SF_KB_word_cell_fw, SF_KB_word_cell_bw)
		SF_KB_phrase_state, _ = biGRU(SF_KB_phrase_train_embedding, SF_KB_phrase_cell_fw, SF_KB_phrase_cell_bw)
		SF_KB_neg_character_state, _ = biGRU(SF_KB_neg_character_train_embedding, SF_KB_character_cell_fw, SF_KB_character_cell_bw)
		SF_KB_neg_word_state, _ = biGRU(SF_KB_neg_word_train_embedding, SF_KB_word_cell_fw, SF_KB_word_cell_bw)
		SF_KB_neg_phrase_state, _ = biGRU(SF_KB_neg_phrase_train_embedding, SF_KB_phrase_cell_fw, SF_KB_phrase_cell_bw)
	'''
		QA_NL_character_state, _ = biGRU(QA_NL_character_train_embedding, QA_NL_character_cell_fw, QA_NL_character_cell_bw)
		QA_NL_word_state, _ = biGRU(QA_NL_word_train_embedding, QA_NL_word_cell_fw, QA_NL_word_cell_bw)
		QA_NL_phrase_state, _ = biGRU(QA_NL_phrase_train_embedding, QA_NL_phrase_cell_fw, QA_NL_phrase_cell_bw)
		QA_KB_character_state, _ = biGRU(QA_KB_character_train_embedding, QA_KB_character_cell_fw, QA_KB_character_cell_bw)
		QA_KB_word_state, _ = biGRU(QA_KB_word_train_embedding, QA_KB_word_cell_fw, QA_KB_word_cell_bw)
		QA_KB_phrase_state, _ = biGRU(QA_KB_phrase_train_embedding, QA_KB_phrase_cell_fw, QA_KB_phrase_cell_bw)
		QA_KB_neg_character_state, _ = biGRU(QA_KB_neg_character_train_embedding, QA_KB_character_cell_fw, QA_KB_character_cell_bw)
		QA_KB_neg_word_state, _ = biGRU(QA_KB_neg_word_train_embedding, QA_KB_word_cell_fw, QA_KB_word_cell_bw)
		QA_KB_neg_phrase_state, _ = biGRU(QA_KB_neg_phrase_train_embedding, QA_KB_phrase_cell_fw, QA_KB_phrase_cell_bw)
'''
###3.计算声明或问句的每个字母、单词、词组位置的匹配分数
#3.1计算声明的字母匹配分数（正样例）
#将每个样例的每个nl_character和每个kb_character的state拼接
		SF_NL_character_state_temp = tf.tile(tf.expand_dims(SF_NL_character_state,2),[1,1,params.sf_kb_character_max_length,1])
		SF_KB_character_state_temp = tf.tile(tf.expand_dims(SF_KB_character_state,1),[1,params.sf_nl_character_max_length,1，1])
		SF_concat_character_state = tf.concat(3,[SF_NL_character_state_temp,SF_KB_character_state_temp])
#四维变二维，方便感知机运算
		SF_concat_character_state_temp = tf.reshape(SF_concat_character_state,[-1,params.character_hidden_size*4])
		SF_character_unnormalized_score = tf.reshape(tf.mul(tf.tanh(tf.mul(SF_concat_character_state_temp, SF_character_weight)+SF_character_bias), SF_character_pro),[params.sf_batch_size, params.sf_nl_character_max_length, params.sf_kb_character_max_length])
#制作三维mask,并作用于unnormalized_score
		SF_NL_character_mask_temp = tf.tile(tf.expand_dims(SF_NL_character_mask,2),[1,1,params.sf_kb_character_max_length])
		SF_KB_character_mask_temp = tf.tile(tf.expand_dims(SF_KB_character_mask,1),[1,params.sf_nl_character_max_length,1])
		SF_character_mask = SF_NL_character_mask_temp*SF_KB_character_mask_temp
		SF_character_unnormalized_score = SF_character_unnormalized_score*SF_character_mask
#每个nl_character对应的所有kb_character的score归一化
		SF_character_normalized_score = SF_character_unnormalized_score/tf.reduce_sum(SF_character_unnormalized_score, 2, keep_dims=True)
#分别扩张attention_score和kb_character_state的维度，便于两者相乘
		SF_character_normalized_score_temp = tf.tile(tf.expand_dims(SF_character_normalized_score,3),[1, 1 ,1, params.character_hidden_size*2])
#attention后的kb_character_state，已沿kb_character加和降维，跟SF_NL_character_state形状一样了
		SF_KB_character_attentioned_state = tf.reduce_sum(SF_character_normalized_score_temp*SF_KB_character_state_temp, 2)
#内积，得到的matching_score维度是bs*sf_nl_character_max_length
		SF_character_matching_score = tf.reduce_sum(SF_NL_character_state*SF_KB_character_attentioned_state,2)
#3.2计算声明的字母匹配分数（负样例，注意整个运算过程升了一个维度）
		SF_NL_character_state_temp2 = tf.tile(tf.expand_dims(SF_NL_character_state_temp,0),[params.sf_neg_size,1,1,1,1])
		SF_KB_neg_character_state_temp = tf.tile(tf.expand_dims(tf.tile(tf.expand_dims(SF_KB_neg_character_state,1),[1,params.sf_nl_character_max_length,1,1]),1),[1,params.sf_batch_size,1,1,1])
		SF_neg_concat_character_state = tf.concat(4,[SF_NL_character_state_temp2,SF_KB_neg_character_state_temp])

		SF_neg_concat_character_state_temp = tf.reshape(SF_neg_concat_character_state,[-1,params.character_hidden_size*4])
		SF_neg_character_unnormalized_score = tf.reshape(tf.mul(tf.tanh(tf.mul(SF_neg_concat_character_state_temp, SF_character_weight)+SF_character_bias), SF_character_pro),[params.sf_neg_size,params.sf_batch_size, params.sf_nl_character_max_length, params.sf_kb_character_max_length])

		SF_NL_character_mask_temp2 = tf.tile(tf.expand_dims(SF_NL_character_mask_temp,0),[params.sf_neg_size,1,1,1])
		SF_KB_neg_character_mask_temp = tf.tile(tf.expand_dims(tf.tile(tf.expand_dims(SF_KB_neg_character_mask,1),[1,params.sf_nl_character_max_length,1]),1),[1,params.sf_batch_size,1,1])
		SF_neg_character_mask = SF_NL_character_mask_temp2*SF_KB_neg_character_mask_temp
		SF_neg_character_unnormalized_score = SF_neg_character_unnormalized_score*SF_neg_character_mask

		SF_neg_character_normalized_score = SF_neg_character_unnormalized_score/tf.reduce_sum(SF_neg_character_unnormalized_score, 3, keep_dims=True)

		SF_neg_character_normalized_score_temp = tf.tile(tf.expand_dims(SF_neg_character_normalized_score,4),[1,1,1,1,params.character_hidden_size*2])

		SF_KB_neg_character_attentioned_state = tf.reduce_sum(SF_neg_character_normalized_score_temp*SF_KB_neg_character_state_temp, 3)
#注意结果多了一维，在最外面，大小是sf_neg_size,即sf_neg_size*sf_batch_size*sf_nl_character_max_length
		SF_NL_character_state_temp3 = tf.tile(tf.expand_dims(SF_NL_character_state,0),[params.sf_neg_size,1,1,1])
		SF_neg_character_matching_score = tf.reduce_sum(SF_NL_character_state_temp3*SF_KB_neg_character_attentioned_state,3)



#3.3计算声明的单词匹配分数（正样例）
		SF_NL_word_state_temp = tf.tile(tf.expand_dims(SF_NL_word_state,2),[1,1,params.sf_kb_word_max_length,1])
		SF_KB_word_state_temp = tf.tile(tf.expand_dims(SF_KB_word_state,1),[1,params.sf_nl_word_max_length,1，1])
		SF_concat_word_state = tf.concat(3,[SF_NL_word_state_temp,SF_KB_word_state_temp])

		SF_concat_word_state_temp = tf.reshape(SF_concat_word_state,[-1,params.word_hidden_size*4])
		SF_word_unnormalized_score = tf.reshape(tf.mul(tf.tanh(tf.mul(SF_concat_word_state_temp, SF_word_weight)+SF_word_bias), SF_word_pro),[params.sf_batch_size, params.sf_nl_word_max_length, params.sf_kb_word_max_length])

		SF_NL_word_mask_temp = tf.tile(tf.expand_dims(SF_NL_word_mask,2),[1,1,params.sf_kb_word_max_length])
		SF_KB_word_mask_temp = tf.tile(tf.expand_dims(SF_KB_word_mask,1),[1,params.sf_nl_word_max_length,1])
		SF_word_mask = SF_NL_word_mask_temp*SF_KB_word_mask_temp
		SF_word_unnormalized_score = SF_word_unnormalized_score*SF_word_mask

		SF_word_normalized_score = SF_word_unnormalized_score/tf.reduce_sum(SF_word_unnormalized_score, 2, keep_dims=True)

		SF_word_normalized_score_temp = tf.tile(tf.expand_dims(SF_word_normalized_score,3),[1, 1 ,1, params.word_hidden_size*2])

		SF_KB_word_attentioned_state = tf.reduce_sum(SF_word_normalized_score_temp*SF_KB_word_state_temp, 2)

		SF_word_matching_score = tf.reduce_sum(SF_NL_word_state*SF_KB_word_attentioned_state,2)
#3.4计算声明的单词匹配分数（负样例）
		SF_NL_word_state_temp2 = tf.tile(tf.expand_dims(SF_NL_word_state_temp,0),[params.sf_neg_size,1,1,1,1])
		SF_KB_neg_word_state_temp = tf.tile(tf.expand_dims(tf.tile(tf.expand_dims(SF_KB_neg_word_state,1),[1,params.sf_nl_word_max_length,1,1]),1),[1,params.sf_batch_size,1,1,1])
		SF_neg_concat_word_state = tf.concat(4,[SF_NL_word_state_temp2,SF_KB_neg_word_state_temp])

		SF_neg_concat_word_state_temp = tf.reshape(SF_neg_concat_word_state,[-1,params.word_hidden_size*4])
		SF_neg_word_unnormalized_score = tf.reshape(tf.mul(tf.tanh(tf.mul(SF_neg_concat_word_state_temp, SF_word_weight)+SF_word_bias), SF_word_pro),[params.sf_neg_size,params.sf_batch_size, params.sf_nl_word_max_length, params.sf_kb_word_max_length])

		SF_NL_word_mask_temp2 = tf.tile(tf.expand_dims(SF_NL_word_mask_temp,0),[params.sf_neg_size,1,1,1])
		SF_KB_neg_word_mask_temp = tf.tile(tf.expand_dims(tf.tile(tf.expand_dims(SF_KB_neg_word_mask,1),[1,params.sf_nl_word_max_length,1]),1),[1,params.sf_batch_size,1,1])
		SF_neg_word_mask = SF_NL_word_mask_temp2*SF_KB_neg_word_mask_temp
		SF_neg_word_unnormalized_score = SF_neg_word_unnormalized_score*SF_neg_word_mask

		SF_neg_word_normalized_score = SF_neg_word_unnormalized_score/tf.reduce_sum(SF_neg_word_unnormalized_score, 3, keep_dims=True)

		SF_neg_word_normalized_score_temp = tf.tile(tf.expand_dims(SF_neg_word_normalized_score,4),[1,1,1,1,params.word_hidden_size*2])

		SF_KB_neg_word_attentioned_state = tf.reduce_sum(SF_neg_word_normalized_score_temp*SF_KB_neg_word_state_temp, 3)

		SF_NL_word_state_temp3 = tf.tile(tf.expand_dims(SF_NL_word_state,0),[params.sf_neg_size,1,1,1])
		SF_neg_word_matching_score = tf.reduce_sum(SF_NL_word_state_temp3*SF_KB_neg_word_attentioned_state,3)



#3.5计算声明的词组匹配分数（正样例）
		SF_NL_phrase_state_temp = tf.tile(tf.expand_dims(SF_NL_phrase_state,2),[1,1,params.sf_kb_phrase_max_length,1])
		SF_KB_phrase_state_temp = tf.tile(tf.expand_dims(SF_KB_phrase_state,1),[1,params.sf_nl_phrase_max_length,1，1])
		SF_concat_phrase_state = tf.concat(3,[SF_NL_phrase_state_temp,SF_KB_phrase_state_temp])

		SF_concat_phrase_state_temp = tf.reshape(SF_concat_phrase_state,[-1,params.phrase_hidden_size*4])
		SF_phrase_unnormalized_score = tf.reshape(tf.mul(tf.tanh(tf.mul(SF_concat_phrase_state_temp, SF_phrase_weight)+SF_phrase_bias), SF_phrase_pro),[params.sf_batch_size, params.sf_nl_phrase_max_length, params.sf_kb_phrase_max_length])

		SF_NL_phrase_mask_temp = tf.tile(tf.expand_dims(SF_NL_phrase_mask,2),[1,1,params.sf_kb_phrase_max_length])
		SF_KB_phrase_mask_temp = tf.tile(tf.expand_dims(SF_KB_phrase_mask,1),[1,params.sf_nl_phrase_max_length,1])
		SF_phrase_mask = SF_NL_phrase_mask_temp*SF_KB_phrase_mask_temp
		SF_phrase_unnormalized_score = SF_phrase_unnormalized_score*SF_phrase_mask

		SF_phrase_normalized_score = SF_phrase_unnormalized_score/tf.reduce_sum(SF_phrase_unnormalized_score, 2, keep_dims=True)

		SF_phrase_normalized_score_temp = tf.tile(tf.expand_dims(SF_phrase_normalized_score,3),[1, 1 ,1, params.phrase_hidden_size*2])

		SF_KB_phrase_attentioned_state = tf.reduce_sum(SF_phrase_normalized_score_temp*SF_KB_phrase_state_temp, 2)

		SF_phrase_matching_score = tf.reduce_sum(SF_NL_phrase_state*SF_KB_phrase_attentioned_state,2)
#3.6计算声明的词组匹配分数（负样例）
		SF_NL_phrase_state_temp2 = tf.tile(tf.expand_dims(SF_NL_phrase_state_temp,0),[params.sf_neg_size,1,1,1,1])
		SF_KB_neg_phrase_state_temp = tf.tile(tf.expand_dims(tf.tile(tf.expand_dims(SF_KB_neg_phrase_state,1),[1,params.sf_nl_phrase_max_length,1,1]),1),[1,params.sf_batch_size,1,1,1])
		SF_neg_concat_phrase_state = tf.concat(4,[SF_NL_phrase_state_temp2,SF_KB_neg_phrase_state_temp])

		SF_neg_concat_phrase_state_temp = tf.reshape(SF_neg_concat_phrase_state,[-1,params.phrase_hidden_size*4])
		SF_neg_phrase_unnormalized_score = tf.reshape(tf.mul(tf.tanh(tf.mul(SF_neg_concat_phrase_state_temp, SF_phrase_weight)+SF_phrase_bias), SF_phrase_pro),[params.sf_neg_size,params.sf_batch_size, params.sf_nl_phrase_max_length, params.sf_kb_phrase_max_length])

		SF_NL_phrase_mask_temp2 = tf.tile(tf.expand_dims(SF_NL_phrase_mask_temp,0),[params.sf_neg_size,1,1,1])
		SF_KB_neg_phrase_mask_temp = tf.tile(tf.expand_dims(tf.tile(tf.expand_dims(SF_KB_neg_phrase_mask,1),[1,params.sf_nl_phrase_max_length,1]),1),[1,params.sf_batch_size,1,1])
		SF_neg_phrase_mask = SF_NL_phrase_mask_temp2*SF_KB_neg_phrase_mask_temp
		SF_neg_phrase_unnormalized_score = SF_neg_phrase_unnormalized_score*SF_neg_phrase_mask

		SF_neg_phrase_normalized_score = SF_neg_phrase_unnormalized_score/tf.reduce_sum(SF_neg_phrase_unnormalized_score, 3, keep_dims=True)

		SF_neg_phrase_normalized_score_temp = tf.tile(tf.expand_dims(SF_neg_phrase_normalized_score,4),[1,1,1,1,params.phrase_hidden_size*2])

		SF_KB_neg_phrase_attentioned_state = tf.reduce_sum(SF_neg_phrase_normalized_score_temp*SF_KB_neg_phrase_state_temp, 3)

		SF_NL_phrase_state_temp3 = tf.tile(tf.expand_dims(SF_NL_phrase_state,0),[params.sf_neg_size,1,1,1])
		SF_neg_phrase_matching_score = tf.reduce_sum(SF_NL_phrase_state_temp3*SF_KB_neg_phrase_attentioned_state,3)
'''


#3.7计算问句的字母匹配分数（正样例）
		QA_NL_character_state_temp = tf.tile(tf.expand_dims(QA_NL_character_state,2),[1,1,params.qa_kb_character_max_length,1])
		QA_KB_character_state_temp = tf.tile(tf.expand_dims(QA_KB_character_state,1),[1,params.qa_nl_character_max_length,1，1])
		QA_concat_character_state = tf.concat(3,[QA_NL_character_state_temp,QA_KB_character_state_temp])

		QA_concat_character_state_temp = tf.reshape(QA_concat_character_state,[-1,params.character_hidden_size*4])
		QA_character_unnormalized_score = tf.reshape(tf.mul(tf.tanh(tf.mul(QA_concat_character_state_temp, QA_character_weight)+QA_character_bias), QA_character_pro),[params.qa_batch_size, params.qa_nl_character_max_length, params.qa_kb_character_max_length])

		QA_NL_character_mask_temp = tf.tile(tf.expand_dims(QA_NL_character_mask,2),[1,1,params.qa_kb_character_max_length])
		QA_KB_character_mask_temp = tf.tile(tf.expand_dims(QA_KB_character_mask,1),[1,params.qa_nl_character_max_length,1])
		QA_character_mask = QA_NL_character_mask_temp*QA_KB_character_mask_temp
		QA_character_unnormalized_score = QA_character_unnormalized_score*QA_character_mask

		QA_character_normalized_score = QA_character_unnormalized_score/tf.reduce_sum(QA_character_unnormalized_score, 2, keep_dims=True)

		QA_character_normalized_score_temp = tf.tile(tf.expand_dims(QA_character_normalized_score,3),[1, 1 ,1, params.character_hidden_size*2])

		QA_KB_character_attentioned_state = tf.reduce_sum(QA_character_normalized_score_temp*QA_KB_character_state_temp, 2)

		QA_character_matching_score = tf.reduce_sum(QA_NL_character_state*QA_KB_character_attentioned_state,2)
#3.8计算问句的字母匹配分数（负样例）
		QA_NL_character_state_temp2 = tf.tile(tf.expand_dims(QA_NL_character_state_temp,0),[params.qa_neg_size,1,1,1,1])
		QA_KB_neg_character_state_temp = tf.tile(tf.expand_dims(tf.tile(tf.expand_dims(QA_KB_neg_character_state,1),[1,params.qa_nl_character_max_length,1,1]),1),[1,params.qa_batch_size,1,1,1])
		QA_neg_concat_character_state = tf.concat(4,[QA_NL_character_state_temp2,QA_KB_neg_character_state_temp])

		QA_neg_concat_character_state_temp = tf.reshape(QA_neg_concat_character_state,[-1,params.character_hidden_size*4])
		QA_neg_character_unnormalized_score = tf.reshape(tf.mul(tf.tanh(tf.mul(QA_neg_concat_character_state_temp, QA_character_weight)+QA_character_bias), QA_character_pro),[params.qa_neg_size,params.qa_batch_size, params.qa_nl_character_max_length, params.qa_kb_character_max_length])

		QA_NL_character_mask_temp2 = tf.tile(tf.expand_dims(QA_NL_character_mask_temp,0),[params.qa_neg_size,1,1,1])
		QA_KB_neg_character_mask_temp = tf.tile(tf.expand_dims(tf.tile(tf.expand_dims(QA_KB_neg_character_mask,1),[1,params.qa_nl_character_max_length,1]),1),[1,params.qa_batch_size,1,1])
		QA_neg_character_mask = QA_NL_character_mask_temp2*QA_KB_neg_character_mask_temp
		QA_neg_character_unnormalized_score = QA_neg_character_unnormalized_score*QA_neg_character_mask

		QA_neg_character_normalized_score = QA_neg_character_unnormalized_score/tf.reduce_sum(QA_neg_character_unnormalized_score, 3, keep_dims=True)

		QA_neg_character_normalized_score_temp = tf.tile(tf.expand_dims(QA_neg_character_normalized_score,4),[1,1,1,1,params.character_hidden_size*2])

		QA_KB_neg_character_attentioned_state = tf.reduce_sum(QA_neg_character_normalized_score_temp*QA_KB_neg_character_state_temp, 3)

		QA_NL_character_state_temp3 = tf.tile(tf.expand_dims(QA_NL_character_state,0),[params.qa_neg_size,1,1,1])
		QA_neg_character_matching_score = tf.reduce_sum(QA_NL_character_state_temp3*QA_KB_neg_character_attentioned_state,3)



#3.9计算问句的单词匹配分数（正样例）
		QA_NL_word_state_temp = tf.tile(tf.expand_dims(QA_NL_word_state,2),[1,1,params.qa_kb_word_max_length,1])
		QA_KB_word_state_temp = tf.tile(tf.expand_dims(QA_KB_word_state,1),[1,params.qa_nl_word_max_length,1，1])
		QA_concat_word_state = tf.concat(3,[QA_NL_word_state_temp,QA_KB_word_state_temp])

		QA_concat_word_state_temp = tf.reshape(QA_concat_word_state,[-1,params.word_hidden_size*4])
		QA_word_unnormalized_score = tf.reshape(tf.mul(tf.tanh(tf.mul(QA_concat_word_state_temp, QA_word_weight)+QA_word_bias), QA_word_pro),[params.qa_batch_size, params.qa_nl_word_max_length, params.qa_kb_word_max_length])

		QA_NL_word_mask_temp = tf.tile(tf.expand_dims(QA_NL_word_mask,2),[1,1,params.qa_kb_word_max_length])
		QA_KB_word_mask_temp = tf.tile(tf.expand_dims(QA_KB_word_mask,1),[1,params.qa_nl_word_max_length,1])
		QA_word_mask = QA_NL_word_mask_temp*QA_KB_word_mask_temp
		QA_word_unnormalized_score = QA_word_unnormalized_score*QA_word_mask

		QA_word_normalized_score = QA_word_unnormalized_score/tf.reduce_sum(QA_word_unnormalized_score, 2, keep_dims=True)

		QA_word_normalized_score_temp = tf.tile(tf.expand_dims(QA_word_normalized_score,3),[1, 1 ,1, params.word_hidden_size*2])

		QA_KB_word_attentioned_state = tf.reduce_sum(QA_word_normalized_score_temp*QA_KB_word_state_temp, 2)

		QA_word_matching_score = tf.reduce_sum(QA_NL_word_state*QA_KB_word_attentioned_state,2)
#3.10计算问句的单词匹配分数（负样例）
		QA_NL_word_state_temp2 = tf.tile(tf.expand_dims(QA_NL_word_state_temp,0),[params.qa_neg_size,1,1,1,1])
		QA_KB_neg_word_state_temp = tf.tile(tf.expand_dims(tf.tile(tf.expand_dims(QA_KB_neg_word_state,1),[1,params.qa_nl_word_max_length,1,1]),1),[1,params.qa_batch_size,1,1,1])
		QA_neg_concat_word_state = tf.concat(4,[QA_NL_word_state_temp2,QA_KB_neg_word_state_temp])

		QA_neg_concat_word_state_temp = tf.reshape(QA_neg_concat_word_state,[-1,params.word_hidden_size*4])
		QA_neg_word_unnormalized_score = tf.reshape(tf.mul(tf.tanh(tf.mul(QA_neg_concat_word_state_temp, QA_word_weight)+QA_word_bias), QA_word_pro),[params.qa_neg_size,params.qa_batch_size, params.qa_nl_word_max_length, params.qa_kb_word_max_length])

		QA_NL_word_mask_temp2 = tf.tile(tf.expand_dims(QA_NL_word_mask_temp,0),[params.qa_neg_size,1,1,1])
		QA_KB_neg_word_mask_temp = tf.tile(tf.expand_dims(tf.tile(tf.expand_dims(QA_KB_neg_word_mask,1),[1,params.qa_nl_word_max_length,1]),1),[1,params.qa_batch_size,1,1])
		QA_neg_word_mask = QA_NL_word_mask_temp2*QA_KB_neg_word_mask_temp
		QA_neg_word_unnormalized_score = QA_neg_word_unnormalized_score*QA_neg_word_mask

		QA_neg_word_normalized_score = QA_neg_word_unnormalized_score/tf.reduce_sum(QA_neg_word_unnormalized_score, 3, keep_dims=True)

		QA_neg_word_normalized_score_temp = tf.tile(tf.expand_dims(QA_neg_word_normalized_score,4),[1,1,1,1,params.word_hidden_size*2])

		QA_KB_neg_word_attentioned_state = tf.reduce_sum(QA_neg_word_normalized_score_temp*QA_KB_neg_word_state_temp, 3)

		QA_NL_word_state_temp3 = tf.tile(tf.expand_dims(QA_NL_word_state,0),[params.qa_neg_size,1,1,1])
		QA_neg_word_matching_score = tf.reduce_sum(QA_NL_word_state_temp3*QA_KB_neg_word_attentioned_state,3)



#3.11计算问句的词组匹配分数（正样例）
		QA_NL_phrase_state_temp = tf.tile(tf.expand_dims(QA_NL_phrase_state,2),[1,1,params.qa_kb_phrase_max_length,1])
		QA_KB_phrase_state_temp = tf.tile(tf.expand_dims(QA_KB_phrase_state,1),[1,params.qa_nl_phrase_max_length,1，1])
		QA_concat_phrase_state = tf.concat(3,[QA_NL_phrase_state_temp,QA_KB_phrase_state_temp])

		QA_concat_phrase_state_temp = tf.reshape(QA_concat_phrase_state,[-1,params.phrase_hidden_size*4])
		QA_phrase_unnormalized_score = tf.reshape(tf.mul(tf.tanh(tf.mul(QA_concat_phrase_state_temp, QA_phrase_weight)+QA_phrase_bias), QA_phrase_pro),[params.qa_batch_size, params.qa_nl_phrase_max_length, params.qa_kb_phrase_max_length])

		QA_NL_phrase_mask_temp = tf.tile(tf.expand_dims(QA_NL_phrase_mask,2),[1,1,params.qa_kb_phrase_max_length])
		QA_KB_phrase_mask_temp = tf.tile(tf.expand_dims(QA_KB_phrase_mask,1),[1,params.qa_nl_phrase_max_length,1])
		QA_phrase_mask = QA_NL_phrase_mask_temp*QA_KB_phrase_mask_temp
		QA_phrase_unnormalized_score = QA_phrase_unnormalized_score*QA_phrase_mask

		QA_phrase_normalized_score = QA_phrase_unnormalized_score/tf.reduce_sum(QA_phrase_unnormalized_score, 2, keep_dims=True)

		QA_phrase_normalized_score_temp = tf.tile(tf.expand_dims(QA_phrase_normalized_score,3),[1, 1 ,1, params.phrase_hidden_size*2])

		QA_KB_phrase_attentioned_state = tf.reduce_sum(QA_phrase_normalized_score_temp*QA_KB_phrase_state_temp, 2)

		QA_phrase_matching_score = tf.reduce_sum(QA_NL_phrase_state*QA_KB_phrase_attentioned_state,2)
#3.12计算问句的词组匹配分数（负样例）
		QA_NL_phrase_state_temp2 = tf.tile(tf.expand_dims(QA_NL_phrase_state_temp,0),[params.qa_neg_size,1,1,1,1])
		QA_KB_neg_phrase_state_temp = tf.tile(tf.expand_dims(tf.tile(tf.expand_dims(QA_KB_neg_phrase_state,1),[1,params.qa_nl_phrase_max_length,1,1]),1),[1,params.qa_batch_size,1,1,1])
		QA_neg_concat_phrase_state = tf.concat(4,[QA_NL_phrase_state_temp2,QA_KB_neg_phrase_state_temp])

		QA_neg_concat_phrase_state_temp = tf.reshape(QA_neg_concat_phrase_state,[-1,params.phrase_hidden_size*4])
		QA_neg_phrase_unnormalized_score = tf.reshape(tf.mul(tf.tanh(tf.mul(QA_neg_concat_phrase_state_temp, QA_phrase_weight)+QA_phrase_bias), QA_phrase_pro),[params.qa_neg_size,params.qa_batch_size, params.qa_nl_phrase_max_length, params.qa_kb_phrase_max_length])

		QA_NL_phrase_mask_temp2 = tf.tile(tf.expand_dims(QA_NL_phrase_mask_temp,0),[params.qa_neg_size,1,1,1])
		QA_KB_neg_phrase_mask_temp = tf.tile(tf.expand_dims(tf.tile(tf.expand_dims(QA_KB_neg_phrase_mask,1),[1,params.qa_nl_phrase_max_length,1]),1),[1,params.qa_batch_size,1,1])
		QA_neg_phrase_mask = QA_NL_phrase_mask_temp2*QA_KB_neg_phrase_mask_temp
		QA_neg_phrase_unnormalized_score = QA_neg_phrase_unnormalized_score*QA_neg_phrase_mask

		QA_neg_phrase_normalized_score = QA_neg_phrase_unnormalized_score/tf.reduce_sum(QA_neg_phrase_unnormalized_score, 3, keep_dims=True)

		QA_neg_phrase_normalized_score_temp = tf.tile(tf.expand_dims(QA_neg_phrase_normalized_score,4),[1,1,1,1,params.phrase_hidden_size*2])

		QA_KB_neg_phrase_attentioned_state = tf.reduce_sum(QA_neg_phrase_normalized_score_temp*QA_KB_neg_phrase_state_temp, 3)

		QA_NL_phrase_state_temp3 = tf.tile(tf.expand_dims(QA_NL_phrase_state,0),[params.qa_neg_size,1,1,1])
		QA_neg_phrase_matching_score = tf.reduce_sum(QA_NL_phrase_state_temp3*QA_KB_neg_phrase_attentioned_state,3)
'''
###4.计算声明或问句的正负样例最终得分（实施hierarchical summarization）
#4.1计算声明的最终得分（正样例）
#再进行一遍掩码，确保正确
		SF_character_matching_score = SF_character_matching_score*SF_NL_character_mask
		SF_word_matching_score = SF_word_matching_score*SF_NL_word_mask
		SF_phrase_matching_score = SF_phrase_matching_score*SF_NL_phrase_mask

		SF_character_matching_score_temp = tf.tile(tf.expand_dims(SF_character_matching_score,1),[1,params.sf_nl_word_max_length,1])
		SF_word_matching_score_from_lower = tf.reduce_sum(SF_character_matching_score_temp*SF_c2w_tree,2)/tf.reduce_sum(SF_c2w_tree,2)
		SF_word_matching_score = tf.maximum(SF_word_matching_score,SF_word_matching_score_from_lower)

		SF_word_matching_score_temp = tf.tile(tf.expand_dims(SF_word_matching_score,1),[1,params.sf_nl_phrase_max_length,1])
		SF_phrase_matching_score_from_lower = tf.reduce_sum(SF_word_matching_score_temp*SF_w2p_tree,2)/tf.reduce_sum(SF_w2p_tree,2)
		SF_phrase_matching_score = tf.maximum(SF_phrase_matching_score,SF_phrase_matching_score_from_lower)
		SF_final_score = tf.reduce_sum(SF_phrase_matching_score,1)/tf.reduce_sum(SF_NL_phrase_mask,1)
#4.2计算声明的最终得分（负样例）
		SF_neg_character_matching_score = SF_neg_character_matching_score*tf.tile(tf.expand_dims(SF_NL_character_mask,0),[params.sf_neg_size,1,1])
		SF_neg_word_matching_score = SF_neg_word_matching_score*tf.tile(tf.expand_dims(SF_NL_word_mask,0),[params.sf_neg_size,1,1])
		SF_NL_phrase_mask_temp = tf.tile(tf.expand_dims(SF_NL_phrase_mask,0),[params.sf_neg_size,1,1])
		SF_neg_phrase_matching_score = SF_neg_phrase_matching_score*SF_NL_phrase_mask_temp

		SF_neg_character_matching_score_temp = tf.tile(tf.expand_dims(SF_neg_character_matching_score,2),[1,1,params.sf_nl_word_max_length,1])
		SF_c2w_tree_temp = tf.tile(tf.expand_dims(SF_c2w_tree_temp,0),[params.sf_neg_size,1,1,1])
		SF_neg_word_matching_score_from_lower = tf.reduce_sum(SF_neg_character_matching_score_temp*SF_c2w_tree_temp,3)/tf.reduce_sum(SF_c2w_tree_temp,3)
		SF_neg_word_matching_score = tf.maximum(SF_neg_word_matching_score,SF_neg_word_matching_score_from_lower)

		SF_neg_word_matching_score_temp = tf.tile(tf.expand_dims(SF_neg_word_matching_score,2),[1,1,params.sf_nl_phrase_max_length,1])
		SF_w2p_tree_temp = tf.tile(tf.expand_dims(SF_w2p_tree_temp,0),[params.sf_neg_size,1,1,1])
		SF_neg_phrase_matching_score_from_lower = tf.reduce_sum(SF_neg_word_matching_score_temp*SF_w2p_tree_temp,3)/tf.reduce_sum(SF_w2p_tree_temp,3)
		SF_neg_phrase_matching_score = tf.maximum(SF_neg_phrase_matching_score_temp,SF_neg_phrase_matching_score_from_lower)
		SF_neg_final_score = tf.reduce_sum(SF_neg_phrase_matching_score,2)/tf.reduce_sum(SF_NL_phrase_mask_temp,2)
'''


#4.3计算问句的最终得分（正样例）
		QA_character_matching_score = QA_character_matching_score*QA_NL_character_mask
		QA_word_matching_score = QA_word_matching_score*QA_NL_word_mask
		QA_phrase_matching_score = QA_phrase_matching_score*QA_NL_phrase_mask

		QA_character_matching_score_temp = tf.tile(tf.expand_dims(QA_character_matching_score,1),[1,params.qa_nl_word_max_length,1])
		QA_word_matching_score_from_lower = tf.reduce_sum(QA_character_matching_score_temp*QA_c2w_tree,2)/tf.reduce_sum(QA_c2w_tree,2)
		QA_word_matching_score = tf.maximum(QA_word_matching_score,QA_word_matching_score_from_lower)

		QA_word_matching_score_temp = tf.tile(tf.expand_dims(QA_word_matching_score,1),[1,params.qa_nl_phrase_max_length,1])
		QA_phrase_matching_score_from_lower = tf.reduce_sum(QA_word_matching_score_temp*QA_w2p_tree,2)/tf.reduce_sum(QA_w2p_tree,2)
		QA_phrase_matching_score = tf.maximum(QA_phrase_matching_score,QA_phrase_matching_score_from_lower)
		QA_final_score = tf.reduce_sum(QA_phrase_matching_score,1)/tf.reduce_sum(QA_NL_phrase_mask,1)
#4.4计算问句的最终得分（负样例）
		QA_neg_character_matching_score = QA_neg_character_matching_score*tf.tile(tf.expand_dims(QA_NL_character_mask,0),[params.qa_neg_size,1,1])
		QA_neg_word_matching_score = QA_neg_word_matching_score*tf.tile(tf.expand_dims(QA_NL_word_mask,0),[params.qa_neg_size,1,1])
		QA_NL_phrase_mask_temp = tf.tile(tf.expand_dims(QA_NL_phrase_mask,0),[params.qa_neg_size,1,1])
		QA_neg_phrase_matching_score = QA_neg_phrase_matching_score*QA_NL_phrase_mask_temp

		QA_neg_character_matching_score_temp = tf.tile(tf.expand_dims(QA_neg_character_matching_score,2),[1,1,params.qa_nl_word_max_length,1])
		QA_c2w_tree_temp = tf.tile(tf.expand_dims(QA_c2w_tree_temp,0),[params.qa_neg_size,1,1,1])
		QA_neg_word_matching_score_from_lower = tf.reduce_sum(QA_neg_character_matching_score_temp*QA_c2w_tree_temp,3)/tf.reduce_sum(QA_c2w_tree_temp,3)
		QA_neg_word_matching_score = tf.maximum(QA_neg_word_matching_score,QA_neg_word_matching_score_from_lower)

		QA_neg_word_matching_score_temp = tf.tile(tf.expand_dims(QA_neg_word_matching_score,2),[1,1,params.qa_nl_phrase_max_length,1])
		QA_w2p_tree_temp = tf.tile(tf.expand_dims(QA_w2p_tree_temp,0),[params.qa_neg_size,1,1,1])
		QA_neg_phrase_matching_score_from_lower = tf.reduce_sum(QA_neg_word_matching_score_temp*QA_w2p_tree_temp,3)/tf.reduce_sum(QA_w2p_tree_temp,3)
		QA_neg_phrase_matching_score = tf.maximum(QA_neg_phrase_matching_score_temp,QA_neg_phrase_matching_score_from_lower)
		QA_neg_final_score = tf.reduce_sum(QA_neg_phrase_matching_score,2)/tf.reduce_sum(QA_NL_phrase_mask_temp,2)

###5.计算损失
		SF_loss = tf.reduce_sum(tf.nn.relu(tf.(params.gamma_sf - SF_final_score)+SF_neg_final_score))
		QA_loss = tf.reduce_sum(tf.nn.relu(tf.(params.gamma_qa - QA_final_score)+QA_neg_final_score))
		loss = SF_loss + QA_loss

###6.定义optimizer
	with tf.Session(graph=graph) as sess:
		tf.initialize_all_variables().run()
		for e in range(params.epoch_num):
			loader.reset_batch_pointer()
			for b in range(loader.qa_batch_num):
				now_batch = loader.next_batch()
				'''
				feed = {SF_NL_character_train_inputs: now_batch[0],
						SF_NL_word_train_inputs: now_batch[1],
						SF_NL_phrase_train_inputs: now_batch[2],
						SF_KB_character_train_inputs: now_batch[3],
						SF_KB_word_train_inputs: now_batch[4],
						SF_KB_phrase_train_inputs: now_batch[5],
						SF_KB_neg_character_train_inputs: now_batch[6],
						SF_KB_neg_word_train_inputs: now_batch[7],
						SF_KB_neg_phrase_train_inputs: now_batch[8],
						QA_NL_character_train_inputs: now_batch[9],
						QA_NL_word_train_inputs: now_batch[10],
						QA_NL_phrase_train_inputs: now_batch[11],
						QA_KB_character_train_inputs: now_batch[12],
						QA_KB_word_train_inputs: now_batch[13],
						QA_KB_phrase_train_inputs: now_batch[14],
						QA_KB_neg_character_train_inputs: now_batch[15],
						QA_KB_neg_word_train_inputs: now_batch[16],
						QA_KB_neg_phrase_train_inputs: now_batch[17],
						SF_NL_oov_word_train_inputs: now_batch[18],
						SF_NL_oov_phrase_train_inputs: now_batch[19],
						SF_KB_oov_word_train_inputs: now_batch[20],
						SF_KB_oov_phrase_train_inputs: now_batch[21],
						SF_KB_neg_oov_word_train_inputs: now_batch[22],
						SF_KB_neg_oov_phrase_train_inputs: now_batch[23],
						QA_NL_oov_word_train_inputs: now_batch[24],
						QA_NL_oov_phrase_train_inputs: now_batch[25],
						QA_KB_oov_word_train_inputs: now_batch[26],
						QA_KB_oov_phrase_train_inputs: now_batch[27],
						QA_KB_neg_oov_word_train_inputs: now_batch[28],
						QA_KB_neg_oov_phrase_train_inputs: now_batch[29],
						NL_character_of_oov_word: now_batch[30],
						NL_word_of_oov_phrase: now_batch[31],
						NL_oov_word_of_oov_phrase: now_batch[32],
						KB_character_of_oov_word: now_batch[33],
						KB_word_of_oov_phrase: now_batch[34],
						KB_oov_word_of_oov_phrase: now_batch[35],
						SF_c2w_tree: now_batch[36],
						SF_w2p_tree: now_batch[37],
						QA_c2w_tree: now_batch[38],
						QA_w2p_tree: now_batch[39],
						SF_NL_character_mask: now_batch[40],
						SF_NL_word_mask: now_batch[41],
						SF_NL_phrase_mask: now_batch[42],
						SF_KB_character_mask: now_batch[43],
						SF_KB_word_mask: now_batch[44],
						SF_KB_phrase_mask: now_batch[45],
						SF_KB_neg_character_mask: now_batch[46],
						SF_KB_neg_word_mask: now_batch[47],
						SF_KB_neg_phrase_mask: now_batch[48],
						QA_NL_character_mask: now_batch[49],
						QA_NL_word_mask: now_batch[50],
						QA_NL_phrase_mask: now_batch[51],
						QA_KB_character_mask: now_batch[52],
						QA_KB_word_mask: now_batch[53],
						QA_KB_phrase_mask: now_batch[54],
						QA_KB_neg_character_mask: now_batch[55],
						QA_KB_neg_word_mask: now_batch[56],
						QA_KB_neg_phrase_mask: now_batch[57]}
				'''
				feed = {
						QA_NL_character_train_inputs: now_batch[9],
						QA_NL_word_train_inputs: now_batch[10],
						QA_NL_phrase_train_inputs: now_batch[11],
						QA_KB_character_train_inputs: now_batch[12],
						QA_KB_word_train_inputs: now_batch[13],
						QA_KB_phrase_train_inputs: now_batch[14],
						QA_KB_neg_character_train_inputs: now_batch[15],
						QA_KB_neg_word_train_inputs: now_batch[16],
						QA_KB_neg_phrase_train_inputs: now_batch[17],
						QA_NL_oov_word_train_inputs: now_batch[24],
						QA_NL_oov_phrase_train_inputs: now_batch[25],
						QA_KB_oov_word_train_inputs: now_batch[26],
						QA_KB_oov_phrase_train_inputs: now_batch[27],
						QA_KB_neg_oov_word_train_inputs: now_batch[28],
						QA_KB_neg_oov_phrase_train_inputs: now_batch[29],
						NL_character_of_oov_word: now_batch[30],
						NL_word_of_oov_phrase: now_batch[31],
						NL_oov_word_of_oov_phrase: now_batch[32],
						KB_character_of_oov_word: now_batch[33],
						KB_word_of_oov_phrase: now_batch[34],
						KB_oov_word_of_oov_phrase: now_batch[35],
						QA_c2w_tree: now_batch[38],
						QA_w2p_tree: now_batch[39],
						QA_NL_character_mask: now_batch[49],
						QA_NL_word_mask: now_batch[50],
						QA_NL_phrase_mask: now_batch[51],
						QA_KB_character_mask: now_batch[52],
						QA_KB_word_mask: now_batch[53],
						QA_KB_phrase_mask: now_batch[54],
						QA_KB_neg_character_mask: now_batch[55],
						QA_KB_neg_word_mask: now_batch[56],
						QA_KB_neg_phrase_mask: now_batch[57]}
				'''
				loss_val, _ = sess.run([loss, optimizer], feed)
				loss_all += loss_val
				print 'loss %d batch:' % b, loss_val
				print '%d round' % e
				'''
				sess.run([NL_oov_word_embedding], feed)

if __name__ == '__main__':
	main()
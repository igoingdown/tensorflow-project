#coding=utf-8
import numpy as np
import math
import random
import zmx_params as params
import copy
import cPickle
import time


class Textloader():
	def __init__(self):
		self.load_from_file()
		self.nl_character_oov_begin = 39
		self.kb_character_oov_begin = 65
		self.nl_word_oov_begin = self.detect_oov_begin(self.nl_word_dict_freq, params.freq_limit)
		# print self.nl_word_oov_begin

		self.nl_phrase_oov_begin = self.detect_oov_begin(self.nl_phrase_dict_freq, params.freq_limit)
		self.kb_word_oov_begin = self.detect_oov_begin(self.kb_word_dict_freq, params.freq_limit)
		self.kb_phrase_oov_begin = self.detect_oov_begin(self.kb_phrase_dict_freq, params.freq_limit)
		self.create_batches()

	def detect_oov_begin(self, idict, freq_limit):
		length = len(idict)
		index_begin = 3
		for index in range(index_begin, length + index_begin):
			if idict[index] < freq_limit:
				return index
		return length + index_begin

	def load_from_file(self):
		with open(params.train_questions_pickle_path, 'rb') as f:
			self.train_questions = cPickle.load(f)
		with open(params.train_answers_pickle_path, 'rb') as f:
			self.train_answers = cPickle.load(f)
		with open(params.train_triples_pickle_path, 'rb') as f:
			self.train_triples = cPickle.load(f)
		with open(params.nl_word_dict_freq_pickle_path, 'rb') as f:
			self.nl_word_dict_freq = cPickle.load(f)
		with open(params.nl_phrase_dict_freq_pickle_path, 'rb') as f:
			self.nl_phrase_dict_freq = cPickle.load(f)
		with open(params.kb_word_dict_freq_pickle_path, 'rb') as f:
			self.kb_word_dict_freq = cPickle.load(f)
		with open(params.kb_phrase_dict_freq_pickle_path, 'rb') as f:
			self.kb_phrase_dict_freq = cPickle.load(f)
		with open(params.nl_character2word_dict_pickle_path, 'rb') as f:
			self.nl_character2word_dict = cPickle.load(f)
		with open(params.nl_word2phrase_dict_pickle_path, 'rb') as f:
			self.nl_word2phrase_dict = cPickle.load(f)
		with open(params.kb_character2word_dict_pickle_path, 'rb') as f:
			self.kb_character2word_dict = cPickle.load(f)
		with open(params.kb_word2phrase_dict_pickle_path, 'rb') as f:
			self.kb_word2phrase_dict = cPickle.load(f)

		"""
		# this is old old code

		file = open(params.pickle_path, 'rb')
		self.train_questions = cPickle.load(file)
		self.train_answers = cPickle.load(file)
		self.train_triples = cPickle.load(file)
		self.nl_word_dict_freq = cPickle.load(file)
		self.nl_phrase_dict_freq = cPickle.load(file)
		self.kb_word_dict_freq = cPickle.load(file)
		self.kb_phrase_dict_freq = cPickle.load(file)
		self.nl_character2word_dict = cPickle.load(file)
		self.nl_word2phrase_dict = cPickle.load(file)
		self.kb_character2word_dict = cPickle.load(file)
		self.kb_word2phrase_dict = cPickle.load(file)
		"""

	def truncate_and_transform_seq(self, sequence, character2word, word2phrase, characterseq_max_length, wordseq_max_length, phraseseq_max_length, isnl):
		phrases = copy.deepcopy(sequence)
		# print sequence
		if isnl:
			phrases.insert(0, 1)
			phrases.append(2)
			words = []
		else:
			words = [1]
		characters = []
		if len(phrases) > phraseseq_max_length:
			phrases = phrases[:phraseseq_max_length]
		index = -1
		for x in range(len(phrases)):
			phrase = phrases[x]
			if phrase == 0 or phrase == 1 or phrase == 2:
				words_exd = [phrase]
				characters_exd = []
			else:
				words_exd = copy.deepcopy(word2phrase[phrase])
				characters_exd = []
				for word in words_exd:
					temp = [1]
					temp.extend(copy.deepcopy(character2word[word]))
					temp.append(2)
					characters_exd.extend(temp)
				if not isnl and x == len(phrases) - 1:
					words_exd.append(2)
			if len(words)+ len(words_exd) > wordseq_max_length or len(characters) + len(characters_exd) > characterseq_max_length:
				# print len(words), "\t", len(words_exd),  "\t", len(characters),  "\t", len(characters_exd)
				index = x
				break
			else:
				words.extend(words_exd)
				characters.extend(characters_exd)
				# print index;
				# print phrases
		if index != -1:
			phrases = phrases[:index]
		return characters, words, phrases

	def reset_batch_pointer(self):
		self.pointer = 0

	def create_batches(self):
		'''
		self.sf_batch_num = int(len(self.train_statements) / params.sf_batch_size)
		statement_inputs = np.array(self.train_statements[:params.sf_batch_size * params.sf_batch_num])
		fact_inputs = np.array(self.train_facts[:params.sf_batch_size * params.sf_batch_num])
		'''
		self.qa_batch_num = int(len(self.train_questions) / params.qa_batch_size)
		# print "total num of batch is {0}".format(self.qa_batch_num)
		# time.sleep(5)
		question_inputs = np.array(self.train_questions[:params.qa_batch_size * self.qa_batch_num])
		answer_inputs = np.array(self.train_answers[:params.qa_batch_size * self.qa_batch_num])
		'''
		self.statement_input_batches = np.split(statement_inputs, params.sf_batch_num, 0)
		self.fact_input_batches = np.split(fact_inputs, params.sf_batch_num, 0)
		'''
		self.question_input_batches = np.split(question_inputs, self.qa_batch_num, 0)
		self.answer_input_batches = np.split(answer_inputs, self.qa_batch_num, 0)
		self.pointer = 0

	def handle_batch(self, batch, character2word, word2phrase, \
					characterseq_max_length, wordseq_max_length, phraseseq_max_length, \
					word_oov_begin, phrase_oov_begin, oov_word_index, oov_phrase_index, isnl):
		character_train_inputs = list()
		word_train_inputs = list()
		phrase_train_inputs = list()
		oov_word_train_inputs = list()
		oov_phrase_train_inputs = list()
		character_mask = list()
		word_mask = list()
		phrase_mask = list()
		c2w_tree = list()
		w2p_tree = list()

		"""
		# 得到10088（第一个oov_word）对应的characters，用来填充一些batch中 NL_character_of_oov_word为空的情况
		# 上述情况会导致使用GPU运行程序时崩溃， 这种补充只是暂时的……
		if isnl:
			print character2word[10088]
		"""
		for sequence in batch:
			# 每个batch中有batch_size个问题（对应NL）或者答案（对应KB）对应的词组序列。
			characters, words, phrases = self.truncate_and_transform_seq(sequence, character2word, word2phrase, \
			                characterseq_max_length, wordseq_max_length, phraseseq_max_length, isnl)
			#print phrases
			oov_words = list()
			oov_phrases = list()

			# 要恢复原样，注释掉就行
			# 这就是我要改的初始化部分,oov_words和oov_phrases都要初始化，两个是同样的处理，都有同样的问题
			# 这是我添加的部分
			for _ in range(len(phrases)):
				oov_phrases.append(0)
			for _ in range(len(words)):
				oov_words.append(0)

			c2w_slice = list()
			w2p_slice = list()
			c2w_index = 0
			w2p_index = 0
			#可能要改，c2w_tree和w2p_tree在序列位置上的映射关系，是否加2
			for x in range(len(words)):
				word = words[x]
				c2w_line = [0]*characterseq_max_length
				if word == 0 or word == 1 or word == 2:
					# 以下的代码会导致oov_word和非oov_word的长度不一致
					# oov_words.append(0)
					# 这是我改过的代码
					# 由于oov_words已经初始化为全0，这里是多余的，注释掉。
					#oov_words[x] = 0

					c2w_slice.append(c2w_line)

				else:
					for z in range(c2w_index,c2w_index+len(character2word[word])+2):
						c2w_line[z] = 1
					c2w_index = c2w_index + len(character2word[word])+2
					c2w_slice.append(c2w_line)
					if word >= word_oov_begin:
						if not oov_word_index.has_key(word):
							# 下面的代码oov_words从0开始，而在查找embedding的时候把0视为无意义，造成矛盾
							#oov_word_index[word] = len(oov_word_index)
							oov_word_index[word] = len(oov_word_index) + 1;
						else:
							pass
						words[x] = 0

						# 以下是原来的代码，有问题，注释掉
						# oov_words.append(oov_word_index[word])

						# 以下是我添加的代码
						oov_words[x] = oov_word_index[word]

					"""
					# 这是多余的，因为在初始化中已经置零
					else:
						oov_words.append(0)
					"""
					#print "raw phrases:", phrases
			for y in range(len(phrases)):
				phrase = phrases[y]
				w2p_line = [0]*wordseq_max_length
				if phrase == 0 or phrase == 1 or phrase == 2:
					for z in range(w2p_index, w2p_index+1):
						w2p_line[z] = 1
					w2p_index = w2p_index + 1
					w2p_slice.append(w2p_line)
				else:
					for z in range(w2p_index, w2p_index+len(word2phrase[phrase])):
						w2p_line[z] = 1
					w2p_index = w2p_index + len(word2phrase[phrase])
					w2p_slice.append(w2p_line)
					if phrase >= phrase_oov_begin:
						if not oov_phrase_index.has_key(phrase):
							#oov_phrase_index[phrase] = len(oov_phrase_index)
							# 这是原来的代码，oov_embedding的索引从0开始，而0在后来使用的时候是无意义的。
							oov_phrase_index[phrase] = len(oov_phrase_index) + 1

						else:
							pass
						phrases[y] = 0
						#print 'boom!!!\n'*10

						# oov_phrases.append(oov_phrase_index[phrase])
						# 这里错了吧，oov_phrase应该和非oov_phrase保持一致，在phrase补0的位上补上非0值（phrase在oov字典中的位置）
						# 因为后面embedding相加就是用的这个原理。
						# 可以按照以下这种方式改：
						# 先将oov_phrase初始化为和phrase一样长，元素全部置0， 然后如果有oov出现，就把相应位置为phrase在oov_phrase字典中的对应的value。
						oov_phrases[y] = oov_phrase_index[phrase]

					"""
					#这是多余的，在初始化中已经全部置零
					else:
						oov_phrases.append(0)
					"""

			character_mask_line = [1]*len(characters)
			word_mask_line = [1]*len(words)
			phrase_mask_line = [1]*len(phrases)
                        # print phrases, '\t'*2, len(phrases)
			while(len(characters) < characterseq_max_length):
				characters.append(0)
				character_mask_line.append(0)
			while(len(words) < wordseq_max_length):
				words.append(0)
				word_mask_line.append(0)
			while(len(oov_words) < wordseq_max_length):
				oov_words.append(0)

			# 忽略由处理phrase出错导致的kb_mask不全为1的情况，这里将kb的mask全部置为1
			if not isnl:
				phrase_mask_line = [1] * phraseseq_max_length

			while(len(phrases) < phraseseq_max_length):
				phrases.append(0)
                        while(len(phrase_mask_line) < phraseseq_max_length):
				phrase_mask_line.append(0)
			while(len(oov_phrases) < phraseseq_max_length):
				oov_phrases.append(0)
			while(len(c2w_slice) < wordseq_max_length):
				c2w_slice.append([0]*characterseq_max_length)
			while(len(w2p_slice) < phraseseq_max_length):
				w2p_slice.append([0]*wordseq_max_length)
			character_train_inputs.append(characters)
			word_train_inputs.append(words)
			phrase_train_inputs.append(phrases)
			#print phrases
			#print oov_phrase_index, '\n'*2
			oov_word_train_inputs.append(oov_words)
			oov_phrase_train_inputs.append(oov_phrases)
			c2w_tree.append(c2w_slice)
			w2p_tree.append(w2p_slice)
			character_mask.append(character_mask_line)
			word_mask.append(word_mask_line)
			phrase_mask.append(phrase_mask_line)
		if isnl:
			return (character_train_inputs, word_train_inputs, phrase_train_inputs, \
					oov_word_train_inputs, oov_phrase_train_inputs, c2w_tree, w2p_tree, character_mask, word_mask, phrase_mask)
		else:
			return (character_train_inputs, word_train_inputs, phrase_train_inputs, \
					oov_word_train_inputs, oov_phrase_train_inputs, character_mask, word_mask, phrase_mask)

	def handle_oov(self, character2word, word2phrase, oov_word_index, oov_phrase_index):
		character_of_oov_word = list()
		word_of_oov_phrase = list()
		oov_word_of_oov_phrase = list()
		oov_word_index_sorted = sorted(oov_word_index.iteritems(), key=lambda x:x[1], reverse = True)
		for k, v in oov_word_index_sorted:
			ins = copy.deepcopy(character2word[k])
			while(len(ins) < params.word_max_length):
				ins.append(0)
			if(len(ins) > params.word_max_length):
				ins = ins[0:params.word_max_length]
			character_of_oov_word.append(ins)
		oov_phrase_index_sorted = sorted(oov_phrase_index.iteritems(), key=lambda x:x[1], reverse = True)

		for k, v in oov_phrase_index_sorted:
			ins = copy.deepcopy(word2phrase[k])
			new_ins = list()
			# print oov_word_index

			# 这是我加的初始化代码，以保证oov_phra的oov_word和非oov_word的顺序对应
			for _ in range(len(ins)):
				new_ins.append(0)

			# print "new_ins_1:", new_ins
			for x in range(len(ins)):
				# if oov_word_index.has_key(ins[x]):
				if ins[x] in oov_word_index:
					
                                        
					# 以下代码有问题，这会导致oov_phrase的oov_word和非oov_word对应位的值不对应
					# new_ins.append(oov_word_index[ins[x]])
					# 以下是我增加的代码
					new_ins[x] = oov_word_index[ins[x]]

					ins[x] = 0


				"""
				# 这种情况在初始化中已经考虑了。
				else:
					new_ins.append(0)
				"""
			# print "new_ins_2:", new_ins

			while(len(ins) < params.phrase_max_length):
				ins.append(0)
			if(len(ins) > params.phrase_max_length):
				ins = ins[0:params.phrase_max_length]
			while(len(new_ins) < params.phrase_max_length):
				new_ins.append(0)
			if(len(new_ins) > params.phrase_max_length):
				new_ins = new_ins[0:params.phrase_max_length]
			word_of_oov_phrase.append(ins)
			oov_word_of_oov_phrase.append(new_ins)
		return (character_of_oov_word, word_of_oov_phrase, oov_word_of_oov_phrase)


#填补所有placeholder
	def next_batch(self):
		nl_oov_word_index = dict()
		nl_oov_phrase_index = dict()
		kb_oov_word_index = dict()
		kb_oov_phrase_index = dict()
		'''
		statements = self.statement_input_batches[self.pointer]
		facts = self.fact_input_batches[self.pointer]
		'''
		questions = self.question_input_batches[self.pointer]
		answers = self.answer_input_batches[self.pointer].tolist()
		#注意，neg_fact和neg_answer应从整个FB5M中取，这样负样例才不至于局限在sf,qa中
		train_triple_length = len(self.train_triples)
		'''
		neg_facts = [self.train_triples[random.randint(1, train_triple_length) - 1] for i in range(params.sf_neg_size)]
		'''
		neg_answers = list()
		for i in range(params.qa_neg_size):
			random_fact = self.train_triples[random.randint(1, train_triple_length) - 1]
			random_answer = [random_fact[0], random_fact[1]]
			neg_answers.append(random_answer)
		'''
		SF_NL_character_train_inputs, SF_NL_word_train_inputs, SF_NL_phrase_train_inputs, \
		SF_NL_oov_word_train_inputs, SF_NL_oov_phrase_train_inputs, SF_c2w_tree, SF_w2p_tree, SF_NL_character_mask, SF_NL_word_mask, SF_NL_phrase_mask = \
		handle_batch(statements, self.nl_character2word_dict, self.nl_word2phrase_dict, \
					 params.sf_nl_character_max_length, params.sf_nl_word_max_length, params.sf_nl_phrase_max_length, \
					 self.nl_word_oov_begin, self.nl_phrase_oov_begin, nl_oov_word_index, nl_oov_phrase_index, 1)
		SF_KB_character_train_inputs, SF_KB_word_train_inputs, SF_KB_phrase_train_inputs, \
		SF_KB_oov_word_train_inputs, SF_KB_oov_phrase_train_inputs, SF_KB_character_mask, SF_KB_word_mask, SF_KB_phrase_mask = \
		handle_batch(facts, self.kb_character2word_dict, self.kb_word2phrase_dict, \
					 params.sf_kb_character_max_length, params.sf_kb_word_max_length, params.sf_kb_phrase_max_length, \
					 self.kb_word_oov_begin, self.kb_phrase_oov_begin, kb_oov_word_index, kb_oov_phrase_index, 0)
		SF_KB_neg_character_train_inputs, SF_KB_neg_word_train_inputs, SF_KB_neg_phrase_train_inputs, \
		SF_KB_neg_oov_word_train_inputs, SF_KB_neg_oov_phrase_train_inputs, SF_KB_neg_character_mask, SF_KB_neg_word_mask, SF_KB_neg_phrase_mask = \
		handle_batch(neg_facts, self.kb_character2word_dict, self.kb_word2phrase_dict, \
					 params.sf_kb_character_max_length, params.sf_kb_word_max_length, params.sf_kb_phrase_max_length, \
					 self.kb_word_oov_begin, self.kb_phrase_oov_begin, kb_oov_word_index, kb_oov_phrase_index, 0)
		'''
		QA_NL_character_train_inputs, QA_NL_word_train_inputs, QA_NL_phrase_train_inputs, \
		QA_NL_oov_word_train_inputs, QA_NL_oov_phrase_train_inputs, QA_c2w_tree, QA_w2p_tree, QA_NL_character_mask, QA_NL_word_mask, QA_NL_phrase_mask = \
		self.handle_batch(questions, self.nl_character2word_dict, self.nl_word2phrase_dict, \
					 params.qa_nl_character_max_length, params.qa_nl_word_max_length, params.qa_nl_phrase_max_length, \
					 self.nl_word_oov_begin, self.nl_phrase_oov_begin, nl_oov_word_index, nl_oov_phrase_index, 1)
		QA_KB_character_train_inputs, QA_KB_word_train_inputs, QA_KB_phrase_train_inputs, \
		QA_KB_oov_word_train_inputs, QA_KB_oov_phrase_train_inputs, QA_KB_character_mask, QA_KB_word_mask, QA_KB_phrase_mask = \
		self.handle_batch(answers, self.kb_character2word_dict, self.kb_word2phrase_dict, \
					 params.qa_kb_character_max_length, params.qa_kb_word_max_length, params.qa_kb_phrase_max_length, \
					 self.kb_word_oov_begin, self.kb_phrase_oov_begin, kb_oov_word_index, kb_oov_phrase_index, 0)
		QA_KB_neg_character_train_inputs, QA_KB_neg_word_train_inputs, QA_KB_neg_phrase_train_inputs, \
		QA_KB_neg_oov_word_train_inputs, QA_KB_neg_oov_phrase_train_inputs, QA_KB_neg_character_mask, QA_KB_neg_word_mask, QA_KB_neg_phrase_mask = \
		self.handle_batch(neg_answers, self.kb_character2word_dict, self.kb_word2phrase_dict, \
					 params.qa_kb_character_max_length, params.qa_kb_word_max_length, params.qa_kb_phrase_max_length, \
					 self.kb_word_oov_begin, self.kb_phrase_oov_begin, kb_oov_word_index, kb_oov_phrase_index, 0)
		NL_character_of_oov_word, NL_word_of_oov_phrase, NL_oov_word_of_oov_phrase = \
		self.handle_oov(self.nl_character2word_dict, self.nl_word2phrase_dict, nl_oov_word_index, nl_oov_phrase_index)

		if len(NL_character_of_oov_word) == 0:
			NL_character_of_oov_word = []
			NL_character_of_oov_word.append([23, 13, 28])
			while len(NL_character_of_oov_word[0]) < params.word_max_length:
				NL_character_of_oov_word[0].append(0)
			# print NL_character_of_oov_word
			# print "complement of NL_character_of_oov_word over!"
			# time.sleep(10)

		if len(NL_word_of_oov_phrase) == 0:
			NL_word_of_oov_phrase = []
			NL_word_of_oov_phrase.append([467, 1575])
			while len(NL_word_of_oov_phrase[0]) < params.phrase_max_length:
				NL_word_of_oov_phrase[0].append(0)
			# print NL_word_of_oov_phrase
			# print "complement of NL_word_of_oov_phrase over"
			# time.sleep(10)

		if len(NL_oov_word_of_oov_phrase) == 0:
			NL_oov_word_of_oov_phrase = []
			NL_oov_word_of_oov_phrase.append([0] * params.phrase_max_length)
			# print NL_oov_word_of_oov_phrase
			# print "complement of NL_oov_word_of_oov_phrase over"
			# time.sleep(10)

		KB_character_of_oov_word, KB_word_of_oov_phrase, KB_oov_word_of_oov_phrase = \
		self.handle_oov(self.kb_character2word_dict, self.kb_word2phrase_dict, kb_oov_word_index, kb_oov_phrase_index)
		self.pointer = (self.pointer + 1) % self.qa_batch_num
		'''
		return SF_NL_character_train_inputs, SF_NL_word_train_inputs, SF_NL_phrase_train_inputs, \
				SF_KB_character_train_inputs, SF_KB_word_train_inputs, SF_KB_phrase_train_inputs, \
				SF_KB_neg_character_train_inputs, SF_KB_neg_word_train_inputs, SF_KB_neg_phrase_train_inputs, \
				QA_NL_character_train_inputs, QA_NL_word_train_inputs, QA_NL_phrase_train_inputs, \
				QA_KB_character_train_inputs, QA_KB_word_train_inputs, QA_KB_phrase_train_inputs, \
				QA_KB_neg_character_train_inputs, QA_KB_neg_word_train_inputs, QA_KB_neg_phrase_train_inputs, \
				SF_NL_oov_word_train_inputs, SF_NL_oov_phrase_train_inputs, SF_KB_oov_word_train_inputs, \
				SF_KB_oov_phrase_train_inputs, SF_KB_neg_oov_word_train_inputs, SF_KB_neg_oov_phrase_train_inputs, \
				QA_NL_oov_word_train_inputs, QA_NL_oov_phrase_train_inputs, QA_KB_oov_word_train_inputs, \
				QA_KB_oov_phrase_train_inputs, QA_KB_neg_oov_word_train_inputs, QA_KB_neg_oov_phrase_train_inputs, \
				NL_character_of_oov_word, NL_word_of_oov_phrase,	NL_oov_word_of_oov_phrase, \
				KB_character_of_oov_word, KB_word_of_oov_phrase, KB_oov_word_of_oov_phrase, \
				SF_c2w_tree, SF_w2p_tree, QA_c2w_tree, QA_w2p_tree, \
				SF_NL_character_mask, SF_NL_word_mask, SF_NL_phrase_mask, \
				SF_KB_character_mask, SF_KB_word_mask, SF_KB_phrase_mask, \
				SF_KB_neg_character_mask, SF_KB_neg_word_mask, SF_KB_neg_phrase_mask, \
				QA_NL_character_mask, QA_NL_word_mask, QA_NL_phrase_mask, \
				QA_KB_character_mask, QA_KB_word_mask, QA_KB_phrase_mask, \
				QA_KB_neg_character_mask, QA_KB_neg_word_mask, QA_KB_neg_phrase_mask
		'''


		"""
				for i in range(10):
					for j in range(20):
						print QA_NL_phrase_train_inputs[i][j], "\t", QA_NL_oov_phrase_train_inputs[i][j], "\n" * 3

				for i in range(10):
					for j in range(2):
						print QA_KB_phrase_train_inputs[i][j], "\t", QA_KB_oov_phrase_train_inputs[i][j], "\n" * 3
		"""

		return QA_NL_character_train_inputs, QA_NL_word_train_inputs, QA_NL_phrase_train_inputs, \
				QA_KB_character_train_inputs, QA_KB_word_train_inputs, QA_KB_phrase_train_inputs, \
				QA_KB_neg_character_train_inputs, QA_KB_neg_word_train_inputs, QA_KB_neg_phrase_train_inputs, \
				QA_NL_oov_word_train_inputs, QA_NL_oov_phrase_train_inputs, QA_KB_oov_word_train_inputs, \
				QA_KB_oov_phrase_train_inputs, QA_KB_neg_oov_word_train_inputs, QA_KB_neg_oov_phrase_train_inputs, \
				NL_character_of_oov_word, NL_word_of_oov_phrase,	NL_oov_word_of_oov_phrase, \
				KB_character_of_oov_word, KB_word_of_oov_phrase, KB_oov_word_of_oov_phrase, \
				QA_c2w_tree, QA_w2p_tree,\
				QA_NL_character_mask, QA_NL_word_mask, QA_NL_phrase_mask, \
				QA_KB_character_mask, QA_KB_word_mask, QA_KB_phrase_mask, \
				QA_KB_neg_character_mask, QA_KB_neg_word_mask, QA_KB_neg_phrase_mask

if __name__ == '__main__':
	loader = Textloader()
	loader.reset_batch_pointer()
	for i in range(4000):
		data = loader.next_batch()
		print "get the data of batch {0}......".format(i)
		d = {"QA_NL_character_train_inputs": data[0],
					"QA_NL_word_train_inputs": data[1],
					"QA_NL_phrase_train_inputs": data[2],
					"QA_KB_character_train_inputs": data[3],
					"QA_KB_word_train_inputs": data[4],
					"QA_KB_phrase_train_inputs": data[5],
					"QA_KB_neg_character_train_inputs": data[6],
					"QA_KB_neg_word_train_inputs": data[7],
					"QA_KB_neg_phrase_train_inputs": data[8],
					"QA_NL_oov_word_train_inputs": data[9],
					"QA_NL_oov_phrase_train_inputs": data[10],
					"QA_KB_oov_word_train_inputs": data[11],
					"QA_KB_oov_phrase_train_inputs": data[12],
					"QA_KB_neg_oov_word_train_inputs": data[13],
					"QA_KB_neg_oov_phrase_train_inputs": data[14],
					"NL_character_of_oov_word": data[15],
					"NL_word_of_oov_phrase": data[16],
					"NL_oov_word_of_oov_phrase": data[17],
					"KB_character_of_oov_word": data[18],
					"KB_word_of_oov_phrase": data[19],
					"KB_oov_word_of_oov_phrase": data[20],
					"QA_c2w_tree": data[21],
					"QA_w2p_tree": data[22],
					"QA_NL_character_mask": data[23],
					"QA_NL_word_mask": data[24],
					"QA_NL_phrase_mask": data[25],
					"QA_KB_character_mask": data[26],
					"QA_KB_word_mask": data[27],
					"QA_KB_phrase_mask": data[28],
					"QA_KB_neg_character_mask": data[29],
					"QA_KB_neg_word_mask": data[30],
					"QA_KB_neg_phrase_mask": data[31]}

		has_empty = False
		for k, v in d.iteritems():
			if len(v) == 0:
				print "boom!\n" * 3
				has_empty = True

		"""
		# if has_empty:

			with open("after_raw_data/batch {0} data".format(i), "a") as f:
				print >> f, "loader.nl_word_oov_begin", loader.nl_word_oov_begin, "\n" * 3
				for k, v in d.iteritems():
					
					print >> f, "*" * 50
					print >> f, "\t" * 5 , k
						
					for j in range(len(v)):
						print >> f, v[j]

					print >> f, "*" * 50, "\n" * 3
		"""


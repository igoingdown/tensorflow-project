#coding=utf-8

import numpy as np
import math
import random
import zmx_params as params
import copy
import cPickle
import time
import re


class TestTextLoader():
	def __init__(self):
		# read necessary dicts and test data from pickle file
		with open(params.test_candidate_pickle_path, 'rb') as f:
			self.test_candidates = cPickle.load(f)
			print "test_candidates length: ", len(self.test_candidates)
		with open(params.test_question_pickle_path, 'rb') as f:
			self.test_questions = cPickle.load(f)
			print "test_questions length: ", len(self.test_questions)
		with open(params.test_answer_pickle_path, 'rb') as f:
			self.test_answers = cPickle.load(f)
			print "test_answers length: ", len(self.test_answers)
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
		with open(params.nl_phrase_index_dict_pickle_path, 'rb') as f:
			self.nl_phrase_index_dict = cPickle.load(f)
		with open(params.nl_word_index_dict_pickle_path, 'rb') as f:
			self.nl_word_index_dict = cPickle.load(f)
		with open(params.nl_character_index_dict_pickle_path, 'rb') as f:
			self.nl_character_index_dict = cPickle.load(f)
		with open(params.kb_phrase_index_dict_pickle_path, 'rb') as f:
			self.kb_phrase_index_dict = cPickle.load(f)
		with open(params.kb_word_index_dict_pickle_path, 'rb') as f:
			self.kb_word_index_dict = cPickle.load(f)
		with open(params.kb_character_index_dict_pickle_path, 'rb') as f:
			self.kb_character_index_dict = cPickle.load(f)
		with open(params.nl_word4phrase_dict_pickle_path, 'rb') as f:
			self.nl_word4phrase_dict = cPickle.load(f)

		# some other fields
		self.pointer = 0
		self.nl_character_oov_begin = 39
		self.kb_character_oov_begin = 65
		self.test_questions_phrases_indices, self.test_candidates_indices, self.test_answer_indices = \
				self.change_natural_language_into_indices()
		"""
		test_questions_phrases_indices is a 2-D array, every question is represented as a list of the id of phrases.
		test_candidates_indices is a 3-D array, every question has many candidates, 
				every candidate is represented as a list of the id of phrases
		test_answer_indices is a 2-D array, every answer is represented as a list of the id of phrases.
		"""
		self.nl_word_oov_begin = self.detect_oov_begin(self.nl_word_dict_freq, params.freq_limit)
		self.nl_phrase_oov_begin = self.detect_oov_begin(self.nl_phrase_dict_freq, params.freq_limit)
		self.kb_word_oov_begin = self.detect_oov_begin(self.kb_word_dict_freq, params.freq_limit)
		self.kb_phrase_oov_begin = self.detect_oov_begin(self.kb_phrase_dict_freq, params.freq_limit)
		self.create_test_batches_a_question_is_a_batch()
		self.pointer = 0


	def create_test_batches_a_question_is_a_batch(self):
		# TODO: never seperate test_data into batches.
		self.qa_batch_num = len(self.test_questions_phrases_indices)
		self.question_input_batches = []
		for i in range(len(self.test_questions_phrases_indices)):
			self.question_input_batches.append([self.test_questions_phrases_indices[i] for j in range(len(self.test_candidates_indices[i]))])
		self.answer_input_batches = []
		for i in xrange(len(self.test_questions)):
			self.answer_input_batches.append([self.test_answer_indices[i] for j in range(len(self.test_candidates_indices[i]))])
		self.candidate_input_batches = self.test_candidates_indices
		self.pointer = 0


	def identify_phrases(self, word4phrase, words, wordisphrase):
		words_tuple = tuple(words)
		phrases = []
		length = len(words)
		last_pos = 0
		while last_pos < length:
			for pos in range(length, last_pos-1,-1):
				seg = words_tuple[last_pos:pos]
				rescan = 0
				if len(seg) == 1:
					word = seg[0]
					rescan = 1
				elif word4phrase.has_key(seg):
					phrase = word4phrase[seg]
					rescan = 2
				else:
					pass
				if rescan == 0:
					continue
				else:
					last_pos = pos
					if rescan == 1:
						if wordisphrase:
							phrases.append(word)
					if rescan == 2:
						phrases.append(phrase)
					break
		return phrases


	# nl_phrase_index_dict, kb_phrase_index_dict 都不全，后期要想办法补全！
	def change_natural_language_into_indices(self):
		# TODO: change question, candidate s-r pairs and answer into indentical indices in our system
		test_questions_words_natural_language = [self.split_into_words(x) for x in self.test_questions]
		test_questions_phrases_natural_language = \
				[self.identify_phrases(self.nl_word4phrase_dict, x, False) for x in test_questions_words_natural_language]
		test_questions_phrases_indices = []
		for q in test_questions_phrases_natural_language:
			indices = []
			for p in q:
				if p in self.nl_phrase_index_dict:
					indices.append(self.nl_phrase_index_dict[p])
			test_questions_phrases_indices.append(indices)

		test_candidates_indices = []
		for candidates_of_one_question in self.test_candidates:
			indices = []
			for candidate in candidates_of_one_question:
				s = candidate[0]
				r = candidate[1]
				if s in self.kb_phrase_index_dict and r in self.kb_phrase_index_dict:
					indices.append([self.kb_phrase_index_dict[s], self.kb_phrase_index_dict[r]])
			test_candidates_indices.append(indices)

		test_answer_indices = []
		for answer in self.test_answers:
			indices = []
			s = answer[0]
			r = answer[1]
			if s in self.kb_phrase_index_dict and r in self.kb_phrase_index_dict:
				indices.append(self.kb_phrase_index_dict[s])
				indices.append(self.kb_phrase_index_dict[r])
			test_answer_indices.append(indices)
		return test_questions_phrases_indices, test_candidates_indices, test_answer_indices


	def detect_oov_begin(self, idict, freq_limit):
		length = len(idict)
		index_begin = 3
		for index in range(index_begin, length + index_begin):
			if idict[index] < freq_limit:
				return index
		return length + index_begin


	def split_into_words(self, content):
		return filter(lambda x: not re.match('\s',x) and x != '', 
				re.split("([`|~|\!|@|#|\$|%|\^|&|\*|\(|\)|\-|_|\=|\+|\[|\{|\]|\}|\||\\|;|\|:|'|\"|\,|\<|\.|\>|\/|\?|\s])", content))


	# input phrase sequence, generate corect shape character sequence, word sequence and phrase sequence
	# test_questions_phrases_indices and test_answer_indices can call this funtion directly
	# while test_candidates_indices can't because it's a 3-D array, parameter sequence must be a 2-D phrase-indice sequence
	# be careful about this hint!
	def truncate_and_transform_seq(self, sequence, character2word, word2phrase, characterseq_max_length, 
				wordseq_max_length, phraseseq_max_length, isnl):
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


	def handle_batch(self, batch, character2word, word2phrase, 
				characterseq_max_length, wordseq_max_length, phraseseq_max_length, 
				word_oov_begin, phrase_oov_begin, oov_word_index, oov_phrase_index, isnl):
		character_test_inputs = list()
		word_test_inputs = list()
		phrase_test_inputs = list()
		oov_word_test_inputs = list()
		oov_phrase_test_inputs = list()
		character_mask = list()
		word_mask = list()
		phrase_mask = list()
		c2w_tree = list()
		w2p_tree = list()

		for sequence in batch:
			# 每个batch中有batch_size个问题（对应NL）或者答案（对应KB）对应的词组序列。
			characters, words, phrases = self.truncate_and_transform_seq(sequence, character2word, word2phrase, 
					characterseq_max_length, wordseq_max_length, phraseseq_max_length, isnl)
			#print phrases
			oov_words = list()
			oov_phrases = list()

			for _ in range(len(phrases)):
				oov_phrases.append(0)
			for _ in range(len(words)):
				oov_words.append(0)

			c2w_slice = list()
			w2p_slice = list()
			c2w_index = 0
			w2p_index = 0
			for x in range(len(words)):
				word = words[x]
				c2w_line = [0]*characterseq_max_length
				if word == 0 or word == 1 or word == 2:
					c2w_slice.append(c2w_line)
				else:
					for z in range(c2w_index,c2w_index+len(character2word[word])+2):
						c2w_line[z] = 1
					c2w_index = c2w_index + len(character2word[word])+2
					c2w_slice.append(c2w_line)
					if word >= word_oov_begin:
						if not oov_word_index.has_key(word):
							oov_word_index[word] = len(oov_word_index) + 1;
						words[x] = 0
						oov_words[x] = oov_word_index[word]

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
							oov_phrase_index[phrase] = len(oov_phrase_index) + 1

						phrases[y] = 0
						oov_phrases[y] = oov_phrase_index[phrase]

			character_mask_line = [1]*len(characters)
			word_mask_line = [1]*len(words)
			phrase_mask_line = [1]*len(phrases)

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
			character_test_inputs.append(characters)
			word_test_inputs.append(words)
			phrase_test_inputs.append(phrases)
			oov_word_test_inputs.append(oov_words)
			oov_phrase_test_inputs.append(oov_phrases)
			c2w_tree.append(c2w_slice)
			w2p_tree.append(w2p_slice)
			character_mask.append(character_mask_line)
			word_mask.append(word_mask_line)
			phrase_mask.append(phrase_mask_line)
		if isnl:
			return (character_test_inputs, word_test_inputs, phrase_test_inputs, 
					oov_word_test_inputs, oov_phrase_test_inputs, c2w_tree, w2p_tree, character_mask, word_mask, phrase_mask)
		else:
			return (character_test_inputs, word_test_inputs, phrase_test_inputs, 
					oov_word_test_inputs, oov_phrase_test_inputs, character_mask, word_mask, phrase_mask)


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

			for _ in range(len(ins)):
				new_ins.append(0)

			for x in range(len(ins)):
				if ins[x] in oov_word_index:
					new_ins[x] = oov_word_index[ins[x]]

					ins[x] = 0

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


	# 填补所有placeholder, 不能让placeholder为空，在GPU环境下会报错！
	# deal with it as there is only one batch.
	# dealing with candidates is different from dealing with questions and answers.
	# because candidates is a 3-D arrary, questions and questions are 2-D arrays
	def next_batch(self):
		nl_oov_word_index = dict()
		nl_oov_phrase_index = dict()
		kb_oov_word_index = dict()
		kb_oov_phrase_index = dict()
		kb_oov_word_index_for_answers = dict()
		kb_oov_phrase_index_for_answers = dict()

		questions = self.question_input_batches[self.pointer]
		answers = self.answer_input_batches[self.pointer]
		candidates = self.candidate_input_batches[self.pointer]

		answer_pos = [0 for _ in range(len(candidates))]
		for i in xrange(len(candidates)):
			if answers[i] == candidates[i]:
				answer_pos[i] = 1

		# deal with questions
		QA_NL_character_test_inputs, QA_NL_word_test_inputs, QA_NL_phrase_test_inputs, \
				QA_NL_oov_word_test_inputs, QA_NL_oov_phrase_test_inputs, QA_c2w_tree, QA_w2p_tree, \
				QA_NL_character_mask, QA_NL_word_mask, QA_NL_phrase_mask = \
				self.handle_batch(questions, self.nl_character2word_dict, self.nl_word2phrase_dict, 
						params.qa_nl_character_max_length, params.qa_nl_word_max_length, params.qa_nl_phrase_max_length, 
						self.nl_word_oov_begin, self.nl_phrase_oov_begin, nl_oov_word_index, nl_oov_phrase_index, 1)

		"""
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
		"""

		NL_character_of_oov_word, NL_word_of_oov_phrase, NL_oov_word_of_oov_phrase = \
				self.handle_oov(self.nl_character2word_dict, self.nl_word2phrase_dict, nl_oov_word_index, nl_oov_phrase_index)

		# deal with candidates
		QA_KB_character_test_inputs, QA_KB_word_test_inputs, QA_KB_phrase_test_inputs, \
				QA_KB_oov_word_test_inputs, QA_KB_oov_phrase_test_inputs, \
				QA_KB_character_mask, QA_KB_word_mask, QA_KB_phrase_mask \
				= self.handle_batch(candidates, self.kb_character2word_dict, self.kb_word2phrase_dict, 
						params.qa_kb_character_max_length, params.qa_kb_word_max_length, params.qa_kb_phrase_max_length, 
						self.kb_word_oov_begin, self.kb_phrase_oov_begin, 
						kb_oov_word_index, kb_oov_phrase_index, 0)

		KB_character_of_oov_word, KB_word_of_oov_phrase, KB_oov_word_of_oov_phrase = [], [], []
		character_of_oov_word, word_of_oov_phrase, oov_word_of_oov_phrase = \
				self.handle_oov(self.kb_character2word_dict, self.kb_word2phrase_dict, kb_oov_word_index, kb_oov_phrase_index)
		KB_character_of_oov_word.append(character_of_oov_word)

		KB_word_of_oov_phrase.append(word_of_oov_phrase)
		KB_oov_word_of_oov_phrase.append(oov_word_of_oov_phrase)

		# deal with answers
		QA_KB_character_test_inputs_for_answers, QA_KB_word_test_inputs_for_answers, QA_KB_phrase_test_inputs_for_answers, \
				QA_KB_oov_word_test_inputs_for_answers, QA_KB_oov_phrase_test_inputs_for_answers, QA_KB_character_mask_for_answers, \
				QA_KB_word_mask_for_answers, QA_KB_phrase_mask_for_answers = \
				self.handle_batch(answers, self.kb_character2word_dict, self.kb_word2phrase_dict, 
						params.qa_kb_character_max_length, params.qa_kb_word_max_length, params.qa_kb_phrase_max_length, 
						self.kb_word_oov_begin, self.kb_phrase_oov_begin, kb_oov_word_index, kb_oov_phrase_index, 0)
		KB_character_of_oov_word_for_answers, KB_word_of_oov_phrase_for_answers, KB_oov_word_of_oov_phrase_for_answers = \
				self.handle_oov(self.kb_character2word_dict, self.kb_word2phrase_dict, 
						kb_oov_word_index_for_answers, kb_oov_phrase_index_for_answers)

		self.pointer = (self.pointer + 1) % self.qa_batch_num

		return QA_NL_character_test_inputs, QA_NL_word_test_inputs, QA_NL_phrase_test_inputs, \
				QA_KB_character_test_inputs, QA_KB_word_test_inputs, QA_KB_phrase_test_inputs, \
				QA_KB_character_test_inputs_for_answers, QA_KB_word_test_inputs_for_answers, QA_KB_phrase_test_inputs_for_answers, \
				QA_NL_oov_word_test_inputs, QA_NL_oov_phrase_test_inputs, \
				QA_KB_oov_word_test_inputs, QA_KB_oov_phrase_test_inputs, \
				QA_KB_oov_word_test_inputs_for_answers, QA_KB_oov_phrase_test_inputs_for_answers, \
				NL_character_of_oov_word, NL_word_of_oov_phrase, NL_oov_word_of_oov_phrase, \
				KB_character_of_oov_word, KB_word_of_oov_phrase, KB_oov_word_of_oov_phrase, \
				KB_character_of_oov_word_for_answers, KB_word_of_oov_phrase_for_answers, KB_oov_word_of_oov_phrase_for_answers, \
				QA_c2w_tree, QA_w2p_tree, \
				QA_NL_character_mask, QA_NL_word_mask, QA_NL_phrase_mask, \
				QA_KB_character_mask, QA_KB_word_mask, QA_KB_phrase_mask, \
				QA_KB_character_mask_for_answers, QA_KB_word_mask_for_answers, QA_KB_phrase_mask_for_answers, \
				answer_pos


if __name__ == '__main__':
	loader = TestTextLoader()
	data = loader.next_batch()

	d = {"QA_NL_character_test_inputs": data[0], 
			"QA_NL_word_test_inputs": data[1], 
			"QA_NL_phrase_test_inputs": data[2], 
			"QA_KB_character_test_inputs": data[3], 
			"QA_KB_word_test_inputs": data[4], 
			"QA_KB_phrase_test_inputs": data[5], 
			"QA_KB_character_test_inputs_for_answers": data[6], 
			"QA_KB_word_test_inputs_for_answers": data[7], 
			"QA_KB_phrase_test_inputs_for_answers": data[8], 
			"QA_NL_oov_word_test_inputs": data[9], 
			"QA_NL_oov_phrase_test_inputs": data[10], 
			"QA_KB_oov_word_test_inputs": data[11], 
			"QA_KB_oov_phrase_test_inputs": data[12],
			"QA_KB_oov_word_test_inputs_for_answers": data[13], 
			"QA_KB_oov_phrase_test_inputs_for_answers": data[14],
			"NL_character_of_oov_word": data[15], 
			"NL_word_of_oov_phrase": data[16], 
			"NL_oov_word_of_oov_phrase": data[17], 
			"KB_character_of_oov_word": data[18], 
			"KB_word_of_oov_phrase": data[19], 
			"KB_oov_word_of_oov_phrase": data[20], 
			"KB_character_of_oov_word_for_answers": data[21], 
			"KB_word_of_oov_phrase_for_answers": data[22], 
			"KB_oov_word_of_oov_phrase_for_answers": data[23], 
			"QA_c2w_tree": data[24], 
			"QA_w2p_tree": data[25], 
			"QA_NL_character_mask": data[26], 
			"QA_NL_word_mask": data[27], 
			"QA_NL_phrase_mask": data[28], 
			"QA_KB_character_mask": data[29], 
			"QA_KB_word_mask": data[30], 
			"QA_KB_phrase_mask": data[31], 
			"QA_KB_character_mask_for_answers": data[32], 
			"QA_KB_word_mask_for_answers": data[33], 
			"QA_KB_phrase_mask_for_answers": data[34],
			"ans_pos": data[35]}

	has_empty = False
	for k, v in d.iteritems():
		if len(v) == 0:
			print k
			print "-" * 100, "\nsome placeholders are empty!\n", "-" * 100, "\n\n"
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








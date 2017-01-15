#!/usr/bin/env python
# -*-coding:utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import multiprocessing
from multiprocessing import Pool
import os
import random
import zmx_params as params
from time_wrapper import *
import time
import cPickle
import re



@time_recorder
def load_dicts_from_file():
	with open(params.test_word_subjects_dict_pickle_path, 'rb') as f:
		word_subjects_dict = cPickle.load(f)
	with open(params.test_question_pickle_path, 'rb') as f:
		test_questions = cPickle.load(f)
	with open(params.test_alias_dict_pickle_path, 'rb') as f:
		alias_dict = cPickle.load(f)

	delete_keys = []
	for k, v in word_subjects_dict.iteritems():
		if len(k) < 2:
			delete_keys.append(k)
	for k in delete_keys:
		del word_subjects_dict[k]

	return word_subjects_dict, test_questions, alias_dict


"""
	find overlap between subjects and the question
"""
def get_overlaps(question, candidate_subjects, alias_dict):
	overlap_list = []
	for subject in candidate_subjects:
		subject_string = alias_dict[subject]
		overlap_list.append(lcs(question, subject_string))
	return overlap_list


"""
	implement the algorithm that finds the longest continuous string(lcs for short) of two strings
"""
def lcs(input_x, input_y):
	# input_y as column, input_x as row
	dp = [([0] * len(input_y)) for _ in range(len(input_x))]
	max_len = max_index = 0
	for i in range(0, len(input_x)):
		for j in range(0, len(input_y)):
			if input_x[i] == input_y[j]:
				if i != 0 and j != 0:
					dp[i][j] = dp[i - 1][j - 1] + 1
				if i == 0 or j == 0:
					dp[i][j] = 1
				if dp[i][j] > max_len:
					max_len = dp[i][j]
					max_index = i + 1 - max_len
	return input_x[max_index:max_index + max_len]


"""
	find the candidate subjects for one question
"""
def get_candidate_subjects(question, word_subjects_dict):
	candidate_subjects = set()
	question_words = split_into_words(question.lower())
	for word in question_words:
		if word in word_subjects_dict:
			candidate_subjects |= word_subjects_dict[word]
	return candidate_subjects


def split_into_words(content):
	return filter(lambda x: not re.match('\s',x) and x != '', 
		re.split("([`|~|\!|@|#|\$|%|\^|&|\*|\(|\)|\-|_|\=|\+|\[|\{|\]|\}|\||\\|;|\|:|'|\"|\,|\<|\.|\>|\/|\?|\s])", content))


"""
	get the ratio of every overlap in one question
	each overlap represent the overlap string of the candidate_subject and the question
	return a list of ratios
"""
def get_overlap_question_ratio(question, overlaps):
	overlap_ratios_in_question = []
	for overlap in overlaps:
		overlap_ratios_in_question.append(len(overlap) / len(question))
	return overlap_ratios_in_question


"""
	get the ratio of every overlap in the correspondiong subject
	each overlap represent the overlap string of the correspondiong candidate_subject and the question
	return a list of ratios
"""
def get_overlap_subject_ratio(overlaps, subjects, alias_dict):
	overlap_ratios_in_subject = []
	for i in xrange(len(overlaps)):
		overlap_ratios_in_subject.append(len(overlaps[i]) / len(alias_dict[subjects[i]]))
	return overlap_ratios_in_subject


"""
	get the position of every overlap in one question
	each overlap represent the overlap string of the correspondiong candidate_subject and the question
	return a list of positions
"""
def get_overlap_position_in_question(question, overlaps):
	pos_list = []
	for overlap in overlaps:
		pos = question.index(overlap)
		if pos < 0:
			print "can\'t find the overlap in the question!!"
		pos_list.append(pos)
	return pos_list


def get_candidate_top_k_subjects_by_overlap_length(subjects, overlaps, k, alias_dict):
	candidate_subjects = []
	subject_overlap_length_dict = {}
	subjects_list = list(subjects)
	for i in xrange(len(subjects)):
		subject_overlap_length_dict[subjects_list[i]] = len(overlaps[i])
	sorted_subject_overlap_length_list = sorted(subject_overlap_length_dict.iteritems(), key=lambda x: x[1], reverse=True)
	if len(sorted_subject_overlap_length_list) < k:
		print "length of the candidate_subjects_list is less than k, and k is {0}".format(k)

	for i in xrange(min(k, len(sorted_subject_overlap_length_list))):
		candidate_subjects.append(sorted_subject_overlap_length_list[i][0])
	return candidate_subjects


@time_recorder
def save_as_file(candidate_subjects_for_all_questions):
	# params should be changed!
	with open(params.test_candidate_subjects_for_all_questions_pickle_path_multiprocessing, 'wb') as f:
		cPickle.dump(candidate_subjects_for_all_questions, f)


def deal_with_one_question(question):
	global word_subjects_dict, alias_dict
	candidate_subjects_without_filtering = get_candidate_subjects(question, word_subjects_dict)
	overlaps = get_overlaps(question, candidate_subjects_without_filtering, alias_dict)
	candidate_subjects = get_candidate_top_k_subjects_by_overlap_length(candidate_subjects_without_filtering, 
			overlaps, 100, alias_dict)
	return candidate_subjects


if __name__ == '__main__':
	word_subjects_dict, questions, alias_dict = load_dicts_from_file()
	print "the root pid of this program is {0}".format(os.getpid())
	print "cpu_count: {0}".format(multiprocessing.cpu_count())
	p = Pool(multiprocessing.cpu_count())
	candidate_subjects_for_all_questions = p.map(deal_with_one_question, questions)
	print "all questions done!"

	save_as_file(candidate_subjects_for_all_questions)
	with open("zmx_candidate_top100_subjects_for_all_questions_multiprocessing_version.txt", "w") as f:
		for candidate_subjects in candidate_subjects_for_all_questions:
			print >> f, "subjects for one question:"
			for subject in candidate_subjects:
				print >> f, subject
			print >> f, "\n" * 3













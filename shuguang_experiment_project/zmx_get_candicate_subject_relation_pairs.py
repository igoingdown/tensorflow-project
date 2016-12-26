#!/usr/bin/env python
# -*-coding:utf-8 -*-

import sys
import zmx_params as params
from multiprocessing import Pool
import multiprocessing
from time_wrapper import *
import cPickle
reload(sys)
sys.setdefaultencoding('utf-8')


@time_recorder
def load_dicts_from_file():
	"""
	test_subject_relations_dict may need to modified,
	because it only contains the sujects in the test_set!
	"""
	with open(params.test_subject_relations_dict_pickle_path, 'rb') as f:
		subject_relations_dict = cPickle.load(f)
	with open(params.test_candidate_subjects_for_all_questions_pickle_path_multiprocessing, 'rb') as f:
		candidate_subjects_for_all_questions = cPickle.load(f)
	return subject_relations_dict, candidate_subjects_for_all_questions


@time_recorder
def save_as_file(subject_relation_pairs):
	with open(params.test_candidate_pickle_path, 'wb') as f:
		cPickle.dump(subject_relation_pairs, f)


def deal_with_subjects_for_one_question(candidate_subjects_for_one_question):
	global subject_relations_dict
	candidate_subject_relation_pairs_for_one_question = []
	for subject in candidate_subjects_for_one_question:
		if subject in subject_relations_dict:
			relations = subject_relations_dict[subject]
			for relation in relations:
				candidate_subjects_for_one_question.append((subject, relation))
	return candidate_subject_relation_pairs_for_one_question


@time_recorder
def get_subject_relation_pairs(candidate_subjects_for_all_questions, subject_relations_dict):
	candidate_subject_relation_pairs_for_all_questions = []
	for candidate_subjects_for_one_question in candidate_subjects_for_all_questions:
		candidate_subject_relation_pairs_for_one_question = []
		for subject in candidate_subjects_for_one_question:
			if subject in subject_relations_dict:
				relations = subject_relations_dict[subject]
				for relation in relations:
					candidate_subject_relation_pairs_for_one_question.append((subject, relation))
		candidate_subject_relation_pairs_for_all_questions.append(candidate_subject_relation_pairs_for_one_question)
	return candidate_subject_relation_pairs_for_all_questions


if __name__ == '__main__':
	subject_relations_dict, candidate_subjects_for_all_questions = load_dicts_from_file()
	print len(candidate_subjects_for_all_questions)

	"""
	# these codes use multiprocess to acceralate
	# however there are some bugs......
	p = Pool(multiprocessing.cpu_count())
	candidate_subject_relation_pairs_for_all_questions = p.map(deal_with_subjects_for_one_question, candidate_subjects_for_all_questions)
	"""
	candidate_subject_relation_pairs_for_all_questions = get_subject_relation_pairs(candidate_subjects_for_all_questions, subject_relations_dict)

	no_candidate_number = 0
	for candidate_subject_relation_pairs_for_one_question in candidate_subject_relation_pairs_for_all_questions:
		if len(candidate_subject_relation_pairs_for_one_question) == 0:
			no_candidate_number += 1
	print " no_candidate_number: ", no_candidate_number

	save_as_file(candidate_subject_relation_pairs_for_all_questions)




























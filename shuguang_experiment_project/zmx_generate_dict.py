#coding=utf-8

import zmx_params as params
from time_wrapper import *
import time
import cPickle
import re
from zmx_virtuoso_api import *
from multiprocessing import Pool
import multiprocessing


def split_into_words(content):
	return filter(lambda x: not re.match('\s',x) and x != '', 
		re.split("([`|~|\!|@|#|\$|%|\^|&|\*|\(|\)|\-|_|\=|\+|\[|\{|\]|\}|\||\\|;|\|:|'|\"|\,|\<|\.|\>|\/|\?|\s])", content))


@time_recorder
def get_suject_words_dict(alias_dict):
	subject_words_dict = dict()
	for k, v in alias_dict.iteritems():
		words = split_into_words(v)
		subject_words_dict[k] = words
	return subject_words_dict


@time_recorder
def get_word_sujects_dict(alias_dict, subject_words_dict):
	# word-sujects dict, used when choosing subject candidates
	word_subjects_dict = dict()
	"""
	word_subjects_dict: key is a word.
	value is the id of a set of subjects whose corpus include the word
	split question into words, use these words to look up subject candidates
	maybe some words are not in the dict! such as what.
	"""

	counter = 0
	for k, v in alias_dict.iteritems():
		# print "processing number {0} subject, the subject is {1}, its alias is {2}".format(counter, k, v)
		counter += 1

		words = split_into_words(v)
		for word in words:
			if word not in word_subjects_dict:
				word_subjects_dict[word] = set()

			word_subjects_dict[word].add(k)	

	return word_subjects_dict


	"""
	# the following function is not as effective as the funtion above!
	# so it's abandoned

	# TODO: read qa_test file, generate the mapping between word of question and subjects
	with open(params.qa_test_path) as f:
		for line in f:
			content = line.strip().split('\t')
			subject = content[0].replace("www.freebase.com","")
			question = content[3]
			question_words = split_into_words(question.lower())
			print "subject:", subject
			for word in question_words:
				if word not in word_subjects_dict:
					word_subjects_dict[word] = set()
				for k, v in subject_words_dict.iteritems():
					if word in v and k in alias_dict:
						word_subjects_dict[word].add(alias_dict[k])

	end_time = time.time()
	dur = end_time - start_time
	print "get_word_sujects_dict over......"
	print "the time get_word_sujects_dict used is {0}".format(dur)

	return word_subjects_dict
	"""


@time_recorder
def get_relation_sujects_dict():
	# relation and subjects dict, used when choosing relation candidates and subject candidates
	relation_sujects_dict = dict()

	"""
	with open(params.qa_path) as f:
		for line in f:
			content = line.strip().split('\t')
			subject = content[0].replace("www.freebase.com","")
			relation = content[1].replace("www.freebase.com","")

			if relation not in relation_sujects_dict:
				relation_sujects_dict[relation] = set()

			print relation, "\t", subject
			relation_sujects_dict[relation].add(subject)
	"""

	with open(params.fact_path) as f:
		for line in f:
			content = line.strip().split('\t')
			subject = content[0].replace("www.freebase.com","")
			relation = content[1].replace("www.freebase.com","")

			if relation not in relation_sujects_dict:
				relation_sujects_dict[relation] = set()
			relation_sujects_dict[relation].add(subject)

	"""
	for k, v in relation_sujects_dict.iteritems():
		print "*" * 100
		print k, "\n" * 2
		for subject in v:
			print subject
		print "*" * 100, "\n" * 3
	"""

	return relation_sujects_dict


@time_recorder
def get_alias_dict():
	# alias_dict, key is alias, value is the natural language words which represent the entity
	alias_dict = dict()

	with open(params.fb2m_type_path) as f:
		for line in f:
			content = line.strip().split('\t')
			mid = content[0]
			alias = content[1].lower()
			alias_dict[mid] = alias

	with open(params.fb5m_2m_type_path) as f:
		for line in f:
			content = line.strip().split('\t')
			mid = content[0]
			alias = content[1].lower()
			alias_dict[mid] = alias

	return alias_dict


@time_recorder
def get_subject_relations_dict(relation_sujects_dict):
	subject_relations_dict = {}
	"""
	subject_relations_dict: key is the id of subject.
	value is a set of relations whose subject includes the key
	"""

	for k, v in relation_sujects_dict.iteritems():
		for s in v:
			if s not in subject_relations_dict:
				subject_relations_dict[s] = set()
			subject_relations_dict[s].add(k)

	return subject_relations_dict


@time_recorder
def save_as_file(relation_sujects_dict, word_subjects_dict, subject_relations_dict, alias_dict):
	with open(params.test_subject_relations_dict_pickle_path, 'wb') as f:
		cPickle.dump(subject_relations_dict, f)
	with open(params.test_relation_sujects_dict_pickle_path, 'wb') as f:
		cPickle.dump(relation_sujects_dict, f)
	with open(params.test_word_subjects_dict_pickle_path, 'wb') as f:
		cPickle.dump(word_subjects_dict, f)
	with open(params.test_alias_dict_pickle_path, 'wb') as f:
		cPickle.dump(alias_dict, f)
	with open(params.test_word_subjects_dict_txt_path, 'w') as f:
		print >> f, "~" * 40
		for k, v in word_subjects_dict.iteritems():
			for subject in v:
				print >> f, k, "\t", subject
			print >> f, "~" * 40


if __name__ == '__main__':
	alias_dict = get_alias_dict()
	"""
	subject_words_dict = get_suject_words_dict(alias_dict)
	relation_sujects_dict = get_relation_sujects_dict()
	word_subjects_dict = get_word_sujects_dict(alias_dict, subject_words_dict)
	subject_relations_dict = get_subject_relations_dict(relation_sujects_dict)
	save_as_file(relation_sujects_dict, word_subjects_dict, subject_relations_dict, alias_dict)
	"""
	p = Pool(10)
	id_list = []
	for k,v in alias_dict.iteritems():
		subject_id = k.strip("/").replace("/", ".")
		id_list.append(subject_id)

	alias_list = p.map(sparql_query, id_list)
	with open("zmx_test_alias_list.txt", "w") as f:
		for alias in alias_list:
			print >> f, alias

	with open("zmx_test_id_list.txt", "w") as f:
		for x in id_list:
			print >> f, x

















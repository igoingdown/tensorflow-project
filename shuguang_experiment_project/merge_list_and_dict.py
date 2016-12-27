#!/usr/bin/python
# -*- coding:utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import zmx_params as params
from time_wrapper import *
import cPickle


def load_pickle(name):
	with open(name) as f:
		list_or_dict = cPickle.load(f)
	return list_or_dict


def merge_list_and_dict(found_dict_6, found_dict_7, not_found_list_6, not_found_list_7, dict_path, list_path):
	found_dict = dict(found_dict_6.items() + found_dict_7.items())
	not_found_list = list(set(not_found_list_7 + not_found_list_6))
	merge_list = []
	for i in not_found_list:
		if i not in found_dict:
			merge_list.append(i)
	with open(dict_path, "w") as f:
		cPickle.dump(found_dict, f)
	with open(list_path, "w") as f:
		cPickle.dump(merge_list, f)
	return found_dict, merge_list


if __name__ == '__main__':
	test_list_6 = load_pickle(params.test_set_subject_without_name_yet_list_pickle_path)
	test_dict_6 = load_pickle(params.test_set_subject_name_dict_pickle_path)
	test_dict_7 = load_pickle(params.test_set_subject_name_dict_pickle_path_virtuoso_7)
	test_list_7 = load_pickle(params.test_set_subject_without_name_yet_list_pickle_path_virtuoso_7)
	found_dict, merge_list = merge_list_and_dict(test_dict_6, test_dict_7, test_list_6, test_list_7, \
			params.merged_dict_of_test_set_pickle_path, params.merged_list_of_test_set_pickle_path)
	print "merged_test"
	print len(found_dict)
	print len(list(set(merge_list)))


	train_list_6 = load_pickle(params.train_set_subject_without_name_yet_list_pickle_path)
	train_dict_6 = load_pickle(params.train_set_subject_name_dict_pickle_path)
	train_dict_7 = load_pickle(params.train_set_subject_name_dict_pickle_path_virtuoso_7)
	train_list_7 = load_pickle(params.train_set_subject_without_name_yet_list_pickle_path_virtuoso_7)
	found_dict, merge_list = merge_list_and_dict(train_dict_6, train_dict_7, train_list_6, train_list_7, \
			params.merged_dict_of_train_set_pickle_path, params.merged_list_of_train_set_pickle_path)
	print "merged_train"
	print len(found_dict)
	print len(list(set(merge_list)))


	valid_list_6 = load_pickle(params.valid_set_subject_without_name_yet_list_pickle_path)
	valid_dict_6 = load_pickle(params.valid_set_subject_name_dict_pickle_path)
	valid_dict_7 = load_pickle(params.valid_set_subject_name_dict_pickle_path_virtuoso_7)
	valid_list_7 = load_pickle(params.valid_set_subject_without_name_yet_list_pickle_path_virtuoso_7)
	found_dict, merge_list = merge_list_and_dict(valid_dict_6, valid_dict_7, valid_list_6, valid_list_7, \
			params.merged_dict_of_valid_set_pickle_path, params.merged_list_of_valid_set_pickle_path)
	print "merged_valid"
	print len(found_dict)
	print len(list(set(merge_list)))



































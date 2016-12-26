#!/usr/bin/python
# -*- coding:utf-8 -*-

import zmx_params as params
from time_wrapper import *
import cPickle

import sys
reload(sys)
sys.setdefaultencoding('utf-8')


def load_pickle(name):
	with open(name) as f:
		list_or_dict = cPickle.load(f)
	return list_or_dict


if __name__ == '__main__':
	merge_dict_test = load_pickle(params.merged_dict_of_test_set_pickle_path)
	merge_dict_train = load_pickle(params.merged_dict_of_train_set_pickle_path)
	merge_dict_valid = load_pickle(params.merged_dict_of_valid_set_pickle_path)
	merge_list_test = load_pickle(params.merged_list_of_test_set_pickle_path)
	merge_list_train = load_pickle(params.merged_list_of_train_set_pickle_path)
	merge_list_valid = load_pickle(params.merged_list_of_valid_set_pickle_path)

	fb5m_dict_6 = load_pickle(params.fb5m_subject_name_dict_pickle_path)
	fb5m_dict_7 = load_pickle(params.fb5m_subject_name_dict_pickle_path_virtuoso_7)
	fb5m_list_6 = load_pickle(params.fb5m_subject_without_name_yet_list_pickle_path)
	fb5m_list_7 = load_pickle(params.fb5m_subject_without_name_yet_list_pickle_path_virtuoso_7)

	all_merge_dict = dict(fb5m_dict_6.items() + fb5m_dict_7.items() + merge_dict_train.items() + merge_dict_test.items() + merge_dict_valid.items())
	merge_list = list(set(merge_list_test + merge_list_train + merge_list_valid + fb5m_list_6 + fb5m_list_7))
	all_merge_list = []
	for i in merge_list:
		if i not in all_merge_dict:
			all_merge_list.append(i)

	with open(params.all_merge_dict_pickle_path, "w") as f:
		cPickle.dump(all_merge_dict, f)
	with open(params.all_merge_list_pickle_path, "w") as f:
		cPickle.dump(all_merge_list, f)

	print "all_merged:"
	print len(all_merge_dict)
	print len(list(set(all_merge_list)))


























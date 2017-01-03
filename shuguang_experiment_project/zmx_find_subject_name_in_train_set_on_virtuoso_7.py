#!/usr/bin/python
# -*- coding:utf-8 -*-

import zmx_params as params
import cPickle
from zmx_virtuoso_api import *
from time import sleep
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

subject_without_name_yet = []
subject_name_dict = {}
with open(params.qa_path) as f:
	for line in f:
		str_list = line.split()
		# print str_list
		sub = str_list[0].replace("www.freebase.com/", "").replace("/", ".").strip()
		# print sub, "\t", obj, "\n\n"
		sub_name = sparql_query(sub)
		# sleep(1)
		if sub_name == "":
			subject_without_name_yet.append(sub)
		else:
			# print sub_name
			subject_name_dict[sub] = sub_name

with open(params.train_set_subject_name_dict_pickle_path_virtuoso_7, "w") as f:
	cPickle.dump(subject_name_dict, f)

with open(params.train_set_subject_without_name_yet_list_pickle_path_virtuoso_7, "w") as f:
	cPickle.dump(subject_without_name_yet, f)


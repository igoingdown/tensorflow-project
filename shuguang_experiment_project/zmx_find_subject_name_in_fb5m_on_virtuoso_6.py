#!/usr/bin/python
# -*- coding:utf-8 -*-

import zmx_params as params
import traceback
import cPickle
from zmx_virtuoso_api_for_virtuoso_6 import *
from time import sleep
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

subject_without_name_yet = []
subject_name_dict = {}
last_sub = ""
counter = 0
with open(params.fact_path) as f:
	for line in f:
		print counter
		counter += 1
		str_list = line.split()
		# print str_list
		sub = str_list[0].replace("www.freebase.com/", "").replace("/", ".").strip()
		if last_sub == sub:
			continue
		last_sub = sub

		try:
			sub_name = sparql_query(sub)
			# sleep(1)
			if sub_name == "":
				subject_without_name_yet.append(sub)
			else:
				# print sub_name
				subject_name_dict[sub] = sub_name
		except Exception as e:
			traceback.print_exc()
			print "\n", "sub:\t", sub
			break

with open(params.fb5m_subject_name_dict_pickle_path, "w") as f:
	cPickle.dump(subject_name_dict, f)

with open(params.fb5m_subject_without_name_yet_list_pickle_path, "w") as f:
	cPickle.dump(subject_without_name_yet, f)



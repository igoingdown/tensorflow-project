#!/usr/bin/python
# -*- coding:utf-8 -*-

import zmx_params as params
from time_wrapper import *
from zmx_virtuoso_api_for_virtuoso_6 import *
import cPickle
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


def load_pickle(name):
	with open(name) as f:
		list_or_dict = cPickle.load(f)
	return list_or_dict


if __name__ == '__main__':
	nameless_list = load_pickle(params.all_merge_list_pickle_path)
	named_dict = load_pickle(params.all_merge_dict_pickle_path)

	new_nameless_list = []
	counter = 0
	for i in nameless_list:
		name = sparql_query_name(i)
		if name != "":
			named_dict[i] = name
		else:
			alias = sparql_query_alias(i)
			en_alias = sparql_query_en_alias(i)
			en_name = sparql_query_en_name(i)
			if alias != "":
				print "the alias of {0} is : {1}".format(i, alias)
			if en_alias != "":
				print "the en_alias of {0} is: {1}".format(i, en_alias)
			if en_name != "":
				print "the en_name of {0} is {1}".format(i, en_name)

			if en_name != "" or en_alias != "" or alias != "":
				counter += 1
			else:
				new_nameless_list.append(i)

			print "\n\n\n"

	with open(params.all_merge_dict_pickle_path, "w") as f:
		cPickle.dump(named_dict, f)
	with open(params.all_merge_list_pickle_path, "w") as f:
		cPickle.dump(new_nameless_list, f)

	print "{0} subjects name not found while alias or en_name or en_alias found!".format(counter)
	print "{0} subjects name found!".format(len(named_dict))
	print "{0} subjects name not found yet!".format(len(new_nameless_list))













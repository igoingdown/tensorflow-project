import sys
import zmx_params as params
from time_wrapper import *
from zmx_virtuoso_api import *
import cPickle
reload(sys)
sys.setdefaultencoding('utf-8')


def load_pickle(name):
	with open(name) as f:
		list_or_dict = cPickle.load(f)
	return list_or_dict


if __name__ == '__main__':
	test_nameless_list = load_pickle(params.merged_list_of_test_set_pickle_path)
	counter = 0
	for i in test_nameless_list:
		alias = sparql_query_alias(i)
		en_alias = sparql_query_en_alias(i)
		en_name = sparql_query_en_name(i)
		if alias != "":
			print "the alias of {0} is : {1}".format(i, alias)
		if en_alias != "":
			print "the en_alias of {0} is: {1}".format(i, en_alias)
		if en_name != "":
			print "the en_name of {0} is {1}".format(i, en_name)
		if en_name != "" or en_alias != "" or alias != "" :
			counter += 1
	print "{0}/{1} nameless subject found in test set!".format(counter, len(test_nameless_list))














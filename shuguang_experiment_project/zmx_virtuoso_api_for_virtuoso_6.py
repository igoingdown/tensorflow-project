#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
===============================================================================
author: 赵明星
desc:   专门为virtuoso 6设计的实体name， alias等字段的查询解析API。
===============================================================================
"""

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import requests
import re
import zmx_params as params
import traceback
import cPickle

# Setting global variables

# 双老师组服务器上的virtuoso-6
# query_url = 'http://10.108.113.134:8890/sparql/'

# deeplearning服务器上的virtuoso-6
query_url = 'http://10.108.112.28:8890/sparql/'

PREFIX =  "<http://rdf.freebase.com/ns/>"

# HTTP URL is constructed accordingly with JSON query results format in mind.
def sparql_query_name(subject_id):
	query = '''
		PREFIX ns:%s
		SELECT ?name WHERE {ns:%s ns:type.object.name ?name}
	'''% (PREFIX, subject_id)
	# print query
	params = {'default-graph': '', 
			'query': query.encode('utf8'), 
			'format': "text/html"}

	r = requests.post(query_url, data=params)
	http_result = r.text
	# print http_result
	# print "http_result:\n", http_result, "\n\n"
	# name_pattern = re.compile("<td>(.*?)@en</td>", re.S)
	name_pattern = re.compile("<td>(.*?)@(.*?)</td>", re.S)
	alias_language_list = name_pattern.findall(http_result)
	# print alias_list

	try:
		alias_list = [x[0] for x in alias_language_list]
		lang_list = [x[1] for x in alias_language_list]
		# print "lang_list: ", lang_list
		alias = alias_list[lang_list.index(u"en")]
		return alias
	except Exception as e:
		print "subject_id:\t", 
			  subject_id, 
			  "\nhttp_result:\n", 
			  http_result, 
			  "\n\n\n"
		return ""


def sparql_query_en_name(subject_id):
	query = '''
		PREFIX ns:%s
		SELECT ?name WHERE {ns:%s ns:type.object.en_name ?name}
	'''% (PREFIX, subject_id)
	# print query
	params = {'default-graph': '', 
			  'query': query.encode('utf8'), 
			  'format': "text/html"}

	r = requests.post(query_url, data=params)
	http_result = r.text
	# print http_result
	# print "http_result:\n", http_result, "\n\n"
	# name_pattern = re.compile("<td>(.*?)@en</td>", re.S)
	name_pattern = re.compile("<td>(.*?)</td>", re.S)
	alias_language_list = name_pattern.findall(http_result)
	# print alias_list

	try:
		return alias_language_list[0]
	except Exception as e:
		print "subject_id:\t", 
			  subject_id, 
			  "\nhttp_result:\n", 
			  http_result, 
			  "\n\n\n"
		return ""


def sparql_query_alias(subject_id):
	query = '''
		PREFIX ns:%s
		SELECT ?name WHERE {ns:%s ns:common.topic.alias ?name}
	'''% (PREFIX, subject_id)
	# print query
	params = {'default-graph': '', 
			  'query': query.encode('utf8'), 
			  'format': "text/html"}

	r = requests.post(query_url, data=params)
	http_result = r.text
	# print http_result
	# print "http_result:\n", http_result, "\n\n"
	# name_pattern = re.compile("<td>(.*?)@en</td>", re.S)
	name_pattern = re.compile("<td>(.*?)</td>", re.S)
	alias_language_list = name_pattern.findall(http_result)
	# print alias_list

	try:
		return alias_language_list[0]
	except Exception as e:
		print "subject_id:\t", 
			  subject_id, 
			  "\nhttp_result:\n", 
			  http_result, 
			  "\n\n\n"
		return ""

def sparql_query_en_alias(subject_id):
	query = '''
		PREFIX ns:%s
		SELECT ?name WHERE {ns:%s ns:common.topic.en_alias ?name}
	'''% (PREFIX, subject_id)
	# print query
	params = {'default-graph': '', 
			  'query': query.encode('utf8'), 
			  'format': "text/html"}

	r = requests.post(query_url, data=params)
	http_result = r.text
	# print http_result
	# print "http_result:\n", http_result, "\n\n"
	# name_pattern = re.compile("<td>(.*?)@en</td>", re.S)
	name_pattern = re.compile("<td>(.*?)</td>", re.S)
	alias_language_list = name_pattern.findall(http_result)
	# print alias_list

	try:
		return alias_language_list[0]
	except Exception as e:
		print "subject_id:\t", 
			  subject_id, 
			  "\nhttp_result:\n", 
			  http_result, 
			  "\n\n\n"
		return ""


if __name__ == '__main__':
	with open(params.test_alias_dict_pickle_path) as f:
		alias_dict = cPickle.load(f)
	# sparql_query("m/01vsll")
	for k, v in alias_dict.iteritems():
		# print k, "\n", v, "\n" * 3
		# print k.strip("/").replace("/", ".") 
		print sparql_query(k.strip("/").replace("/", "."))
	# sparql_query("m.01_d5")







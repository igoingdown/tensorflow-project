#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
	@auth	赵明星 
	@date	2016.12.26
	@desc	计算CRF模型的准确率precision和召回率recall
"""

import zmx_params as params
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


if __name__ == '__main__':
	with open(params.ls_crf_predict_data_path) as f:
		lines = f.readlines()

	prediction_pos_matrix = []
	for i in xrange(len(lines) / 2):
		pos_list = []
		score_list = lines[2 * i + 1].strip().split()
		word_list = lines[2 * i].strip().split()
		for i in range(len(score_list)):
			if score_list[i] == '1.001':
				pos_list.append(i)
		prediction_pos_matrix.append(pos_list)

	real_data_subject_name_list = []
	prediction_subject_name_list = []
	line_counter = 0
	with open(params.ls_crf_real_data_path) as f:
		for line in f:
			pos_list = []
			word_string, num_string = line.split("\t")
			words = word_string.split()
			nums = num_string.split()
			for n in nums:
				try:
					pos_list.append(int(n))
				except Exception as e:
					raise e
			real_data_subject_name_list.append(" ".join(words[x] for x in pos_list))
			prediction_subject_name_list.append(" ".join(words[x] for x in prediction_pos_matrix[line_counter]))
			line_counter += 1

	counter = 0
	for i in xrange(len(prediction_pos_matrix)):
		if real_data_subject_name_list[i] != prediction_subject_name_list[i]:
			print "predict: ", prediction_subject_name_list[i]
			print "real: ", real_data_subject_name_list[i]
			print "~" * 100
			counter += 1
	print counter
	print len(real_data_subject_name_list) - counter































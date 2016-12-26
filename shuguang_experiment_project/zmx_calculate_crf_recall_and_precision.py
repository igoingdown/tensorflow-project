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
	prediction_line_counter = 0
	prediction_score_matrix = []
	prediction_subject_name_list = []
	with open(params.ls_crf_predict_data_path) as f:
		for line in f:
			score_list = []
			prediction_line_counter += 1
			words_or_nums = line.split()
			try:
				if prediction_line_counter % 2 == 0:
					for num in words_or_nums:
						score_list.append(float(num))
			except Exception as e:
				pass
			prediction_score_matrix.append(score_list)
			pos_list = []
			for i in range(len(score_list)):
				if score_list[i] == 1.001:
					# 记录index
					pos_list.append(i)
			prediction_subject_name_list.append(" ".join(words_or_nums[x] for x in pos_list))

	real_data_pos_matrix = []
	real_data_line_counter = 0
	real_data_subject_name_list = []
	with open(params.ls_crf_real_data_path) as f:
		for line in f:
			pos_list = []
			real_data_line_counter += 1
			words = line.split()
			for word in words:
				try:
					pos_list.append(int(word))
				except Exception as e:
					continue
			real_data_subject_name_list.append(" ".join(words[x] for x in pos_list))
			real_data_pos_matrix.append(pos_list)

	# print "predict data set subject pos matrix length: {0}.".format(len(prediction_score_matrix))
	# print "real data set subject pos matrix length: {0}".format(len(real_data_pos_matrix))
	# print prediction_score_matrix
	prediction_pos_matrix = []
	for line in prediction_score_matrix:
		pos_list = []
		for i in range(len(line)):
			if line[i] == 1.001:
				# 记录index
				pos_list.append(i)
		prediction_pos_matrix.append(pos_list)

	for i in xrange(len(prediction_pos_matrix)):
		print "prediction pos:", prediction_pos_matrix[i]
		print "real pos:", real_data_pos_matrix[i]
		print prediction_subject_name_list[i]
		print real_data_subject_name_list[i]
		print "\n\n"































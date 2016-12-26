# -*-coding:utf-8 -*-

import sys
import zmx_params as params
from time_wrapper import *
import cPickle
reload(sys)
sys.setdefaultencoding('utf-8')


@time_recorder
def read_question_and_answers():
	answers = []
	questions = []

	with open(params.qa_test_path) as f:
		for line in f:
			candidates = []
			content = line.strip().split('\t')
			subject = content[0].replace("www.freebase.com","")
			# print "subject:", subject
			relation = content[1].replace("www.freebase.com","")
			# print "relation", relation
			answers.append((subject, relation))
			# relation在kb_phrase中全大写很重要，但要确定一个规则，这样可以将relation和其他文本中相同的区分开来，使relation中的词汇表达更精确。
			# 更能体现relation的特殊语义
			question = content[3]
			questions.append(question)

	return questions, answers


@time_recorder
def save_as_file(questions, answers):
	with open(params.test_answer_pickle_path, 'wb') as f:
		cPickle.dump(answers, f)
	with open(params.test_question_pickle_path, 'wb') as f:
		cPickle.dump(questions, f)


if __name__ == '__main__':
	questions, answers = read_question_and_answers()
	save_as_file(questions, answers)










# -*-coding:utf-8 -*-

import sys
import params
import time
import cPickle
import re
reload(sys)
sys.setdefaultencoding('utf-8')


def load_dicts_from_file(self):
	with open(params.test_pickle_path, 'rb') as f:
		relation_sujects_dict = cPickle.load(f)
		word_subjects_dict = cPickle.load(f)
		subject_relations_dict = cPickle.load(f)

	with open(params.pickle_path, 'rb') as f:
		train_questions = cPickle.load(f)
		train_answers = cPickle.load(f)
		train_triples = cPickle.load(f)
		nl_word_dict_freq = cPickle.load(f)
		nl_phrase_dict_freq = cPickle.load(f)
		kb_word_dict_freq = cPickle.load(f)
		kb_phrase_dict_freq = cPickle.load(f)
		nl_character2word_dict = cPickle.load(f)
		nl_word2phrase_dict = cPickle.load(f)
		kb_character2word_dict = cPickle.load(f)
		kb_word2phrase_dict = cPickle.load(f)
		nl_phrase_dict = cPickle.load(f)
		# rerun input_data.py, store nl_phrase_dict into the file!
	return relation_sujects_dict, word_subjects_dict, subject_relations_dict, nl_phrase_dict


def split_into_words(content):
	return filter(lambda x: not re.match('\s',x) and x != '', 
		re.split("([`|~|\!|@|#|\$|%|\^|&|\*|\(|\)|\-|_|\=|\+|\[|\{|\]|\}|\||\\|;|\|:|'|\"|\,|\<|\.|\>|\/|\?|\s])", content))


def get_candidate_subject_relation_pairs_and_answers(relation_sujects_dict, word_subjects_dict, subject_relations_dict):
	subject_relation_pairs = []
	answers = []
	with open(params.qa_path) as f:
		for line in f:
			candidates = []
			content = line.strip().split('\t')
			subject = content[0].replace("www.freebase.com","")
			# print "subject:", subject
			relation = content[1].replace("www.freebase.com","")
			# print "relation", relation
			answers.append(turple(subject, relation))
			# 其实relation在kb_phrase中全大写还是保持原始状态不重要，但要确定一个规则，此处保留原始状态
			question = content[3]
			question_words = split_into_words(question.lower())
			for word in question_words:
				candidate_subjects = word_subjects_dict[word]
				for subject in candidate_subjects:
					candidate_relations = subject_relations_dict[subject]
					for r in candidate_relations:
						candidates.append(turple(subject, r))
			subject_relation_pairs.append(candidates)
	return subject_relation_pairs, answers


def save_as_file(subject_relation_pairs, answers):
	with open(params.candidate_and_answer_pickle_path, 'wb') as f:
		cPickle.dump(subject_relation_pairs, f)
		cPickle.dump(answers, f)


if __name__ == '__main__':
	relation_sujects_dict, word_subjects_dict, subject_relations_dict = load_dicts_from_file()
	subject_relation_pairs, answers = get_candidate_subject_relation_pairs_and_answers(relation_sujects_dict, word_subjects_dict, subject_relations_dict)
	save_as_file(subject_relation_pairs, answers)



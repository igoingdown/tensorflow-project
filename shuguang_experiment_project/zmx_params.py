#!/usr/bin/python
# -*- coding:utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

shuguang_data_path = '/home/deeplearning/zsg/data/'
simplequestion_dataset_path = '/home/deeplearning/zsg/data/simplequestion_dataset'

fb2m_type_path = shuguang_data_path + 'Freebase/FB2M_type.csv'
fb5m_2m_type_path = shuguang_data_path + 'Freebase/FB5M_2M_type.csv'
phrase_dict_path = '/home/deeplearning/zsg/code/Integration/DictionaryProcessing/dict/nl_phrase_dict.txt'

qa_path = simplequestion_dataset_path + 'annotated_fb_data_train.txt'
train_set_subject_name_dict_pickle_path = simplequestion_dataset_path + 'train_set_subject_name_dict_pickle'
train_set_subject_without_name_yet_list_pickle_path = simplequestion_dataset_path + 'train_set_subject_without_name_yet_list_pickle'
train_set_subject_name_dict_pickle_path_virtuoso_7 = simplequestion_dataset_path + 'train_set_subject_name_dict_pickle_virtuoso_7'
train_set_subject_without_name_yet_list_pickle_path_virtuoso_7 = simplequestion_dataset_path + 'train_set_subject_without_name_yet_list_pickle_virtuoso_7'
merged_dict_of_train_set_pickle_path = simplequestion_dataset_path + 'merged_dict_of_train_set'
merged_list_of_train_set_pickle_path = simplequestion_dataset_path + 'merged_list_of_train_set'

qa_test_path = simplequestion_dataset_path + 'annotated_fb_data_test.txt'
test_set_subject_name_dict_pickle_path = simplequestion_dataset_path + 'test_set_subject_name_dict_pickle'
test_set_subject_without_name_yet_list_pickle_path = simplequestion_dataset_path + 'test_set_subject_without_name_yet_list_pickle'
test_set_subject_name_dict_pickle_path_virtuoso_7 = simplequestion_dataset_path + 'test_set_subject_name_dict_pickle_virtuoso_7'
test_set_subject_without_name_yet_list_pickle_path_virtuoso_7 = simplequestion_dataset_path + 'test_set_subject_without_name_yet_list_pickle_virtuoso_7'
merged_dict_of_test_set_pickle_path = simplequestion_dataset_path + 'merged_dict_of_test_set'
merged_list_of_test_set_pickle_path = simplequestion_dataset_path + 'merged_list_of_test_set'

qa_validation_path = simplequestion_dataset_path + 'annotated_fb_data_valid.txt'
valid_set_subject_name_dict_pickle_path = simplequestion_dataset_path + 'valid_set_subject_name_dict_pickle'
valid_set_subject_without_name_yet_list_pickle_path = simplequestion_dataset_path + 'valid_set_subject_without_name_yet_list_pickle'
valid_set_subject_name_dict_pickle_path_virtuoso_7 = simplequestion_dataset_path + 'valid_set_subject_name_dict_pickle_virtuoso_7'
valid_set_subject_without_name_yet_list_pickle_path_virtuoso_7 = simplequestion_dataset_path + 'valid_set_subject_without_name_yet_list_pickle_virtuoso_7'
merged_dict_of_valid_set_pickle_path = simplequestion_dataset_path + 'merged_dict_of_valid_set'
merged_list_of_valid_set_pickle_path = simplequestion_dataset_path + 'merged_list_of_valid_set'

fact_path = shuguang_data_path + 'freebase-FB5M.txt'
fb5m_subject_name_dict_pickle_path = shuguang_data_path + 'fb5m_subject_name_dict_pickle'
fb5m_subject_without_name_yet_list_pickle_path = shuguang_data_path + 'fb5m_subject_without_name_yet_list_pickle'
fb5m_subject_name_dict_pickle_path_virtuoso_7 = shuguang_data_path + 'fb5m_subject_name_dict_pickle_virtuoso_7'
fb5m_subject_without_name_yet_list_pickle_path_virtuoso_7 = shuguang_data_path + 'fb5m_subject_without_name_yet_list_pickle_virtuoso_7'
merged_dict_of_fb5m_pickle_path = simplequestion_dataset_path + 'merged_dict_of_fb5m'
merged_list_of_fb5m_pickle_path = simplequestion_dataset_path + 'merged_list_of_fb5m'

all_merge_dict_pickle_path = simplequestion_dataset_path + 'all_merge_dict'
all_merge_list_pickle_path = simplequestion_dataset_path + 'all_merge_list'

# crf files
ls_crf_real_data_path = shuguang_data_path + 'test_data_focused_labeling_file'
ls_crf_predict_data_path = shuguang_data_path + 'test_data_CRF_predict_res_file'

pickle_path = shuguang_data_path + 'pickle_file2'

# train pickle path:
train_questions_pickle_path = shuguang_data_path + 'train_question_pickle_file2'
train_answers_pickle_path = shuguang_data_path + 'train_answers_pickle_file2'
train_triples_pickle_path = shuguang_data_path + 'train_triples_pickle_file2'
nl_word_dict_freq_pickle_path = shuguang_data_path + 'nl_word_dict_freq_pickle_file2'
nl_phrase_dict_freq_pickle_path = shuguang_data_path + 'nl_phrase_dict_freq_pickle_file2'
kb_word_dict_freq_pickle_path = shuguang_data_path + 'kb_word_dict_freq_pickle_file2'
kb_phrase_dict_freq_pickle_path = shuguang_data_path + 'kb_phrase_dict_freq_pickle_file2'
nl_character2word_dict_pickle_path = shuguang_data_path + 'nl_character2word_dict_pickle_file2'
nl_word2phrase_dict_pickle_path = shuguang_data_path + 'nl_word2phrase_dict_pickle_file2'
kb_character2word_dict_pickle_path = shuguang_data_path + 'kb_character2word_dict_pickle_file2'
kb_word2phrase_dict_pickle_path = shuguang_data_path + 'kb_word2phrase_dict_pickle_file2'
nl_phrase_index_dict_pickle_path = shuguang_data_path + 'nl_phrase_index_dict_pickle_file2'
nl_word_index_dict_pickle_path = shuguang_data_path + 'nl_word_index_dict_pickle_file2'
nl_character_index_dict_pickle_path = shuguang_data_path + 'nl_character_index_dict_pickle_file2'
kb_phrase_index_dict_pickle_path = shuguang_data_path + 'kb_phrase_index_dict_pickle_file2'
kb_word_index_dict_pickle_path = shuguang_data_path + 'kb_word_index_dict_pickle_file2'
kb_character_index_dict_pickle_path = shuguang_data_path + 'kb_character_index_dict_pickle_file2'
nl_word4phrase_dict_pickle_path = shuguang_data_path + 'nl_word4phrase_dict_pickle_file'

# test pickle path
test_subject_relations_dict_pickle_path = shuguang_data_path + 'test_subject_relations_dict_pickle_file'
test_relation_sujects_dict_pickle_path = shuguang_data_path + 'test_relation_sujects_dict_pickle_file'
test_word_subjects_dict_pickle_path = shuguang_data_path + 'test_word_subjects_dict_pickle_file'
test_question_pickle_path = shuguang_data_path + 'test_question_pickle_file'
test_answer_pickle_path = shuguang_data_path + 'test_answer_pickle_file'
test_candidate_pickle_path = shuguang_data_path + 'test_candidates/test_candidate_pickle_file'
test_alias_dict_pickle_path = shuguang_data_path + 'test_alias_dict_pickle_file'
test_candicate_subjects_for_all_questions_pickle_path = shuguang_data_path + 'test_candidates/test_candicate_subjects_for_all_questions_pickle_file'
test_candidate_subjects_for_all_questions_pickle_path_debug = shuguang_data_path + 'test_candidates/test_candicate_subjects_for_all_questions_pickle_file_debug'
test_candidate_subjects_for_all_questions_pickle_path_multiprocessing = shuguang_data_path + 'test_candidates/test_candicate_subjects_for_all_questions_pickle_file_multiprocessing'

# txt path, store human readable file to check if there are some exceptions in the dict
test_word_subjects_dict_txt_path = shuguang_data_path + 'test_word_subjects_dict_txt_file'


# tensorflow graph parameters:
sf_batch_size, qa_batch_size = (10, 10)
sf_neg_size, qa_neg_size = (10, 10)
epoch_num = 3
sf_nl_character_max_length, sf_nl_word_max_length, sf_nl_phrase_max_length = (300, 100, 100)
sf_kb_character_max_length, sf_kb_word_max_length, sf_kb_phrase_max_length = (300, 50, 50)
qa_nl_character_max_length, qa_nl_word_max_length, qa_nl_phrase_max_length = (100, 20, 20)
qa_kb_character_max_length, qa_kb_word_max_length, qa_kb_phrase_max_length = (100, 20, 2)
character_hidden_size, word_hidden_size, phrase_hidden_size = (128, 128, 128)
word_max_length, phrase_max_length = (30, 10)
freq_limit = 3
gru_hidden_size = 128
gamma_sf,gamma_qa = (0.1, 0.1)


if __name__ == '__main__':
    pass

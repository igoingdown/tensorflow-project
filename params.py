fb2m_type_path = '/home/deeplearning/zsg/data/Freebase/FB2M_type.csv'
fb5m_2m_type_path = '/home/deeplearning/zsg/data/Freebase/FB5M_2M_type.csv'
phrase_dict_path = '/home/deeplearning/zsg/code/Integration/DictionaryProcessing/dict/nl_phrase_dict.txt'
qa_path = '/home/deeplearning/zsg/data/simplequestion_dataset/annotated_fb_data_train.txt'
fact_path = '/home/deeplearning/zsg/data/freebase-FB5M.txt'
pickle_path = '/home/deeplearning/zsg/data/pickle_file2'
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
gamma_sf, gamma_qa = (0.1, 0.1)

if __name__ == '__main__':
    pass

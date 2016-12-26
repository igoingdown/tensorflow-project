#coding=utf-8
import re
import zmx_params as params
import time
import cPickle

# freq统计每个单元出现次数，方便按频率排序
nl_character_freq = dict()
nl_word_freq = dict()
nl_phrase_freq = dict()
kb_character_freq = dict()
kb_word_freq = dict()
kb_phrase_freq = dict()

"""
	以下的6个dict为按照出现次数排序后编号的词典，即“单词-编号”,
	key is character/word/phrase, value is index
	these dicts should be stored into pickle file
	because when change natural language subject/relation/question into indice
	these dicts are necessary!
"""
nl_character_dict = dict()
nl_word_dict = dict()
nl_phrase_dict = dict()
kb_character_dict = dict()
kb_word_dict = dict()
kb_phrase_dict = dict()

# 将编号和出现次数连接，因为after_raw_data根据出现次数判断到哪个序号开始OOV
nl_character_dict_freq = dict()
nl_word_dict_freq = dict()
nl_phrase_dict_freq = dict()
kb_character_dict_freq = dict()
kb_word_dict_freq = dict()
kb_phrase_dict_freq = dict()

# 实体的字母和词级别信息，需要由爬取得到的alias信息获得
alias_dict = dict()

# 将phrase切割成word，方便进行phrase的识别
nl_word4phrase_dict = dict()

# 训练样例，包含statement-fact,question-answer和sentences,triples(为了预训练)
train_sentences = list()
train_triples = list()
train_questions = list()
train_answers = list()
train_statements = list()
train_facts = list()

"""
	以下是映射表
	these dicts should also be stored into pickle file
	because when change natural language subject/relation/question into indice
	these dicts are necessary!
"""
nl_character2word_dict = dict()
nl_word2phrase_dict = dict()
kb_character2word_dict = dict()
kb_word2phrase_dict = dict()


def split_into_words(content):
	return filter(lambda x: not re.match('\s',x) and x != '', 
			re.split("([`|~|\!|@|#|\$|%|\^|&|\*|\(|\)|\-|_|\=|\+|\[|\{|\]|\}|\||\\|;|\|:|'|\"|\,|\<|\.|\>|\/|\?|\s])", content))


def get_alias_dict():
	with open(params.fb2m_type_path) as file:
		while 1:
			line = file.readline()
			if not line:
				break
			content = line.strip().split('\t')
			mid = content[0]
			alias = content[1].lower()
			if not alias_dict.has_key(mid):
				alias_dict[mid] = alias
	with open(params.fb5m_2m_type_path) as file:
		while 1:
			line = file.readline()
			if not line:
				break
			content = line.strip().split('\t')
			mid = content[0]
			alias = content[1].lower()
			if not alias_dict.has_key(mid):
				alias_dict[mid] = alias


def get_phrase_dict():
	with open(params.phrase_dict_path) as file:
		while 1:
			line = file.readline()
			if not line:
				break
			content = line.strip().split('\t')
			phrase = content[0].lower()
			if not nl_phrase_freq.has_key(phrase):
				nl_phrase_freq[phrase] = 0


def get_word4phrase_dict():
	for k, _ in nl_phrase_freq.iteritems():
		nl_word4phrase_dict[tuple(split_into_words(k))] = k


def identify_phrases(word4phrase, words, wordisphrase):
	words_tuple = tuple(words)
	phrases = []
	length = len(words)
	last_pos = 0
	while last_pos < length:
		for pos in range(length, last_pos-1,-1):
			seg = words_tuple[last_pos:pos]
			rescan = 0
			if len(seg) == 1:
				word = seg[0]
				rescan = 1
			elif word4phrase.has_key(seg):
				phrase = word4phrase[seg]
				rescan = 2
			else:
				pass
			if rescan == 0:
				continue
			else:
				last_pos = pos
				if rescan == 1:
					if wordisphrase:
						phrases.append(word)
				if rescan == 2:
					phrases.append(phrase)
				break
	return phrases


def merge_into_dict(sdict, tdict):
	for k, v in sdict.iteritems():
		if not tdict.has_key(k):
			tdict[k] = v


def index_by_freq(sdict, tdict):
	sorted_list = sorted(sdict.iteritems(), key=lambda d:d[1], reverse=True)
	index = 3
	for pair in sorted_list:
		tdict[pair[0]] = index
		index += 1


def map_dict_and_freq(dict1, dict2, tdict):
	for k, v in dict1.iteritems():
		tdict[v] = dict2[k]


def del_value_0_in_dict(idict, jdict):
	delary = []
	for k, v in idict.iteritems():
		if v == 0:
			delary.append(k)
	for x in delary:
		del idict[x]
		del jdict[x]


def create_unindex_dict():
	start_time = time.time()
	print "create_unindex_dict begin......"

#先按第二版方案，后面可能把空格和标点算作character
	nl_character_initialize = {'0','1','2','3','4','5','6','7','8','9','a','b','c','d','e',
			'f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'}
	kb_character_initialize = {'0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g',
			'h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',\
							'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'}
	word_initialize = {'`','~','!','@','#','$','%','^','&','*','(',')','-','_','=','+','[','{',']','}',
			'\\','|',';',':','\'','\"',',','<','.','>','?'}
	for x in nl_character_initialize:
		nl_character_freq[x] = 0
	for x in kb_character_initialize:
		kb_character_freq[x] = 0
	for x in word_initialize:
		nl_word_freq[x] = 0
		kb_word_freq[x] = 0
	get_phrase_dict()
	get_alias_dict()
	for _, entity in alias_dict.iteritems():
		print "*" * 100
		print "entity:", entity
		print "*" * 100, "\n" * 3
		entity_words = split_into_words(entity)
		print "~" * 100
		print entity_words
		print "~" * 100
		for word in entity_words:
			if not kb_word_freq.has_key(word):
				kb_word_freq[word] = 0
			if not nl_word_freq.has_key(word):
				# 向nl_word中添加alias的单词
				nl_word_freq[word] = 0
	for phrase, _ in nl_phrase_freq.iteritems():
		phrase_words = split_into_words(phrase)
		for word in phrase_words:
			if not nl_word_freq.has_key(word):
				# 向nl_word中添加phrase的单词
				nl_word_freq[word] = 0
	# 运行到这里发现alias_dict很大
	
	with open(params.qa_path) as f:
		for line in f:
			content = line.strip().split('\t')
			subject = content[0].replace("www.freebase.com","")
			print "subject:", subject 
			relation = content[1].replace("www.freebase.com","")
			print "relation", relation
			# 其实relation在kb_phrase中全大写还是保持原始状态不重要，但要确定一个规则，此处保留原始状态
			object = content[2].replace("www.freebase.com","")
			question = content[3]
			relation_words = split_into_words(relation.upper())
			question_words = split_into_words(question.lower())
			if not kb_phrase_freq.has_key(subject):
				kb_phrase_freq[subject] = 0
			if not kb_phrase_freq.has_key(relation):
				kb_phrase_freq[relation] = 0
			if not kb_phrase_freq.has_key(object):
				kb_phrase_freq[object] = 0
			for word in relation_words:
				if not kb_word_freq.has_key(word):
					kb_word_freq[word] = 0
				if not nl_word_freq.has_key(word.lower()):
					# 向nl_word中添加relation的单词
					nl_word_freq[word.lower()] = 0
			for word in question_words:
				if not nl_word_freq.has_key(word):
					# 向nl_word中添加question的单词
					nl_word_freq[word] = 0

	'''
	with open(params.sf_path) as f:
		for line in f:
			content = line.strip().split('\t')
			subject = content[0].replace("www.freebase.com","")
			relation = content[1].replace("www.freebase.com","")
			object = content[2].replace("www.freebase.com","")
			statement = content[3]
			relation_words = split_into_words(relation.upper())
			statement_words = split_into_words(statement.lower())
			if not kb_phrase_freq.has_key(subject):
				kb_phrase_freq[subject] = 0
			if not kb_phrase_freq.has_key(relation):
				kb_phrase_freq[relation] = 0
			if not kb_phrase_freq.has_key(object):
				kb_phrase_freq[object] = 0
			for word in relation_words:
				if not kb_word_freq.has_key(word):
					kb_word_freq[word] = 0
				if not nl_word_freq.has_key(word.lower()):
					nl_word_freq[word.lower()] = 0
			for word in statement_words:
				if not nl_word_freq.has_key(word):#向nl_word中添加statement的单词
					nl_word_freq[word] = 0
	with open(params.sentence_path) as f:
		for line in f:
			words = split_into_words(line.lower())
			for word in words:
				if not nl_word_freq.has_key(word):#向nl_word中添加sentence的单词
					nl_word_freq[word] = 0
	'''

	with open(params.fact_path) as f:
		for line in f:
			content = line.strip().split('\t')
			subject = content[0].replace("www.freebase.com","")
			relation = content[1].replace("www.freebase.com","")
			object = content[2].replace("www.freebase.com","")
			relation_words = split_into_words(relation.upper())
			if not kb_phrase_freq.has_key(subject):
				kb_phrase_freq[subject] = 0
			if not kb_phrase_freq.has_key(relation):
				kb_phrase_freq[relation] = 0
			if not kb_phrase_freq.has_key(object):
				kb_phrase_freq[object] = 0
			for word in relation_words:
				if not kb_word_freq.has_key(word):
					kb_word_freq[word] = 0
				if not nl_word_freq.has_key(word.lower()):
					# 向nl_word中添加fact的单词
					nl_word_freq[word.lower()] = 0

	end_time = time.time()
	dur = end_time - start_time
	print "create_unindex_dict over......"
	print "the total time that create_unindex_dict used is {0}".format(dur)


def create_index_dict():
	print "create_index_dict begin......"
	start_time = time.time()

	get_word4phrase_dict()
	with open(params.qa_path) as f:
		for line in f:
			content = line.strip().split('\t')
			subject = content[0].replace("www.freebase.com","")
			relation = content[1].replace("www.freebase.com","")
			object = content[2].replace("www.freebase.com","")
			question = content[3]
			if alias_dict.has_key(subject):
				subject_words = split_into_words(alias_dict[subject].lower())
				for word in subject_words:
					kb_word_freq[word] += 1
			relation_words = split_into_words(relation.upper())
			if alias_dict.has_key(object):
				object_words = split_into_words(alias_dict[object].lower())
				for word in object_words:
					kb_word_freq[word] += 1
			question_words = split_into_words(question.lower())
			question_phrases = identify_phrases(nl_word4phrase_dict, question_words, 0)
			kb_phrase_freq[subject] += 1
			kb_phrase_freq[relation] += 1
			kb_phrase_freq[object] += 1
			for word in relation_words:
				kb_word_freq[word] += 1
			for word in question_words:
				nl_word_freq[word] += 1
			for phrase in question_phrases:
				nl_phrase_freq[phrase] += 1

	with open(params.fact_path) as f:
		fact_cnt = 0
		for line in f:
			fact_cnt += 1
			if fact_cnt >= 10000: #人为设定
				break
			content = line.strip().split('\t')
			subject = content[0].replace("www.freebase.com","")
			relation = content[1].replace("www.freebase.com","")
			object = content[2].replace("www.freebase.com","")
			kb_phrase_freq[subject] += 1
			kb_phrase_freq[relation] += 1
			kb_phrase_freq[object] += 1
			if alias_dict.has_key(subject):
				subject_words = split_into_words(alias_dict[subject].lower())
				for word in subject_words:
					kb_word_freq[word] += 1
			relation_words = split_into_words(relation.upper())
			for word in relation_words:
				kb_word_freq[word] += 1
			if alias_dict.has_key(object):
				object_words = split_into_words(alias_dict[object].lower())
				for word in object_words:
					kb_word_freq[word] += 1
	# 有其他语料后进行补充
	merge_into_dict(nl_word_freq, nl_phrase_freq)
	# 把统计过频率的nl_word_freq加入同样统计过的nl_phrase_freq
	# 开始编号(其实可以在编号这里记录某个频率的开始index)
	# 代码是不是有点儿问题？
	index = 0
	for k, _ in nl_character_freq.iteritems():
		nl_character_dict[k] = index
		index += 1
	index = 0
	for k, _ in kb_character_freq.iteritems():
		kb_character_dict[k] = index
		index += 1
	index_by_freq(nl_word_freq, nl_word_dict)
	index_by_freq(nl_phrase_freq, nl_phrase_dict)
	index_by_freq(kb_word_freq, kb_word_dict)
	index_by_freq(kb_phrase_freq, kb_phrase_dict)
	del_value_0_in_dict(nl_word_freq, nl_word_dict)
	del_value_0_in_dict(nl_phrase_freq, nl_phrase_dict)
	del_value_0_in_dict(kb_word_freq, kb_word_dict)
	del_value_0_in_dict(kb_phrase_freq, kb_phrase_dict)
	del_value_0_in_dict(nl_word_freq, nl_word_dict)

	end_time = time.time()
	dur = end_time - start_time
	print "create_index_dict over......"
	print "the total time that create_index_dict used is {0}".format(dur)


def create_index_map():
	print "create_index_map begin......"
	start_time = time.time()

	for k, v in nl_phrase_dict.iteritems():
		words = split_into_words(k)
		word_indexs = [nl_word_dict[x] for x in words]
		nl_word2phrase_dict[v] = word_indexs
	for k, v in kb_phrase_dict.iteritems():
		if alias_dict.has_key(k):
			entity = alias_dict[k]
			entity_words = split_into_words(entity)
			entity_word_indexs = [kb_word_dict[x] for x in entity_words]
			kb_word2phrase_dict[v] = entity_word_indexs
		elif k.find('/m/') >= 0:
			kb_word2phrase_dict[v] = []
		else:
			relation_words = split_into_words(k)
			relation_word_indexs = []
			for word in relation_words:
				if kb_word_dict.has_key(word):
					relation_word_indexs.append(kb_word_dict[word])
			kb_word2phrase_dict[v] = relation_word_indexs
	for k, v in nl_word_dict.iteritems():
		chars = list(k)
		new_chars = list()
		for char in chars:
			if nl_character_dict.has_key(char):
				new_chars.append(nl_character_dict[char])
		nl_character2word_dict[v] = new_chars
	for k, v in kb_word_dict.iteritems():
		chars = list(k)
		new_chars = list()
		for char in chars:
			if kb_character_dict.has_key(char):
				new_chars.append(kb_character_dict[char])
		kb_character2word_dict[v] = new_chars
	map_dict_and_freq(nl_character_dict, nl_character_freq, nl_character_dict_freq)
	map_dict_and_freq(nl_word_dict, nl_word_freq, nl_word_dict_freq)
	map_dict_and_freq(nl_phrase_dict, nl_phrase_freq, nl_phrase_dict_freq)
	map_dict_and_freq(kb_character_dict, kb_character_freq, kb_character_dict_freq)
	map_dict_and_freq(kb_word_dict, kb_word_freq, kb_word_dict_freq)
	map_dict_and_freq(kb_phrase_dict, kb_phrase_freq, kb_phrase_dict_freq)

	end_time = time.time()
	dur = end_time - start_time
	print "create_index_map over......"
	print "the total time that create_index_map used is {0}".format(dur)


#sentences, triples, question-answer pairs, statement-fact pairs
def create_raw_data():
	print "create_raw_data begin......"
	start_time = time.time()

	with open(params.qa_path) as f:
		for line in f:
			content = line.strip().split('\t')
			subject = content[0].replace("www.freebase.com","")
			relation = content[1].replace("www.freebase.com","")
			object = content[2].replace("www.freebase.com","")
			question = content[3]
			question_words = split_into_words(question.lower())
			question_phrases = identify_phrases(nl_word4phrase_dict, question_words, 1)
			question_phrase_indexs = [nl_phrase_dict[x] for x in question_phrases]
			answer_phrase_indexs = [kb_phrase_dict[subject], kb_phrase_dict[relation]]
			train_questions.append(question_phrase_indexs)
			train_answers.append(answer_phrase_indexs)

	with open(params.fact_path) as f:
		fact_cnt = 0
		for line in f:
			fact_cnt += 1
			if fact_cnt >= 10000: #人为设定
				break
			content = line.strip().split('\t')
			subject = content[0].replace("www.freebase.com","")
			relation = content[1].replace("www.freebase.com","")
			object = content[2].replace("www.freebase.com","")
			fact_phrase_indexs = [kb_phrase_dict[subject], kb_phrase_dict[relation], kb_phrase_dict[object]]
			train_triples.append(fact_phrase_indexs)

	end_time = time.time()
	dur = end_time - start_time
	print "create_raw_data over......"
	print "the total time create_raw_data used is {0}".format(dur)


def save_as_file():
	print "save_as_file begin......"
	start_time = time.time()

	with open(params.train_questions_pickle_path, "wb") as f:
		cPickle.dump(train_questions, f)
	with open(params.train_answers_pickle_path, "wb") as f:
		cPickle.dump(train_answers, f)
	with open(params.train_triples_pickle_path, "wb") as f:
		cPickle.dump(train_triples, f)
	with open(params.nl_word_dict_freq_pickle_path, "wb") as f:
		cPickle.dump(nl_word_dict_freq, f)
	with open(params.nl_phrase_dict_freq_pickle_path, "wb") as f:
		cPickle.dump(nl_phrase_dict_freq, f)
	with open(params.kb_word_dict_freq_pickle_path, "wb") as f:
		cPickle.dump(kb_word_dict_freq, f)
	with open(params.kb_phrase_dict_freq_pickle_path, "wb") as f:
		cPickle.dump(kb_phrase_dict_freq, f)
	with open(params.nl_character2word_dict_pickle_path, "wb") as f:
		cPickle.dump(nl_character2word_dict, f)
	with open(params.nl_word2phrase_dict_pickle_path, "wb") as f:
		cPickle.dump(nl_word2phrase_dict, f)
	with open(params.kb_character2word_dict_pickle_path, "wb") as f:
		cPickle.dump(kb_character2word_dict, f)
	with open(params.kb_word2phrase_dict_pickle_path, "wb") as f:
		cPickle.dump(kb_word2phrase_dict, f)
	with open(params.nl_phrase_index_dict_pickle_path, "wb") as f:
		cPickle.dump(nl_phrase_dict, f)
	with open(params.nl_word_index_dict_pickle_path, "wb") as f:
		cPickle.dump(nl_word_dict, f)
	with open(params.nl_character_index_dict_pickle_path, "wb") as f:
		cPickle.dump(nl_character_dict, f)
	with open(params.kb_phrase_index_dict_pickle_path, "wb") as f:
		cPickle.dump(kb_phrase_dict, f)
	with open(params.kb_word_index_dict_pickle_path, "wb") as f:
		cPickle.dump(kb_word_dict, f)
	with open(params.kb_character_index_dict_pickle_path, "wb") as f:
		cPickle.dump(kb_character_dict, f)
	with open(params.nl_word4phrase_dict_pickle_path, "wb") as f:
		cPickle.dump(nl_word4phrase_dict, f)

	end_time = time.time()
	dur = end_time - start_time
	print "save_as_file over......"
	print "the total time save_as_file used is {0}".format(dur)


if __name__ == '__main__':
	create_unindex_dict()
	create_index_dict()
	create_index_map()
	create_raw_data()
	save_as_file()
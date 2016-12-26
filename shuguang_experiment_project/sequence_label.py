import io
import zmx_params as params
#import traceback
import cPickle
import re
import sys
reload(sys)
sys.setdefaultencoding('utf8')
from nltk import word_tokenize
import zmx_virtuoso_api

def get_dict():
    #with open(params.test_alias_dict_pickle_path) as f:
    with open(params.all_merge_dict_pickle_path) as f:
        alias_dict = cPickle.load(f)
        print "load dict already"
        return alias_dict
def get_alias():
    ques_sub ={}
   # fo_q_s = open('../../new_data/ques_subId.txt','w')
   # fo = open('../../new_data/ques_sub.txt','w')
    with open("../../new_data/annotated_fb_data_train.txt",'r') as f:
        while 1:
            line = f.readline()
            if not line:
                break
            fields = line.strip().split('\t')
            sub = fields[0].split('www.freebase.com/')[-1].replace('/','.')
            question = fields[-1].replace('\\\\','').lower()
            alias = zmx_virtuoso_api.sparql_query(sub)
            if alias != '':
                ques_sub[question] = alias_dict[sub]
               #fo.write(question+'\t'+alias_dict[sub]+'\n')
                #print "write..+1"
           # fo_q_s.write(question+'\t'+sub+'\n')
    #fo.close()
    #fo_q_s.close()
    return ques_sub
def reverse_link(question,ques_sub):
    # get question tokens
    tokens = question.split()
    # init default value of returned variables
    text_subject = None
    text_attention_indices = None

    # query name / alias by node_id (subject)
    #res_list = virtuoso.id_query_str(subject)
    #res_list = zmx_virtuoso_api.sparql_query(subject)
    res = ques_sub[question]                                       
    # sorted by length
    #print res
    pattern = r'(^|\s)(%s)($|\s)' % (re.escape(res))
    #print pattern
    if re.search(pattern, question):
        text_subject = res
        text_attention_indices = get_indices(tokens, res.split())
       
    return text_subject, text_attention_indices

def get_indices(src_list, pattern_list):
    indices = None
    for i in range(len(src_list)):
        match = 1
        for j in range(len(pattern_list)):
            if src_list[i+j] != pattern_list[j]:
                match = 0
                break
        if match:
            indices = range(i, i + len(pattern_list))
            break                                           
    return indices

if __name__ == '__main__':
    ques_label = []
    fo_q_label = open('../../new_data/data.train.focused_labeling','w')
    #alias_dict = get_dict()
    ques_sub = get_alias()
    for key in ques_sub:
        text_subject,text_attention_indices = reverse_link(key,ques_sub)
        if text_subject != None:
            length = len(word_tokenize(key))
            ques_label.append((key,text_attention_indices,length))
           # print length
    ques_label = sorted(ques_label, key = lambda data: data[-1], reverse = True)
    
  # for i in range(len(ques_label)):
    #    print ques_label[i]
    for line in ques_label:
           fo_q_label.write(u'%s\t%s\n' % (line[0], ' '.join([str(index) for index in line[1]])))
   # fo_q_label.close()

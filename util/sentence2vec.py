# -*- coding: utf-8 -*-
import gensim
import pandas as pd
import csv
import jieba
import numpy as np
jieba.add_word('花呗')
jieba.add_word('借呗')

# 将一个句子转换成126维的词向量
def sentence2vec(input_file):
    csv_file = csv.reader(open(input_file, 'r'))
    for item in csv_file:
        for word in item:
            sen = word.split('\t')
            if len(sen) > 2:
                sen1 = sen[0]
                sen2 = sen[1]
                print(sen1.decode('utf-8'))
                print(sen2.decode('utf-8'))


def get_vector_one_sentence(sentence,model):
    #结巴分词
    sen = jieba.cut(sentence)
    #处理sen
    res_vec = np.zeros(256, dtype=np.float32)
    count = 0
    for word in (sen):
        try:
            c = model[word]
        except KeyError:
            #print ('not in vocabulary')
            c = np.zeros(256, dtype=np.float32)
        res_vec = res_vec + c  # 将每一个单词转换成向量model[单词]
        count += 1  # 计算相加元素的个数
    res_vec_list = (res_vec / count).tolist()
    return res_vec_list

# 将两个句子转换成一个216+216的向量
def get_vector_two_sentence(sentence1, sentence2, model):
    sen1 = get_vector_one_sentence(sentence1, model)
    sen2 = get_vector_one_sentence(sentence2, model)
    print(sen1)
    print(sen2)
    return np.array(sen1+sen2, dtype=np.float32)


if __name__ == '__main__':
    '''input_file = '../raw_data/atec_nlp_sim_train.csv'
    sentence2vec(input_file)'''
    #word2vec转换成向量
    model = gensim.models.Word2Vec.load('../models/word2vec_wx')
    sentence1 = "借呗逾期短信通知"
    sentence2 = "如何购买花呗短信通知"
    print(get_vector_two_sentence(sentence1, sentence2, model))
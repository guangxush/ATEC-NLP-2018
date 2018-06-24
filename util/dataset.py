# -*- coding: utf-8 -*-
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
import jieba
#加载句子特征，标签2维
def load_data_with_sentences(filename):
    input_file = open(filename, 'r')
    input_x = []
    input_y = []
    i = 0
    for line in input_file:
        i += 1
        record = line.split(',')
        input_x.append(np.array(record[0:512]))
        if record[-1].strip() == '1':
            input_y.append([1.0, 0.0])
        else:
            input_y.append([0.0, 1.0])
        if i > 39332:
            break
    input_X = np.array(input_x, dtype=np.float32)
    input_Y = np.array(input_y, dtype=np.float32)
    print input_X[0]
    print input_Y[0]
    print input_X.shape
    print input_Y.shape
    print input_X.dtype
    print input_Y.dtype
    return input_X, input_Y


#加载句子特征,标签1维
def load_data_with_sentences_single_flag(filename):
    input_file = open(filename, 'r')
    input_x = []
    input_y = []
    i = 0
    for line in input_file:
        i += 1
        record = line.split(',')
        input_x.append(np.array(record[0:512]))
        if record[-1].strip() == '1':
            input_y.append([1.0])
        else:
            input_y.append([0.0])
        if i > 39332:
            break
    input_X = np.array(input_x, dtype=np.float32)
    input_Y = np.array(input_y, dtype=np.float32)
    print input_X[0]
    print input_Y[0]
    print input_X.shape
    print input_Y.shape
    print input_X.dtype
    print input_Y.dtype
    return input_X, input_Y


#加载特征工程之后的句子向量
def load_data_with_features(filename):
    # 读取输入数据
    input_file = open(filename, 'r')
    input_x = []
    input_y = []
    for line in input_file:
        record = line.split(',')
        input_x.append(record[0:512])
        input_y.append([record[512]])
    input_X = np.array(input_x, dtype=np.float32)
    input_Y = np.array(input_y, dtype=np.float32)
    print input_X[0]
    print input_Y[0]
    print input_X.shape
    print input_Y.shape
    print input_X.dtype
    print input_Y.dtype
    return input_X, input_Y

def create_dictionaries(p_model):
    gensim_dict = Dictionary()
    #p_model.build_vocab('./lstm_vocab')
    gensim_dict.doc2bow(p_model.wv.vocab.keys(), allow_update=True)
    w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 词语的索引，从1开始编号
    w2vec = {word: p_model[word] for word in w2indx.keys()}  # 词语的词向量
    return w2indx, w2vec

def load_all_sentence(filename,dim):
    fin = open(filename,'r')
    if dim == '1':
        result_sentences=[]
        result_labels = []
        for line in fin:
            number,sen1,sen2,label = line.strip().split('\t')
            cur_sen = []
            sen1_list = list(jieba.cut(sen1))
            sen2_list = list(jieba.cut(sen2))
            cur_sen = sen1_list+sen2_list
            result_sentences.append(cur_sen);
            result_labels.append(float(label))
        return result_sentences,result_labels
    elif dim == '2':
        result_sentences1=[]
        result_sentences2=[]
        result_labels = []
        for line in fin:
            number,sen1,sen2,label = line.strip().split('\t')
            sen1_list = list(jieba.cut(sen1))
            sen2_list = list(jieba.cut(sen2))
            result_sentences1.append(sen1_list);
            result_sentences2.append(sen2_list);
            result_labels.append(float(label))
        return result_sentences1,result_sentences2,result_labels
    elif dim == '3':
        result_sentences1=[]
        result_sentences2=[]
        result_labels = []
        for line in fin:
            sen1,sen2,label = line.strip().split('\t')
            sen1_list = list(jieba.cut(sen1))
            sen2_list = list(jieba.cut(sen2))
            result_sentences1.append(sen1_list);
            result_sentences2.append(sen2_list);
            result_labels.append(float(label))
        return result_sentences1,result_sentences2,result_labels

def sentence_to_index_array(p_new_dic, p_sen,dim,number):  # 文本转为索引数字模式
    new_sentences = []
    line = 0
    if dim == '1':
        for sen in p_sen:
            new_sen = []
            for word in sen:
                try:
                    new_sen.append(p_new_dic[word])  # 单词转索引数字
                except:
                    new_sen.append(0)  # 索引字典里没有的词转为数字0
            new_sentences.append(new_sen)
    #print(new_sentences)
        return np.array(new_sentences)
    elif dim == '2':
        for sen in p_sen:
            line = line + 1
            c = 0
            for word in sen:
                c = c + 1
                if c <= int(number):
                    try:
                        new_sentences.append(p_new_dic[word])  # 单词转索引数字
                    except:
                        new_sentences.append(0)  # 索引字典里没有的词转为数字0
                else:
                    break
            if c < int(number): 
                addi = 0
                while(addi < (int(number) - c)):
                    new_sentences.append(0)
                    addi = addi  + 1
            #print('new_sen ' + str(new_sentences))
            #if line > 10:
             #   break
        print('new_sentences'+ str(len(new_sentences)))
        return np.array(new_sentences).reshape(line,int(number))

def sentence_to_index_array_for_test(p_new_dic, p_sen):  # 文本转为索引数字模式
    new_sentences = []
    senlist = list(jieba.cut(p_sen))
    for word in senlist:
        try:
            new_sentences.append(p_new_dic[word])  # 单词转索引数字
        except:
            new_sentences.append(0)  # 索引字典里没有的词转为数字
    return np.array(new_sentences)

def get_balance_data():
    fin = open('../data/inputadd.txt','r')
    fw = open('../data/inputadd_balance.txt','w')
    count0 = 0
    for line in fin:
        number,sen1,sen2,label = line.strip().split('\t')
        if label == '1':
            fw.write(line)
        if label == '0':
            count0 = count0 + 1
            if (count0 < 10000):
                fw.write(line)
if __name__ == '__main__':
    get_balance_data()

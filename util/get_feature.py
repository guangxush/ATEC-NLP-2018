#-*- coding: utf-8 -*-
# coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8') #gb2312
import random
import numpy as np
from tflearn.data_utils import pad_sequences
#from pypinyin import pinyin,lazy_pinyin

from collections import Counter
import os
import pickle
import csv
import jieba


_PAD="_PAD"
PAD_ID = 0
UNK_ID=1
_UNK="UNK"
TRUE_LABEL='1'
splitter="&|&"
def data_mining_features(index,input_string_x1,input_string_x2,vocab_word2index,word_vec_fasttext_dict,word_vec_word2vec_dict,tfidf_dict,n_gram=8):
    """
    get data mining feature given two sentences as string.
    1)n-gram similiarity(blue score);
    2) get length of questions, difference of length
    3) how many words are same, how many words are unique
    4) question 1,2 start with how/why/when
    5 edit distance
    6) cos similiarity using bag of words
    :param input_string_x1:
    :param input_string_x2:
    :return:
    """
    input_string_x1=input_string_x1.decode("utf-8")
    input_string_x2 = input_string_x2.decode("utf-8")
    #1. get blue score vector
    feature_list=[]
    #get blue score with n-gram
    for i in range(n_gram):
        x1_list=split_string_as_list_by_ngram(input_string_x1,i+1)
        x2_list = split_string_as_list_by_ngram(input_string_x2, i + 1)
        blue_score_i_1 = compute_blue_ngram(x1_list,x2_list)
        blue_score_i_2 = compute_blue_ngram(x2_list,x1_list)
        feature_list.append(blue_score_i_1)
        feature_list.append(blue_score_i_2)

    #2. get length of questions, difference of length
    length1=float(len(input_string_x1))
    length2=float(len(input_string_x2))
    length_diff=(float(abs(length1-length2)))/((length1+length2)/2.0)
    feature_list.append(length_diff)

    #3. how many words are same, how many words are unique
    sentence_diff_overlap_features_list=get_sentence_diff_overlap_pert(index,input_string_x1,input_string_x2)
    feature_list.extend(sentence_diff_overlap_features_list)

    #4. question 1,2 start with how/why/when
    #how_why_feature_list=get_special_start_token(input_string_x1,input_string_x2,special_start_token)
    #print("how_why_feature_list:",how_why_feature_list)
    #feature_list.extend(how_why_feature_list)

    #5.edit distance
    edit_distance=float(edit(input_string_x1, input_string_x2))/30.0
    feature_list.append(edit_distance)

    #6.cos distance from sentence embedding
    x1_list=token_string_as_list(input_string_x1, tokenize_style='word')
    x2_list = token_string_as_list(input_string_x2, tokenize_style='word')
    distance_list_fasttext = cos_distance_bag_tfidf(x1_list, x2_list, word_vec_fasttext_dict, tfidf_dict)
    distance_list_word2vec = cos_distance_bag_tfidf(x1_list, x2_list, word_vec_word2vec_dict, tfidf_dict)
    #distance_list2 = cos_distance_bag_tfidf(x1_list, x2_list, word_vec_fasttext_dict, tfidf_dict,tfidf_flag=False)
    #sentence_diffence=np.abs(np.subtract(sentence_vec_1,sentence_vec_2))
    #sentence_multiply=np.multiply(sentence_vec_1,sentence_vec_2)
    feature_list.extend(distance_list_fasttext)
    feature_list.extend(distance_list_word2vec)
    #feature_list.extend(list(sentence_diffence))
    #feature_list.extend(list(sentence_multiply))
    return feature_list


def split_string_as_list_by_ngram(input_string,ngram_value):
    #print("input_string0:",input_string)
    input_string="".join([string for string in input_string if string.strip()])
    #print("input_string1:",input_string)
    length = len(input_string)
    result_string=[]
    for i in range(length):
        if i + ngram_value < length + 1:
            result_string.append(input_string[i:i+ngram_value])
    #print("ngram:",ngram_value,"result_string:",result_string)
    return result_string


def compute_blue_ngram(x1_list,x2_list):
    """
    compute blue score use ngram information. x1_list as predict sentence,x2_list as target sentence
    :param x1_list:
    :param x2_list:
    :return:
    """
    count_dict={}
    count_dict_clip={}
    #1. count for each token at predict sentence side.
    for token in x1_list:
        if token not in count_dict:
            count_dict[token]=1
        else:
            count_dict[token]=count_dict[token]+1
    count=np.sum([value for key,value in count_dict.items()])

    #2.count for tokens existing in predict sentence for target sentence side.
    for token in x2_list:
        if token in count_dict:
            if token not in count_dict_clip:
                count_dict_clip[token]=1
            else:
                count_dict_clip[token]=count_dict_clip[token]+1

    #3. clip value to ceiling value for that token
    count_dict_clip={key:(value if value<=count_dict[key] else count_dict[key]) for key,value in count_dict_clip.items()}
    count_clip=np.sum([value for key,value in count_dict_clip.items()])
    result=float(count_clip)/(float(count)+0.00000001)
    return result



def get_sentence_diff_overlap_pert(index,input_string_x1,input_string_x2):
    #0. get list from string
    input_list1=[input_string_x1[token] for token in range(len(input_string_x1)) if input_string_x1[token].strip()]
    input_list2 = [input_string_x2[token] for token in range(len(input_string_x2)) if input_string_x2[token].strip()]
    length1=len(input_list1)
    length2=len(input_list2)

    num_same=0
    same_word_list=[]
    #1.compute percentage of same tokens
    for word1 in input_list1:
        for word2 in input_list2:
           if word1==word2:
               num_same=num_same+1
               same_word_list.append(word1)
               continue
    num_same_pert_min=float(num_same)/float(max(length1,length2))
    num_same_pert_max = float(num_same) / float(min(length1, length2))
    num_same_pert_avg = float(num_same) / (float(length1+length2)/2.0)

    #2.compute percentage of unique tokens in each string
    input_list1_unique=set([x for x in input_list1 if x not in same_word_list])
    input_list2_unique = set([x for x in input_list2 if x not in same_word_list])
    num_diff_x1=float(len(input_list1_unique))/float(length1)
    num_diff_x2= float(len(input_list2_unique)) / float(length2)

    if index==0:#print debug message
        print("input_string_x1:",input_string_x1)
        print("input_string_x2:",input_string_x2)
        print("same_word_list:",same_word_list)
        print("input_list1_unique:",input_list1_unique)
        print("input_list2_unique:",input_list2_unique)
        print("num_same:",num_same,";length1:",length1,";length2:",length2,";num_same_pert_min:",num_same_pert_min,
              ";num_same_pert_max:",num_same_pert_max,";num_same_pert_avg:",num_same_pert_avg,
             ";num_diff_x1:",num_diff_x1,";num_diff_x2:",num_diff_x2)

    diff_overlap_list=[num_same_pert_min,num_same_pert_max, num_same_pert_avg,num_diff_x1, num_diff_x2]
    return diff_overlap_list



def edit(str1, str2):
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    #print("matrix:",matrix)
    for i in xrange(1, len(str1) + 1):
        for j in xrange(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)

    return matrix[len(str1)][len(str2)]

listt = []
def token_string_as_list(string,tokenize_style='char'):
	string=string.decode("utf-8")
	string=string.replace("***","*")
	length=len(string)
	global listt
	listt = []
	if tokenize_style=='char':
		listt=[string[i] for i in range(length)]
	elif tokenize_style=='word':
		listt=jieba.lcut(string) #cut_all=True
	elif tokenize_style=='pinyin':
		string=" ".join(jieba.lcut(string))
		listt = ''.join(lazy_pinyin(string)).split() #list:['nihao', 'wo', 'de', 'pengyou']
	listt=[x for x in listt if x.strip()]
	return listt
			   


def cos_distance_bag_tfidf(input_string_x1, input_string_x2,word_vec_dict, tfidf_dict,tfidf_flag=True):
    #print("input_string_x1:",input_string_x1)
    #1.1 get word vec for sentence 1
    sentence_vec1=get_sentence_vector(word_vec_dict,tfidf_dict, input_string_x1,tfidf_flag=tfidf_flag)
    #print("sentence_vec1:",sentence_vec1)
    #1.2 get word vec for sentence 2
    sentence_vec2 = get_sentence_vector(word_vec_dict, tfidf_dict, input_string_x2,tfidf_flag=tfidf_flag)
    #print("sentence_vec2:", sentence_vec2)
    #2 compute cos similiarity
    numerator=np.sum(np.multiply(sentence_vec1,sentence_vec2))
    denominator=np.sqrt(np.sum(np.power(sentence_vec1,2)))*np.sqrt(np.sum(np.power(sentence_vec2,2)))
    cos_distance=float(numerator)/float(denominator)

    #print("cos_distance:",cos_distance)
    manhattan_distance=np.sum(np.abs(np.subtract(sentence_vec1,sentence_vec2)))
    #print(manhattan_distance,type(manhattan_distance),np.isnan(manhattan_distance))
    if np.isnan(manhattan_distance): manhattan_distance=300.0
    manhattan_distance=np.log(manhattan_distance+0.000001)/5.0

    canberra_distance=np.sum(np.abs(sentence_vec1-sentence_vec2)/np.abs(sentence_vec1+sentence_vec2))
    if np.isnan(canberra_distance): canberra_distance = 300.0
    canberra_distance=np.log(canberra_distance+0.000001)/5.0

    minkowski_distance=np.power(np.sum(np.power((sentence_vec1-sentence_vec2),3)), 0.33333333)
    if np.isnan(minkowski_distance): minkowski_distance = 300.0
    minkowski_distance=np.log(minkowski_distance+0.000001)/5.0

    euclidean_distance=np.sqrt(np.sum(np.power((sentence_vec1-sentence_vec2),2)))
    if np.isnan(euclidean_distance): euclidean_distance =300.0
    euclidean_distance=np.log(euclidean_distance+0.000001)/5.0

    return cos_distance,manhattan_distance,canberra_distance,minkowski_distance,euclidean_distance


def get_sentence_vector(word_vec_dict,tfidf_dict,word_list,tfidf_flag=True):
    vec_sentence=0.0
    length_vec=len(word_vec_dict[u'花呗'])
    for word in word_list:
        #print("word:",word)
        word_vec=word_vec_dict.get(word,None)
        word_tfidf=tfidf_dict.get(word,None)
        #print("word_vec:",word_vec,";word_tfidf:",word_tfidf)
        if word_vec is None is None or word_tfidf is None:
            continue
        else:
            if tfidf_flag==True:
                vec_sentence+=word_vec*word_tfidf
            else:
                vec_sentence += word_vec * 1.0
    vec_sentence=vec_sentence/(np.sqrt(np.sum(np.power(vec_sentence,2))))
    return vec_sentence


def create_vocabulary(training_data_path,vocab_size,name_scope='cnn',tokenize_style='char'):
    """
    create vocabulary
    :param training_data_path:
    :param vocab_size:
    :param name_scope:
    :return:
    """

    cache_vocabulary_label_pik='cache'+"_"+name_scope # path to save cache
    if not os.path.isdir(cache_vocabulary_label_pik): # create folder if not exists.
        os.makedirs(cache_vocabulary_label_pik)

    # if cache exists. load it; otherwise create it.
    cache_path =cache_vocabulary_label_pik+"/"+'vocab_label.pik'
    print("cache_path:",cache_path,"file_exists:",os.path.exists(cache_path))
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as data_f:
            return pickle.load(data_f)
    else:
        vocabulary_word2index={}
        vocabulary_index2word={}
        vocabulary_word2index[_PAD]=PAD_ID
        vocabulary_index2word[PAD_ID]=_PAD
        vocabulary_word2index[_UNK]=UNK_ID
        vocabulary_index2word[UNK_ID]=_UNK

        vocabulary_label2index={'0':0,'1':1}
        vocabulary_index2label={0:'0',1:'1'}

        #1.load raw data
        csvfile = open(training_data_path, 'r')
        spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')

        #2.loop each line,put to counter
        c_inputs=Counter()
        c_labels=Counter()
        for i,row in enumerate(spamreader):#row:['\ufeff1', '\ufeff怎么更改花呗手机号码', '我的花呗是以前的手机号码，怎么更改成现在的支付宝的号码手机号', '1']
            string_list_1=token_string_as_list(row[1],tokenize_style=tokenize_style)
            string_list_2 = token_string_as_list(row[2],tokenize_style=tokenize_style)
            c_inputs.update(string_list_1)
            c_inputs.update(string_list_2)

        #return most frequency words
        vocab_list=c_inputs.most_common(vocab_size)
        #put those words to dict
        for i,tuplee in enumerate(vocab_list):
            word,_=tuplee
            vocabulary_word2index[word]=i+2
            vocabulary_index2word[i+2]=word

        #save to file system if vocabulary of words not exists(pickle).
        if not os.path.exists(cache_path):
            with open(cache_path, 'ab') as data_f:
                pickle.dump((vocabulary_word2index,vocabulary_index2word,vocabulary_label2index,vocabulary_index2label), data_f)
        #save to file system as file(added. for predict purpose when only few package is supported in test env)
        save_vocab_as_file(vocabulary_word2index,vocabulary_index2label,vocab_list,name_scope=name_scope)
    return vocabulary_word2index,vocabulary_index2word,vocabulary_label2index,vocabulary_index2label

def save_vocab_as_file(vocab_word2index,vocab_index2label,vocab_list,name_scope='cnn'):
    #1.1save vocabulary_word2index
    cache_vocab_label_pik = 'cache' + "_" + name_scope
    vocab_word2index_object=open(cache_vocab_label_pik+'/'+'vocab_word2index.txt',mode='a')
    for word,index in vocab_word2index.items():
        vocab_word2index_object.write(word+splitter+str(index)+"\n")
    vocab_word2index_object.close()

    #1.2 save word and frequent
    word_freq_object=open(cache_vocab_label_pik+'/'+'word_freq.txt',mode='a')
    for tuplee in vocab_list:
        word,count=tuplee
        word_freq_object.write(word+"|||"+str(count)+"\n")
    word_freq_object.close()

    #2.vocabulary_index2label
    vocab_index2label_object = open(cache_vocab_label_pik + '/' + 'vocab_index2label.txt',mode='a')
    for index,label in vocab_index2label.items():
        vocab_index2label_object.write(str(index)+splitter+str(label)+"\n")
    vocab_index2label_object.close()

def load_word_vec(file_path):
    source_object = open(file_path, 'r')
    word_vec_dict={}
    for i,line in enumerate(source_object):
        if i==0 and 'word2vec' in file_path:
            continue
        line=line.strip()
        line_list=line.split()
        word=line_list[0].decode("utf-8")
        vec_list=[float(x) for x in line_list[1:]]
        word_vec_dict[word]=np.array(vec_list)
    #print("word_vec_dict:",word_vec_dict)
    return word_vec_dict

def load_tfidf_dict(file_path):#今后|||11.357012399387852
    source_object = open(file_path, 'r')
    tfidf_dict={}
    for line in source_object:
        word,tfidf_value=line.strip().split('&|&')
        word=word.decode("utf-8")
        tfidf_dict[word]=float(tfidf_value)
    #print("tfidf_dict:",tfidf_dict)
    return tfidf_dict


word_vec_fasttext_dict=load_word_vec('../data/fasttext_fin_model_50.vec') #word embedding from fasttxt
word_vec_word2vec_dict = load_word_vec('../data/word2vec.txt') #word embedding from word2vec
tfidf_dict=load_tfidf_dict('../data/atec_nl_sim_tfidf.txt')
vocabulary_word2index, vocabulary_index2word, vocabulary_label2index, vocabulary_index2label = create_vocabulary('../data/atec_nlp_sim_train.csv',60000,name_scope='',tokenize_style='')


def get_feature(number,sen1,sen2):
	features_vector = data_mining_features(number,sen1,sen2,vocabulary_word2index,word_vec_fasttext_dict,word_vec_word2vec_dict,tfidf_dict, n_gram=8)
	return features_vector

if __name__ == '__main__':
	#word_vec_fasttext_dict=load_word_vec('./fasttext_fin_model_50.vec') #word embedding from fasttxt
	#word_vec_word2vec_dict = load_word_vec('./word2vec.txt') #word embedding from word2vec
	#tfidf_dict=load_tfidf_dict('./atec_nl_sim_tfidf.txt')
	#number2 = 18851
	jieba.add_word("花呗")
	jieba.add_word("借呗")
	fout2 = open('train_input3_ag_zero.txt','w')
	#vocabulary_word2index, vocabulary_index2word, vocabulary_label2index, vocabulary_index2label = create_vocabulary('./atec_nlp_sim_train.csv',60000,name_scope='',tokenize_style='')
	with open('../data/train_data_new_bal_ag.txt', 'r') as fin, open('test_data_new.txt', 'a') as fout:
		for line in fin:
			number,sen1,sen2,label = line.strip().split('\t')
			#features_vector = data_mining_features(count,sen1,sen2,vocabulary_word2index,word_vec_fasttext_dict,word_vec_word2vec_dict,tfidf_dict, n_gram=8)
			#if label == '0':
			listsen1 = jieba.cut(sen1)
			listsen2 = jieba.cut(sen2)
			ret = list(set(listsen1)^set(listsen2))
			#if len(ret) > 0:	
			fout2.write(' '.join(ret) + '\n')
			#number2 = number2 + 1
			#if label == 1:
			#	fout.write(str(features_vector) + '\t' + str([1,0]) + '\n')
			#else:
			#	fout.write(str(features_vector) + '\t' + str([0,1]) + '\n')
            

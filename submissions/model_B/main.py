#-*- coding: utf-8 -*-
# coding:utf-8
import jieba
import sys
reload(sys)
sys.path.append("..")
sys.setdefaultencoding("utf-8")
import math
import gensim
import keras
import sim_lstm
from keras.models import load_model
import numpy as np
import tensorflow as tf
from keras.preprocessing import sequence
from tensorflow.python import pywrap_tensorflow
from gensim.models.word2vec import Word2Vec
from dataset import create_dictionariesfromword,sentence_to_index_array_for_test,load_data_with_sentences_single_flag, sentence_to_index_array,load_data_with_features, load_data_with_sentences,create_dictionaries,load_all_sentence

def cos_dist(a, b):
	if len(a) != len(b):
		return None
	part_up = 0.0
	a_sq = 0.0
	b_sq = 0.0
	for a1, b1 in zip(a,b):
		part_up += a1*b1
		a_sq += a1**2
		b_sq += b1**2
	part_down = math.sqrt(a_sq*b_sq)
	if part_down == 0.0:
		return None
	else:
		return part_up / part_down


def process(inpath, outpath):
	#word_vec_fasttext_dict=gf.load_word_vec('../../test/fasttext_fin_model_50.vec') #word embedding from fasttxt
	#word_vec_word2vec_dict = gf.load_word_vec('../../test/word2vec.txt') #word embedding from word2vec
	#tfidf_dict=gf.load_tfidf_dict('../../test/atec_nl_sim_tfidf.txt')
	#vocabulary_word2index, vocabulary_index2word, vocabulary_label2index, vocabulary_index2label = gf.create_vocabulary('../../test/atec_nlp_sim_train.csv',60000,name_scope='',tokenize_style='')
	model = Word2Vec.load('./w2v_256.mod')
	index_dict, word_vectors= create_dictionaries(model)
	#f = open('index_dict.txt','r')  
	#a = f.read()  
	#index_dict = eval(a) 
	new_dic = index_dict
	print ("Setting up Arrays for Keras Embedding Layer...")
	n_symbols = len(index_dict) + 1  # 索引数字的个数，因为有的词语索引为0，所以+1
	#embedding_weights = np.loadtxt('./embedding.txt')
	embedding_weights = np.zeros((n_symbols, 256))  # 创建一个n_symbols * 100的0矩阵
	for w, index in index_dict.items():  # 从索引为1的词语开始，用词向量填充矩阵
	#	if index < len(word_vectors):
		embedding_weights[index, :] = word_vectors[w]  # 词向量矩阵，第一行是0向量（没有索引为0的词语，未被填充）
	kera_model = sim_lstm.get_model(n_symbols,embedding_weights)
	kera_model.load_weights('./lstm_100_100_0.15_0.15myword256_20_true_bal.h5')
	#model = gensim.models.Word2Vec.load('../models/word2vec_wx')
	maxlen = 20
	with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
		for line in fin:
			number,sen1, sen2 = line.strip().split('\t')
			#print sen1,sen2
			features_vector1 = sentence_to_index_array_for_test(new_dic, sen1)
			features_vector2 = sentence_to_index_array_for_test(new_dic, sen2)
			#features_vector=sequence.pad_sequences(features_vector, maxlen=maxlen)
			if len(features_vector1) > 20:
				features_vector1 = features_vector1[0:20]
			elif len(features_vector1) < 20:
				features_vector1 = np.pad(features_vector1,(0,20-len(features_vector1)), 'constant')
			if len(features_vector2) > 20:
				features_vector2 = features_vector2[0:20]
			elif len(features_vector2) < 20:
				features_vector2 = np.pad(features_vector2,(0,20-len(features_vector2)), 'constant')
			result = kera_model.predict([features_vector1.reshape(1,20),features_vector2.reshape(1,20)])
			if result[0][0] > 0.5:
				fout.write(str(number) + '\t1\n')
			elif result[0][0] < 0.5:
				fout.write(str(number) + '\t0\n')
			#else:
				#fout.write(lineno + '\t0\n')

if __name__ == '__main__':
	process(sys.argv[1], sys.argv[2])
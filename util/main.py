#-*- coding: utf-8 -*-
# coding:utf-8
import jieba
import sys
reload(sys)
sys.path.append("..")
sys.setdefaultencoding("utf-8")
import math
import gensim
import numpy as np
import tensorflow as tf
import get_feature as gf
from tensorflow.python import pywrap_tensorflow

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
	word_vec_fasttext_dict=gf.load_word_vec('../../test/fasttext_fin_model_50.vec') #word embedding from fasttxt
	word_vec_word2vec_dict = gf.load_word_vec('../../test/word2vec.txt') #word embedding from word2vec
	tfidf_dict=gf.load_tfidf_dict('../../test/atec_nl_sim_tfidf.txt')
	vocabulary_word2index, vocabulary_index2word, vocabulary_label2index, vocabulary_index2label = gf.create_vocabulary('../../test/atec_nlp_sim_train.csv',60000,name_scope='',tokenize_style='')
	reader = pywrap_tensorflow.NewCheckpointReader('../modelss/mlp_model')  
	var_to_shape_map = reader.get_variable_to_shape_map()  
	for key in var_to_shape_map:  
		print("tensor_name: ", key)  
		#print(reader.get_tensor(key)) # Remove this is you want to print only variable names  
			   
	#m_saver = tf.train.Saver()
	sess=tf.Session()      
	#First let's load meta graph and restore weights  
	m_saver = tf.train.import_meta_graph('../modelss/mlp_model.meta')  
	print("import end")
	m_saver.restore(sess,'../modelss/mlp_model')
	#m_saver.restore(sess,tf.train.latest_checkpoint('../models/')) 
	print("get restore end")
	graph = tf.get_default_graph()
	print("get val begin")
	x=graph.get_operation_by_name('x').outputs[0]
	#y = tf.placeholder(tf.float32, [None, 2])
	keep_prob = graph.get_operation_by_name('keep_probe').outputs[0]
	y=tf.get_collection("pred_network")[0]
	print('begin')
		#saver.restore(sess, "../ATEC-NLP-2018/models/mlp_model")
	count = 1
	with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
		for line in fin:
			sen1, sen2,label = line.strip().split('\t')
			#print sen1,sen2
			features_vector = gf.data_mining_features(count,sen1,sen2,vocabulary_word2index,word_vec_fasttext_dict,word_vec_word2vec_dict,tfidf_dict, n_gram=8)
			#print features_vector
			result = sess.run(y,feed_dict={x: [features_vector],keep_prob:1.0})
			print(result)
			if result[0][0] > result[0][1]:
				fout.write(str(count) + '\t' + result + '\t1\n')
			else:
				fout.write(str(count) + '\t' + result +  '\t0\n')
			count = count + 1
			#else:
				#fout.write(lineno + '\t0\n')

if __name__ == '__main__':
	process(sys.argv[1], sys.argv[2])

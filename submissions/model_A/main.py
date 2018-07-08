#-*- coding: utf-8 -*-
# coding:utf-8
import jieba
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import math
import gensim
import numpy as np

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
	model = gensim.models.Word2Vec.load('word2vec_wx')
	with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
		for line in fin:
			lineno, sen1, sen2 = line.strip().split('\t')
			#print sen1,sen2
			wordlist1 = jieba.cut(sen1)
			count1 = 0
			res_vec1 = np.zeros(256, dtype=np.float32)
			for word in wordlist1:
				try:
					c = model[word]
				except KeyError:
					print ('not in vocabulary')
					c = np.zeros(256, dtype=np.float32)
				res_vec1 = res_vec1 + c  # 将每一个单词转换成向量model[单词]
				count1 += 1  # 计算相加元素的个数
			res_vec1= (res_vec1 / count1)
			wordlist2 = jieba.cut(sen2)
			count2 = 0
			res_vec2 = np.zeros(256, dtype=np.float32)
			for word in wordlist2:
				try:
					c = model[word]
				except KeyError:
					print ('not in vocabulary')
					c = np.zeros(256, dtype=np.float32)
				res_vec2 = res_vec2 + c  # 将每一个单词转换成向量model[单词]
				count2 += 1  # 计算相加元素的个数
			res_vec2 = (res_vec2 / count2) 
			r=cos_dist(res_vec1,res_vec2) 
			if r > 0.8:
				fout.write(lineno + '\t1\n')
			else:
				fout.write(lineno + '\t0\n')

if __name__ == '__main__':
	process(sys.argv[1], sys.argv[2])

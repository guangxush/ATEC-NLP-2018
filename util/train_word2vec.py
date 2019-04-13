# -*- coding:utf-8 -*-
import logging
import os
import sys
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import jieba

reload(sys)
sys.setdefaultencoding("utf-8")


def train_word2vec():
    input_file = '../data/atec_nlp_sim_train.csv'
    data_prepare(input_file)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = LineSentence("../data/train_questions_with_evidence.txt")

    model = Word2Vec(sentences, min_count=1, iter=1000, size=256)

    model.train(sentences, total_examples=model.corpus_count, epochs=1000)

    model.save("../models/w2v_256.mod")
    model_loaded = Word2Vec.load("../models/w2v_256.mod")

    sim = model_loaded.wv.most_similar(positive=[u'花呗'])
    for s in sim:
        print s[0]


def data_prepare(input_file):
    jieba.add_word('花呗')
    jieba.add_word('借呗')
    print "data prepare start....."
    if os.path.exists('../data/train_questions_with_evidence.txt'):
        return
    fr = open(input_file, 'r')
    fw = open('../data/train_questions_with_evidence.txt', 'w')
    for line in fr:
        line = line.strip('\n').split('\t')
        seg_list1 = jieba.cut(line[1])
        fw.write("\t".join(seg_list1) + '\n')
        seg_list2 = jieba.cut(line[2])
        fw.write("\t".join(seg_list2) + '\n')
    fr.close()
    fw.close()
    print('data prepared!!!!!')


if __name__ == '__main__':
    train_word2vec()

# -*- coding:utf-8 -*-
from gensim.models import word2vec
import logging
import os
import pickle
import sys
import numpy as np
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

reload(sys)
sys.setdefaultencoding("utf-8")


def train_word2vec():
    input_file = '../data/atec_nlp_sim_train.csv'
    data_prepare(input_file)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus(input_file)
    save_model_file = '../models/w2v_256.mod'
    model = word2vec.Word2Vec(sentences, min_count=1, size=256, window=5, workers=4)

    # vocab = pickle.load(open('../models/vocabulary_all.pkl', 'rb'))
    weights = model.wv.syn0
    d = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    emb = np.zeros(shape=(len(d) + 2, 256), dtype='float32')
    model.save(save_model_file)
    model.wv.save_word2vec_format(save_model_file, binary=False)
    for i in range(0, len(d)):
        emb[i, :] = weights[i, :]
    np.save(open('../models/sst_256_dim_all.embeddings', 'wb'), emb)
    return model


def data_prepare(input_file):
    print "data prepare start....."
    if os.path.exists('../data/train_questions_with_evidence.txt'):
        return
    fr = open(input_file, 'r')
    fw = open('../data/train_questions_with_evidence.txt', 'w')
    for line in fr:
        line = line.strip('\n').split('\t')
        fw.write(str(line[1]) + '\n')
        fw.write(str(line[2]) + '\n')
    fr.close()
    fw.close()
    print('data prepared!!!!!')


if __name__ == '__main__':
    train_word2vec()

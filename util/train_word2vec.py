# -*- coding:utf-8 -*-

import logging
from gensim.models.word2vec import LineSentence, Word2Vec
import sys
reload(sys)
sys.setdefaultencoding("utf-8")


# raw_sentences = ["the quick brown fox jumps over the lazy dogs","yoyoyo you go home now to sleep"]

sentences= LineSentence("../data/train_questions_with_evidence.txt")

model = Word2Vec(sentences ,min_count=1, iter=1000,size=256)
model.train(sentences, total_examples=model.corpus_count, epochs=1000)

model.save("../models/w2v_256.mod")
model_loaded = Word2Vec.load("../models/w2v_256.mod")

sim = model_loaded.wv.most_similar(positive=[u'花呗'])
for s in sim:
    print s[0]
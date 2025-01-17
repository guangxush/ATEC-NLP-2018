# -*- coding:utf-8 -*-
########################################
## import packages
########################################
import os
import re
import csv
import codecs

import jieba
import numpy as np

from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.preprocessing.text
import sys
import cPickle
reload(sys)
sys.setdefaultencoding('utf-8')
from util.dataset import sentence_to_index_array,create_dictionaries,load_all_sentence
########################################
# set directories and parameters
########################################
DATA_DIR = './dataset/'
EMBEDDING_FILE = './models/w2v/w2v.mod'
TRAIN_DATA_FILE = DATA_DIR + 'mytrain_pair.csv'
TEST_DATA_FILE = DATA_DIR + 'mytest_pair.csv'
MAX_SEQUENCE_LENGTH = 20
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 256
VALIDATION_SPLIT = 0.1

# num_lstm = np.random.randint(175, 275)
# num_dense = np.random.randint(100, 150)
# rate_drop_lstm = 0.15 + np.random.rand() * 0.25
# rate_drop_dense = 0.15 + np.random.rand() * 0.25

num_lstm = 175
num_dense = 100
rate_drop_lstm = 0.15
rate_drop_dense = 0.15

act = 'relu'
re_weight = True  # whether to re-weight classes to fit the 17.5% share in test set

STAMP = './models/lstm/lstm_%d_%d_%.2f_%.2f' % (num_lstm, num_dense, rate_drop_lstm, \
                                                rate_drop_dense)

save = True
load_tokenizer = False
save_path = "./models/lstm"
tokenizer_name = "tokenizer.pkl"
embedding_matrix_path = "./models/lstm/embedding_matrix.npy"

########################################
# define the model structure
########################################
def get_model(nb_words, embedding_matrix):
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)

    merged = concatenate([x1, y1])
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(num_dense, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)
    preds = Dense(1, activation='sigmoid')(merged)
    model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='nadam',
                  metrics=['acc'])
    model.summary()
    return model


#######################################
# train the model
########################################


def train_model(data_1, data_2, labels,test_1,test_2,test_label,embedding_weights):
    print(STAMP)
    print('embeding ' +str(embedding_weights))
    model = get_model(n_symbols,embedding_weights)
    fin = open('./result.txt','w')
    for iteration in (0, 100):
        #if iteration>5:
            #optimizer.lr.set_value(0.02)
            #optimizer.momentum.set_value(0.04) 
        model.fit([data_1, data_2], labels, validation_data=([test_1, test_2], test_label), epochs=1, batch_size=10, shuffle=True)
        #model.fit(x_embed, y, nb_epoch=1, batch_size = batch_size, show_accuracy=True)
        preds = model.predict([data_1, data_2], verbose=0, batch_size = 10)
        print('pred ' + str(preds))
        fin.write(preds)
    bst_model_path = STAMP + '.h5'
    model.save(bst_model_path)
    #early_stopping = EarlyStopping(monitor='val_loss', patience=100)
    #bst_model_path = STAMP + '.h5'
    #model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)
    #hist = model.fit([data_1, data_2], labels, validation_data=([test_1, test_2], test_label), epochs=100, batch_size=10, shuffle=True, callbacks=[early_stopping, model_checkpoint])
    #resultmy = model.predict([data_1, data_2])
    #fin = open('./result.txt','a')
    #fin.write(resultmy)
	#model.load_weights(bst_model_path)
    #bst_score = min(hist.history['loss'])
    #bst_acc = max(hist.history['acc'])
    #print bst_acc, bst_score


if __name__ == '__main__':
    model = Word2Vec.load('./models/word2vec_wx')
    index_dict, word_vectors= create_dictionaries(model)
    new_dic = index_dict
    print ("Setting up Arrays for Keras Embedding Layer...")
    n_symbols = len(index_dict) + 1  # 索引数字的个数，因为有的词语索引为0，所以+1
    embedding_weights = np.zeros((n_symbols, 256))  # 创建一个n_symbols * 100的0矩阵
    for w, index in index_dict.items():  # 从索引为1的词语开始，用词向量填充矩阵
        embedding_weights[index, :] = word_vectors[w]  # 词向量矩阵，第一行是0向量（没有索引为0的词语，未被填充）
    train_dataset1,train_dataset2,labels = load_all_sentence('./data/inputadd_balance.txt','2')
    test_dataset1,test_dataset2,test_labels = load_all_sentence('./data/input.txt','3')
    print('load data1 ' + str(len(train_dataset1)))
    print('load data2 ' + str(len(train_dataset2)))
    train_dataset1 = sentence_to_index_array(new_dic, train_dataset1,'2',MAX_SEQUENCE_LENGTH)
    train_dataset2 = sentence_to_index_array(new_dic, train_dataset2,'2',MAX_SEQUENCE_LENGTH)
    test_dataset1 = sentence_to_index_array(new_dic, test_dataset1,'2',MAX_SEQUENCE_LENGTH)
    test_dataset2 = sentence_to_index_array(new_dic, test_dataset2,'2',MAX_SEQUENCE_LENGTH)
    print('sen2array'+ str(len(train_dataset1)))
    print("训练集1shape： " + str(train_dataset1.shape))
    print("训练集2shape： " + str(train_dataset2.shape))
    print('labels' + str(labels[0:100]))
    train_model(train_dataset1,train_dataset2,labels,test_dataset1,test_dataset2,test_labels,embedding_weights)


# predicts = model.predict([test_data_1, test_data_2], batch_size=10, verbose=1)

# for i in range(len(test_ids)):
#    print "t1: %s, t2: %s, score: %s" % (test_texts_1[i], test_texts_2[i], predicts[i])

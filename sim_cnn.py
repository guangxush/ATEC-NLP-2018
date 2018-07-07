# -*- coding:utf-8 -*-

import numpy as np
from gensim.models import Word2Vec
np.random.seed(1337)
from keras.layers import Dense, Dropout, Activation, Input
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from util.dataset import sentence_to_index_array, create_dictionaries, load_all_sentence
from util.f1 import f1

# set parameters:
DATA_DIR = './dataset/'
EMBEDDING_FILE = './models/w2v/w2v.mod'
TRAIN_DATA_FILE = DATA_DIR + 'mytrain_pair.csv'
TEST_DATA_FILE = DATA_DIR + 'mytest_pair.csv'


max_features = 5001
maxlen = 100
batch_size = 32
embedding_dims = 100
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 10
max_sequence_length = 20
num_cnn = 175
rate_drop_cnn = 0.15
rate_drop_dense = 0.15
num_dense = 100
act = 'relu'
re_weight = True  # whether to re-weight classes to fit the 17.5% share in test set

STAMP = './models/cnn/cnn_f1_%d_%d_%.2f_%.2f' % (num_cnn, num_dense, rate_drop_cnn, \
                                                rate_drop_dense)

save = True
load_tokenizer = False
save_path = "./models/cnn"
tokenizer_name = "tokenizer.pkl"
embedding_matrix_path = "./models/cnn/embedding_matrix.npy"


def cnn_model(nb_words, embedding_matrix):
    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    embedding_layer = Embedding(nb_words,
                                embedding_dims,
                                weights=[embedding_matrix],
                                input_length=max_sequence_length,
                                trainable=True,
                                mask_zero=False)
    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    cov_layer = Conv1D(filters,
                       kernel_size,
                       padding='valid',
                       activation='relu',
                       strides=1)
    # we use max pooling:
    pooling_layer = GlobalMaxPooling1D()

    # We add a vanilla hidden layer:

    sequence_1_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x0 = cov_layer(embedded_sequences_1)
    x1 = pooling_layer(x0)

    sequence_2_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y0 = cov_layer(embedded_sequences_2)
    y1 = pooling_layer(y0)

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
                  metrics=[f1])
    model.summary()
    return model
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['f1'])
    model.summary()
    return model

#######################################
# train the model
########################################


def train_model(data_1, data_2, labels):
    model = cnn_model(n_symbols, embedding_weights)
    early_stopping = EarlyStopping(monitor='val_f1', patience=10)
    bst_model_path = STAMP + '.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, monitor='val_f1', save_best_only=True, save_weights_only=False)
    hist = model.fit([data_1, data_2], labels, validation_data=([data_1, data_2], labels), epochs=100, batch_size=10, shuffle=True, callbacks=[early_stopping, model_checkpoint])
    model.load_weights(bst_model_path)
    bst_loss = min(hist.history['loss'])
    bst_val_loss = min(hist.history['val_loss'])
    print("bst_loss:" + str(bst_loss) + "bst_val_loss" + str(bst_val_loss))
    bst_val_f1 = max(hist.history['val_f1'])
    bst_f1 = max(hist.history['f1'])
    print("bst_f1:"+str(bst_f1)+"bst_val_f1"+str(bst_val_f1))


if __name__ == '__main__':
    model = Word2Vec.load('./models/w2v.mod')
    index_dict, word_vectors = create_dictionaries(model)
    new_dic = index_dict
    print ("Setting up Arrays for Keras Embedding Layer...")
    n_symbols = len(index_dict) + 1  # 索引数字的个数，因为有的词语索引为0，所以+1
    embedding_weights = np.zeros((n_symbols, 100))  # 创建一个n_symbols * 100的0矩阵
    for w, index in index_dict.items():  # 从索引为1的词语开始，用词向量填充矩阵
        embedding_weights[index, :] = word_vectors[w]  # 词向量矩阵，第一行是0向量（没有索引为0的词语，未被填充）
    train_dataset1,train_dataset2,labels = load_all_sentence('./data/inputadd.txt', '2')
    print('load data1 ' + str(len(train_dataset1)))
    print('load data2 ' + str(len(train_dataset2)))
    train_dataset1 = sentence_to_index_array(new_dic, train_dataset1, '2', max_sequence_length)
    train_dataset2 = sentence_to_index_array(new_dic, train_dataset2, '2', max_sequence_length)
    print('sen2array'+ str(len(train_dataset1)))
    print("训练集1shape： " + str(train_dataset1.shape))
    print("训练集2shape： " + str(train_dataset2.shape))
    train_model(train_dataset1, train_dataset2, labels)

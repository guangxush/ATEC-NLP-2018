# -*- coding:utf-8 -*-
########################################
## import packages
########################################

import numpy as np
from gensim.models import Word2Vec
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Bidirectional, CuDNNLSTM
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
import sys
from util.f1 import f1
#from sklearn.metrics import f1_score as f1

reload(sys)
sys.setdefaultencoding('utf-8')
from util.dataset import sentence_to_index_array, create_dictionaries, load_all_sentence

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

num_lstm = 100
num_dense = 100
rate_drop_lstm = 0.15
rate_drop_dense = 0.15

act = 'relu'
re_weight = True  # whether to re-weight classes to fit the 17.5% share in test set

STAMP = './models/CuDNNLSTM_%d_%d_%.2f_%.2f' % (num_lstm, num_dense, rate_drop_lstm, \
                                                rate_drop_dense)

save = True
load_tokenizer = False
save_path = "./models/"
tokenizer_name = "tokenizer.pkl"
embedding_matrix_path = "./models/embedding_matrix.npy"


########################################
# define the model structure
########################################
def get_model(nb_words, embedding_matrix):
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    lstm0 = CuDNNLSTM(num_lstm)

    lstm1 = Bidirectional(CuDNNLSTM(num_lstm))

    lstm2 = CuDNNLSTM(num_lstm)

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm0(embedded_sequences_1)
    x_lstm1 = lstm1(x1)
    x_lstm2 = lstm2(x_lstm1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm0(embedded_sequences_2)
    y_lstm1 = lstm1(y1)
    y_lstm2 = lstm2(y_lstm1)

    merged = concatenate([x_lstm2, y_lstm2])
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(num_dense, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)
    preds = Dense(1, activation='sigmoid')(merged)
    model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[f1])
    model.summary()
    return model


#######################################
# train the model
########################################


def train_model(data_1, data_2, labels, test_1, test_2, test_label, embedding_weights, n_symbols):
    print(STAMP)
    print('embeding ' + str(embedding_weights))
    model = get_model(n_symbols, embedding_weights)
    early_stopping = EarlyStopping(monitor='val_loss', patience=8)
    bst_model_path = STAMP + '_myword256_20' + '.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, monitor='val_f1', save_best_only=True, save_weights_only=True)
    hist = model.fit([data_1, data_2], labels, validation_data=([test_1, test_2], test_label), epochs=101,
                     batch_size=10, shuffle=True, callbacks=[early_stopping, model_checkpoint])
    # resultmy = model.predict([data_1, data_2])
    # fin = open('./result.txt','a')
    # fin.write(resultmy)
    # model.load_weights(bst_model_path)
    bst_loss = min(hist.history['loss'])
    bst_val_loss = min(hist.history['val_loss'])
    print("bst_loss:" + str(bst_loss) + "bst_val_loss" + str(bst_val_loss))
    bst_val_f1 = max(hist.history['val_f1'])
    bst_f1 = max(hist.history['f1'])
    print("bst_f1:"+str(bst_f1)+"bst_val_f1"+str(bst_val_f1))


if __name__ == '__main__':
    model = Word2Vec.load('./models/w2v_256.mod')
    index_dict, word_vectors = create_dictionaries(model)
    new_dic = index_dict
    print ("Setting up Arrays for Keras Embedding Layer...")
    n_symbols = len(index_dict) + 1  # 索引数字的个数，因为有的词语索引为0，所以+1
    embedding_weights = np.zeros((n_symbols, 256))  # 创建一个n_symbols * 100的0矩阵
    for w, index in index_dict.items():  # 从索引为1的词语开始，用词向量填充矩阵
        embedding_weights[index, :] = word_vectors[w]  # 词向量矩阵，第一行是0向量（没有索引为0的词语，未被填充）
    print('length = ' + str(len(embedding_weights)))
    train_dataset1, train_dataset2, labels = load_all_sentence('./data/train_data_new.txt', '2')
    test_dataset1, test_dataset2, test_labels = load_all_sentence('./data/test_data_new.txt', '2')
    print('load data1 ' + str(len(train_dataset1)))
    print('load data2 ' + str(len(train_dataset2)))
    train_dataset1 = sentence_to_index_array(new_dic, train_dataset1, '2', MAX_SEQUENCE_LENGTH)
    train_dataset2 = sentence_to_index_array(new_dic, train_dataset2, '2', MAX_SEQUENCE_LENGTH)
    test_dataset1 = sentence_to_index_array(new_dic, test_dataset1, '2', MAX_SEQUENCE_LENGTH)
    test_dataset2 = sentence_to_index_array(new_dic, test_dataset2, '2', MAX_SEQUENCE_LENGTH)
    print('sen2array' + str(len(train_dataset1)))
    print("训练集1shape： " + str(train_dataset1.shape))
    print("训练集2shape： " + str(train_dataset2.shape))
    print('labels' + str(labels[0:100]))
    train_model(train_dataset1, train_dataset2, labels, test_dataset1, test_dataset2, test_labels, embedding_weights,
                n_symbols)

# predicts = model.predict([test_data_1, test_data_2], batch_size=10, verbose=1)

# for i in range(len(test_ids)):
#    print "t1: %s, t2: %s, score: %s" % (test_texts_1[i], test_texts_2[i], predicts[i])

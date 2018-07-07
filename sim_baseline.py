# -*- coding:utf-8 -*-
########################################
## import packages
########################################

import numpy as np
from gensim.models import Word2Vec
from keras.layers import Dense, Input, LSTM, Embedding, Dropout\
    , CuDNNLSTM, Bidirectional, Concatenate, Multiply, Lambda, Maximum, Subtract
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import sys

from keras.optimizers import Adam

from util.f1 import f1
from keras import backend as K

# from sklearn.metrics import f1_score as f1

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
MAX_CHAR_LENGTH = 30
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

STAMP = './models/lstm_f1_%d_%d_%.2f_%.2f' % (num_lstm, num_dense, rate_drop_lstm, \
                                              rate_drop_dense)

save = True
load_tokenizer = False
save_path = "./models/"
tokenizer_name = "tokenizer.pkl"
embedding_matrix_path = "./models/embedding_matrix.npy"


########################################
# define the model structure
########################################
def get_model(nb_words, nb_chars, embedding_matrix):
    input1 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    input2 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    input3 = Input(shape=(5,))

    embed1 = Embedding(nb_words,
                       EMBEDDING_DIM,
                       weights=[embedding_matrix],
                       input_length=MAX_SEQUENCE_LENGTH,
                       trainable=True,
                       mask_zero=False)

    lstm0 = CuDNNLSTM(num_lstm, return_sequences=True)
    lstm1 = Bidirectional(CuDNNLSTM(num_lstm))
    lstm2 = CuDNNLSTM(num_lstm)
    att1 = Attention(10)
    den = Dense(64, activation='tanh')

    # att1 = Lambda(lambda x: K.max(x,axis = 1))

    v3 = embed1(input3)
    v1 = embed1(input1)
    v2 = embed1(input2)
    v11 = lstm1(v1)
    v22 = lstm1(v2)
    v1ls = lstm2(lstm0(v1))
    v2ls = lstm2(lstm0(v2))
    v1 = Concatenate(axis=1)([att1(v1), v11])
    v2 = Concatenate(axis=1)([att1(v2), v22])

    input1c = Input(shape=(MAX_CHAR_LENGTH,))
    input2c = Input(shape=(MAX_CHAR_LENGTH,))
    embed1c = Embedding(nb_chars, EMBEDDING_DIM)

    lstm1c = Bidirectional(CuDNNLSTM(6))
    att1c = Attention(10)

    v1c = embed1(input1c)
    v2c = embed1(input2c)
    v11c = lstm1c(v1c)
    v22c = lstm1c(v2c)
    v1c = Concatenate(axis=1)([att1c(v1c), v11c])
    v2c = Concatenate(axis=1)([att1c(v2c), v22c])
    mul = Multiply()([v1, v2])
    sub = Lambda(lambda x: K.abs(x))(Subtract()([v1, v2]))
    maximum = Maximum()([Multiply()([v1, v1]), Multiply()([v2, v2])])
    mulc = Multiply()([v1c, v2c])
    subc = Lambda(lambda x: K.abs(x))(Subtract()([v1c, v2c]))
    maximumc = Maximum()([Multiply()([v1c, v1c]), Multiply()([v2c, v2c])])
    sub2 = Lambda(lambda x: K.abs(x))(Subtract()([v1ls, v2ls]))
    matchlist = Concatenate(axis=1)([mul, sub, mulc, subc, maximum, maximumc, sub2])
    matchlist = Dropout(0.05)(matchlist)
    matchlist = Concatenate(axis=1)(
        [Dense(32, activation='relu')(matchlist), Dense(48, activation='sigmoid')(matchlist)])

    res = Dense(1, activation='sigmoid')(matchlist)
    model = Model(inputs=[input1, input2, input3, input1c, input2c], outputs=res)
    model.compile(optimizer=Adam(lr=0.001), loss="binary_crossentropy")
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
    print("bst_f1:" + str(bst_f1) + "bst_val_f1" + str(bst_val_f1))


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

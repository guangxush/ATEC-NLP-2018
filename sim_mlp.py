from __future__ import print_function
from keras import losses
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.metrics import mean_absolute_error
from util.dataset import load_data_with_sentences_single_flag, load_data_with_features, load_data_with_sentences
import numpy as np
import sys
from util.f1 import f1


def mlp(sample_dim, loss_name, result_dim):
    model = Sequential()
    model.add(Dense(512, kernel_initializer='glorot_uniform', activation='relu', input_dim=sample_dim))
    model.add(Dense(128, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dense(64, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dense(32, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dense(result_dim))
    model.compile(loss=loss_name, optimizer='adam', metrics=[f1])
    return model


if __name__ == '__main__':
    data_flag = True if sys.argv[1] == 'sentences' else False
    print('***** Start ATEC-NLP-2018 *****')
    print('Loading data ...')
    if data_flag:
        x_train, y_train = load_data_with_sentences('./data/test_data_balance.csv')
        x_dev, y_dev = load_data_with_sentences('./data/train_data_balance.csv')
        result_dim = 2
        filepath = './models/mlp_sentences.hdf5'
    else:
        x_train, y_train = load_data_with_features('./data/word2vec_avg.csv')
        x_dev, y_dev = load_data_with_features('./data/word2vec_avg.csv')
        result_dim = 1
        filepath = './models/mlp_features.hdf5'
    print('Training MLP model ...')
    check_pointer = ModelCheckpoint(filepath=filepath, monitor='val_f1', verbose=1, save_best_only=True,
                                    save_weights_only=False)
    early_stopping = EarlyStopping(patience=30)
    csv_logger = CSVLogger('logs/mlp.log')
    mlp_model = mlp(sample_dim=x_train.shape[1], loss_name=losses.mae, result_dim=result_dim)
    mlp_model.fit(x_train, y_train, batch_size=128, epochs=100, verbose=1, validation_data=(x_dev, y_dev),
                  callbacks=[check_pointer, early_stopping, csv_logger])
    print('***** End ATEC-NLP-2018 *****')
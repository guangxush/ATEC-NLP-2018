# coding=utf-8
import sys
import pandas as pd
import csv


#  读取处理之后的csv文件
def read_csv_data(input_file):
    train_dataframe = pd.read_csv(input_file, header=None, nrows=1)
    train_dataset = train_dataframe.values
    sen1_train = train_dataset[:, 0:256].astype('float32')
    sen2_train = train_dataset[:, 256:-1].astype('float32')
    sen_flag = train_dataset[:, -1].astype('float32')
    print(sen1_train.shape)
    print(sen2_train.shape)
    print(sen_flag.shape)
    return

if __name__ == '__main__':
    input_file = '../data/word2vec_avg.csv'
    read_csv_data(input_file)
# -*- coding: utf-8 -*-
import numpy as np

#加载句子特征
def load_data_with_sentences():
    input_file = open('../data/word2vec_avg.csv', 'r')
    input_x = []
    input_y = []
    i = 0
    for line in input_file:
        i += 1
        record = line.split(',')
        input_x.append(np.array(record[0:512]))
        if record[-1] == '1':
            input_y.append([1.0, 0.0])
        else:
            input_y.append([0.0, 1.0])
        if i > 39332:
            break
    input_X = np.array(input_x, dtype=np.float32)
    input_Y = np.array(input_y, dtype=np.float32)
    print input_X[0]
    print input_Y[0]
    print input_X.shape
    print input_Y.shape
    print input_X.dtype
    print input_Y.dtype
    return input_X, input_Y


#加载特征工程之后的句子向量
def load_data_with_features():
    # 读取输入数据
    input_file = open('../data/train_data.csv', 'r')
    input_x = []
    input_y = []
    for line in input_file:
        record = line.split('\t')
        input_x.append(np.array(record[0][1:-1].split(',')))
        input_y.append(np.array(record[-1][1:-2].split(',')))

    input_X = np.array(input_x, dtype=np.float32)
    input_Y = np.array(input_y, dtype=np.float32)
    print input_X[0]
    print input_Y[0]
    print input_X.shape
    print input_Y.shape
    print input_X.dtype
    print input_Y.dtype
    return input_X, input_Y

if __name__ == '__main__':
    load_data_with_sentences()
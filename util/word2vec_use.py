# coding=utf-8

import gensim
import pandas as pd
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


def word2vec(input_file):
    #加载已经训练好的model
    model = gensim.models.Word2Vec.load('../models/word2vec_wx')
    #print(model[u'花'])
    input_data = pd.read_csv(input_file)  # 读取已经切好词的句子
    out_file = open('../data/cut_wordvec.csv', 'w', 'utf-8')
    for idx in input_data.index:
        out_line = []
        for word in (input_data.loc[idx][0].split('/')):
            out_line.append(model[word])  # 将每一个单词转换成向量model[单词]
        out_file.write("/".join(out_line) + '\n')  # 将向量重新保存到文件中
    return


if __name__ == '__main__':
    input_file = '../data/cut_word.csv'
    word2vec(input_file)

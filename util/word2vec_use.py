# coding=utf-8
import sys
import gensim
import pandas as pd
import warnings

from numpy.core.multiarray import ndarray

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


# 将句子中的单词转换成向量，整个句子向量计算平均值作为句子的向量
def word2vec_avg(model, input_data, output_file):
    print('******start word to avg_vec******')
    out_file = open(output_file, 'w')
    for idx in input_data.index:  # 逐行遍历
        out_line = []
        print(idx)
        for word in (input_data.loc[idx][0].split('/')):
            #print(word)
            try:
                c = model[word.decode('utf-8')]
            except KeyError:
                print ('not in vocabulary')
                c = 0
            out_line.append(c)  # 将每一个单词转换成向量model[单词]
        sum = 0
        for i in out_line:
            sum += i
        out_file.write(str(sum/len(out_line))+'\n')  # 将向量重新保存到文件中
        print('******end word to avg_vec******')
    return


# 将句子中的单词转换成向量，整个句子作为句子的向量，相邻的两句向量长度大小一样
def word2vec_all(model, input_data, output_file):
    print('******start word to all_vec******')
    out_file = open(output_file, 'w')
    for idx in input_data.index:  # 逐行遍历
        if idx % 2 == 0:  # 只遍历奇数行
            pass
        else:
            cur_line = input_data.loc[idx][0].split('/')
            next_line = input_data.loc[idx+1][0].split('/')
            cur_out_line = []
            next_out_line = []
            print(idx)
            for word in cur_line:
                # print(word)
                try:
                    c = model[word.decode('utf-8')]
                except KeyError:
                    print ('not in vocabulary')
                    c = 0
                cur_out_line.append(str(c))  # 将当前行每一个单词转换成向量model[单词]
            print(idx)
            for word in next_line:
                # print(word)
                try:
                    c = model[word.decode('utf-8')]
                except KeyError:
                    print ('not in vocabulary')
                    c = 0
                next_out_line.append(str(c))  # 将下一行每一个单词转换成向量model[单词]
            if(len(cur_out_line) == len(next_out_line)):
                out_file.write(','.join(cur_out_line)+'\n')
                out_file.write(','.join(next_out_line) + '\n')
            elif(len(cur_out_line) < len(next_out_line)):
                cur_out_line_final = cur_out_line
                for i in range(len(cur_out_line), len(next_out_line)-1):
                    cur_out_line_final.append(str(0))
                out_file.write(','.join(cur_out_line_final) + '\n')
                out_file.write(','.join(next_out_line) + '\n')
            else:
                out_file.write(','.join(cur_out_line) + '\n')
                next_out_line_final = next_out_line
                for i in range(len(next_out_line), len(cur_out_line)-1):
                    next_out_line_final.append(str(0))
                out_file.write(','.join(next_out_line_final) + '\n')
    print('******end word to all_vec******')
    return

if __name__ == '__main__':
    avg_flag = True if sys.argv[1] == "avg" else False
    input_file = '../data/cut_word.csv'
    # 加载已经训练好的model
    model = gensim.models.Word2Vec.load('../models/word2vec_wx')
    input_data = pd.read_csv(input_file)
    # 读取已经切好词的句子
    if avg_flag:
        word2vec_avg(model, input_data, output_file="../data/word2vec_avg.csv")
    else:
        word2vec_all(model, input_data, output_file="../data/word2vec_all.csv")
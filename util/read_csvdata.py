# coding=utf-8
import sys
import pandas as pd
import csv


#  读取处理之后的csv文件
def read_csv_data(input_file):
    sen_vec1, sen_vec2, flag = [], [], []
    csv_file = csv.reader(open(input_file, 'r'))
    i = 0
    for line in csv_file:
        for item in line:
            print(item)
            sen_all = item.split('\t')
            sen_vec1 = sen_all[0]
            sen_vec2 = sen_all[1]
            flag = sen_all[2]
            print(sen_vec1)
            print(sen_vec2)
            print(flag)
            i += 1
            if i >= 2:
                break
    return

if __name__ == '__main__':
    input_file = '../data/word2vec_avg.csv'
    read_csv_data(input_file)
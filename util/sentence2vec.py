# -*- coding: utf-8 -*-
import gensim
import pandas as pd
import csv

def sentence2vec(input_file):
    csv_file = csv.reader(open(input_file, 'r'))
    for item in csv_file:
        for word in item:
            sen = word.split('\t')
            if len(sen) > 2:
                sen1 = sen[0]
                sen2 = sen[1]
                print(sen1.decode('utf-8'))
                print(sen2.decode('utf-8'))


if __name__ == '__main__':
    input_file = '../raw_data/atec_nlp_sim_train.csv'
    sentence2vec(input_file)
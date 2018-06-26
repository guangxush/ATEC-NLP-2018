# -*- coding: utf-8 -*-
# coding:utf-8
##jieba
import sys
import csv
import jieba

reload(sys)
sys.setdefaultencoding("utf-8")
out = open('../data/cut_wordreal.txt','a')

csv_file = csv.reader(open('../data/inputadd.txt', 'r'))
for item in csv_file:
    for word in item:
        s = str(word)
        sen = word.split('\t')
        if len(sen) > 2:
            sen1 = sen[1]
            sen2 = sen[2]
            seg_list = jieba.cut(sen1)  # 默认是精确模式
            out.write("/".join(seg_list) + '\n')
            seg_list = jieba.cut(sen2)
            out.write("/".join(seg_list) + '\n')

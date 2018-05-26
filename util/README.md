## 文件说明
- cal_cosine.py：计算两个输入a,b的欧式距离<br/>
		```python cal_cosine.py input_one input_two```

- cut_word.py：通过jieba对文本进行分词<br/>
		```python cut_word.py```

- senetence2vec.py：使用doc2vec计算两个句子之间的相似度<br/>
		```python senetence2vec.py```

- word2vec_train.py：通过使用大规模文本预料训练word2vec模型<br/>
		```python word2vec_train.py```

- word2vec_use.py：通过已训练好的word2vec模型将分好的词转换成词向量<br/>
		每一个词对应一个1\*255长度的向量，将句子中的所有词累加取平均值作为一个句子的向量<br/>
		```python word2vec_use.py avg```<br/>
		每一个词对应一个1\*255长度的向量，将句子中的n个词拼接作为一个句子的向量（n\*255）<br/>
		```python word2vec_use.py all```

- data_util.py 数据处理&特征工程<br/>
  原始数据格式保存在csv文件中
  数据格式: 行号,句子1,句子2,标签. 共4列数据通过"\n"分割

        001\n question1\n question2\n label
  数据量统计：<br/>
  { 5: 0.11388705332181162, 10: 0.6559243633406191, 15: 0.1654043613073756, 20: 0.04325725613785391})<br/>
  其中句子长度length： 5表示length<=5; 10表示5<=length<=10 ....<br/>
  由此可以看出该问题是短文本处理问题<br/>
  a.交换句子1，句子2

       如果句子1句子2有相同的意思那么句子2句子1也应该是相同的

       check: method get_training_data() at data_util.py

    b.给定的句子中随机交换数据的顺序

        因为同一个关键词可能包含一个句子中最重要的信息，所以这些关键词的改变顺序也应该能够发送这些信息;
        但是可可能存在一种情况，句子关键词顺序的改变导致句子意思发生改变
        check: method get_training_data() at data_util.py

  数据增强之后:training data: 81922 ;validation data: 1600; test data:800; percent of true label: 0.217

    c.符号的风格

        可以训练模型使用字符，或者单词或拼音。
        例如，即使你用拼音训练这个模型，它仍然可以获得较为合理的性能。
        在拼音中标记句子：我们首先将句子标记为单词，然后将其翻译成拼音。
        例如 它现在变成：['nihao'，'wo'，'de'，'pengyou']
    d.通过已有的数据集提取更多特征

    1) n-gram 相似度(blue score for n-gram=1,2,3...);

    2) 句子的长度，长度的差异

    3) 相同单词的数目，不同单词的数目

    4) 问题12开头是不是 how/why/when(wei shen me,zenme，ruhe，weihe）

    5) 编辑距离

    6) 用句袋表示句子的相似度（将tfidf与word2vec、fasttext中的词向量结合起来）

    7) 各种距离计算manhattan_distance,canberra_distance,minkowski_distance,euclidean_distance

            check data_mining_features method under data_util.py




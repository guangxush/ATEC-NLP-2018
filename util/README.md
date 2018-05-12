## 文件说明
- cal_cosine.py：计算两个输入a,b的欧式距离<br/>
		python cal_cosine.py input_one input_two

- cut_word.py：通过jieba对文本进行分词<br/>
		python cut_word.py

- senetence2vec.py：使用doc2vec计算两个句子之间的相似度<br/>
		python senetence2vec.py

- word2vec_train.py：通过使用大规模文本预料训练word2vec模型<br/>
		python word2vec_train.py

- word2vec_use.py：通过已训练好的word2vec模型将分好的词转换成词向量<br/>
		每一个词对应一个1\*255长度的向量，将句子中的所有词累加取平均值作为一个句子的向量<br/>
		python word2vec_use.py avg
		每一个词对应一个1\*255长度的向量，将句子中的n个词拼接作为一个句子的向量（n\*255）<br/>
		python word2vec_use.py all




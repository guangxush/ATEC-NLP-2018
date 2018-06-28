# ATEC-NLP-2018 （文本相似度计算）

## Requirement
Python 2.7 </br>
TensorFlow 1.5 </br>
jieba 0.39 </br>
keras 2.x </br>

## Run
ATEC-NLP-2018目录下的文件为各种不同的ML/DL模型代码，运行方式如下：

- 训练并产生训练模型，训练模型保存在models中：

		CUDA_VISIBLE_DEVICES=0 python XXX.py


## Score

|模型/方法	|Train f1|Dev f1|Pubulic Test A f1|Pubulic Test B f1|备注说明|
|---------|:---:|:----:|:--:|:--:|------|
|cosine距离 |0.4543|0.4189|0.4271|0.4271|直接用文本计算欧式距离|
|word2vec+cosine距离 |0.4543|0.4189|0.3939|0.3939|word2vec之后计算欧式距离|
|word2vec+MLP |0.3643|0.4189|0.3939|0.3939|词向量+MLP二分类|
|features+MLP |0.4836|0.4189|0.3939|0.00|特征+MLP二分类|
|embedding+lstm|0.4836|0.4189|0.3939|0.00|词向量+lstm+mlp二分类|
|embedding+cnn|0.4836|0.4189|0.3939|0.00|词向量+cnn+mlp二分类|


## 评价指标
precision rate = TP / (TP + FP)

recall rate = TP / (TP + FN)

accuracy = (TP + TN) / (TP + FP + TN + FN)

F1-score = 2 * precision rate * recall rate / (precision rate + recall rate)


## 修改日志
1.修复了数据格式[0, 1]的问题</br>
2.修复了学习率太小损失函数维NAN的问题</br>
3.影响准确率为1的因素有：准确率函数定义错误，模型没有更新</br>
4.修改准确率为f1,使用f1进行评分</br>
5.修复了.gitignore导致大文件上传的问题</br>
6.修复了由于微信模型太大导致embedding参数过多的问题</br>

## 修改任务
-- 1.词向量训练方法：根据给出的预料单独训练模型，模型中padding用0向量填充，未找到的词用一个随机生成的固定向量
输入是文本的token,训练5个epoch，embedding的时候用你训练好的词向量初始化，还会继续训练</br>
-- 2.sim_lstm还可以继续改进，最后一层Dense可以修改为计算两个向量的相似度，然后自定义损失函数，自定义层</br>
-- 3.可以把mlp得到的分数作为一个特征，然后找的轮子里面生成一些特征，这些特征采用XGboost或者随机森林预测，得到的效果可能会好一些</br>
-- 4.F1还有一些损失函数可以用sklearn得到，也可以自己写<br>






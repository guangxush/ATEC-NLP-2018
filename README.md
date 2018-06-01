# ATEC-NLP-2018 （文本相似度计算）

## Requirement
Python 2.7 </br>
TensorFlow 1.5 </br>
jieba 0.39 </br>

## Run
ATEC-NLP-2018目录下的文件为各种不同的ML/DL模型代码，运行分为两种方式：

- 只训练，不产生提交结果（用于测试代码的正确性和了解模型效果）,例如：

		python XXX.py train

- 训练且产生测试集结果用于提交，例如：

		python XXX.py submit（bash run.sh INPUT_PATH OUTPUT_PATH）



## 测试结果
### 使用XXX方法
|模型/方法	|Train MAE|Dev MAE|Pubulic Test MAE|备注说明|
|---------|:---:|:----:|:--:|------|
|cosine距离 |0.4543|0.4189|0.4271|直接用文本计算欧式距离|
|word2vec+cosine距离 |0.4543|0.4189|0.3939|word2vec之后计算欧式距离|


## 评价指标
precision rate = TP / (TP + FP)

recall rate = TP / (TP + FN)

accuracy = (TP + TN) / (TP + FP + TN + FN)

F1-score = 2 * precision rate * recall rate / (precision rate + recall rate)


## 修改日志
1.修复了数据格式[0, 1]的问题</br>
2.修复了学习率太小损失函数维NAN的问题</br>
3.影响准确率为1的因素有：准确率函数定义错误，模型没有更新





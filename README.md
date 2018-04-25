# ATEC-NLP-2018

## Requirement
Python 2.7/3
Keras 2.x
TensorFlow
sklearn
pandas

## Run
UAI-CUP-2017目录下的文件为各种不同的ML/DL模型代码，运行分为两种方式：

- 只训练，不产生提交结果（用于测试代码的正确性和了解模型效果）,例如：

		python embedding_mlp.py train

- 训练且产生测试集结果用于提交，例如：

		python embedding_mlp.py submit



## 测试结果
### 使用XXX方法
|模型/方法	|Train MAE|Dev MAE|Pubulic Test MAE|备注说明|
|---------|:---:|:----:|:--:|------|
|MLP       |0.4543|0.4189|2.2674|不使用One-Hot|
|MLP       |0.4530|0.4184|2.2315|对周几和小时使用One-Hot处理|





## MLP方法说明

### mlp_keras.py 基于keras的MLP模型
- 网络参数：
        [512,512,128,64,32,2/1]
- 运行方法：
        使用句子特征进行训练
		python mlp_keras.py sentences
		使用多维特征进行训练
		python mlp_keras.py features

### mlp_with_features.py 基于tensorflow,使用特征进行判别
- 网络参数：
        [33,512,128,64,32,2]
- 运行方法：
		python mlp_with_features.py
### mlp_with_senetnce.py 基于tensorflow,使用句子向量作为特征进行判别
- 网络参数：
        [512,512,128,64,32,2]

- 运行方法
		mlp_with_senetnce.py
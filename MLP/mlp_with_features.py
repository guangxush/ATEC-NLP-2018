# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from util.dataset import load_data_with_features

# import get_feature as gf

# 定义添加隐含层的函数
def add_layer(inputs, in_size, out_size, keep_prob=1.0, activation_function=None):
    Weights = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
    biases = tf.Variable(tf.zeros([out_size]))
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
        outputs = tf.nn.dropout(outputs, keep_prob)  # 随机失活
    return outputs


# holder变量
x = tf.placeholder(tf.float32, [None, 33], name='x')
y_ = tf.placeholder(tf.float32, [None, 2], name='pre')
keep_prob = tf.placeholder(tf.float32, name='keep_probe')  # 概率

h1 = add_layer(x, 33, 512, keep_prob, tf.nn.relu)
h2 = add_layer(h1, 512, 128, keep_prob, tf.nn.relu)
h3 = add_layer(h2, 128, 64, keep_prob, tf.nn.relu)
h4 = add_layer(h3, 64, 32, keep_prob, tf.nn.relu)

# 输出层
w = tf.Variable(tf.truncated_normal([32, 2], stddev=0.1))
# w = tf.Variable(tf.zeros([32, 2]))
b = tf.Variable(tf.zeros([2]))
y = tf.nn.softmax(tf.matmul(h4, w) + b, name='predic')
tf.add_to_collection('pred_network', y)

# 定义loss,optimizer
# cross_entropy = -tf.reduce_mean(tf.log(y_ * tf.clip_by_value(y, 1e-10, 1.0)))
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
train_step = tf.train.AdagradOptimizer(0.0001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # 高维度的
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 要用reduce_mean

# cost_accum = []
acc_prev = 0
# 读取输入数据
input_X, input_Y = load_data_with_features()

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
m_saver = tf.train.Saver()

for i in range(10000):
    sess.run(train_step, feed_dict={x: input_X, y_: input_Y, keep_prob: 0.75})
    if i % 1000 == 0:
        train_accuracy, out, out2, loss = sess.run([accuracy, y, y_, cross_entropy],
                                                   feed_dict={x: input_X, y_: input_Y, keep_prob: 0.75})
        # train_loss = sess.run(correct_prediction, feed_dict={x:input_X, y_:input_Y, keep_prob:0.75})
        print(i)
        print(train_accuracy)
        print(loss)
        print(out)
        print('-----------')
        # cost_accum.append(train_accuracy)
        '''if np.abs(acc_prev - train_accuracy) < 1e-6:
            break
        acc_prev = train_accuracy'''
        if train_accuracy > 0.9:
            break

m_saver.save(sess, '../models/mlp_model')
sess.close()
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
sess = tf.InteractiveSession()



#定义添加隐含层的函数
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
x = tf.placeholder(tf.float32, [None, 33])
y_ = tf.placeholder(tf.float32, [None, 1])
keep_prob = tf.placeholder(tf.float32)     # 概率

h1 = add_layer(x, 33, 512, keep_prob, tf.nn.relu)
h2 = add_layer(h1, 512, 128, keep_prob, tf.nn.relu)
h3 = add_layer(h2, 128, 64, keep_prob, tf.nn.relu)
h4 = add_layer(h3, 64, 32, keep_prob, tf.nn.relu)

# 输出层
w = tf.Variable(tf.zeros([32, 1]))
b = tf.Variable(tf.zeros([1]))
y = tf.nn.softmax(tf.matmul(h4, w)+b)

# 定义loss,optimizer
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdagradOptimizer(0.35).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))       # 高维度的
acuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))    # 要用reduce_mean

tf.global_variables_initializer().run()
#cost_accum = []
acc_prev = 0

# 读取输入数据
input_file = open('./data/output.txt', 'r')
input_x = []
input_y = []
for line in input_file:
    record = line.split('\t')
    input_x.append(np.array(record[0][1:-1].split(',')))
    input_y.append(np.array(record[-1].split(',')))

input_X = np.array(input_x, dtype=np.float32)
input_Y = np.array(input_y, dtype=np.float32)
print input_X.shape
print input_Y.shape
print input_X.dtype
print input_Y.dtype


for i in range(1000):
    train_step.run({x:input_X, y_:input_Y, keep_prob:0.75})
    if i%10 == 0:
        train_accuracy = acuracy.eval({x:input_X,y_:input_Y,keep_prob:1.0})
        print("step %d,train_accuracy %g"%(i,train_accuracy))
        #cost_accum.append(train_accuracy)
        if np.abs(acc_prev - train_accuracy) < 1e-6:
            break
        acc_prev = train_accuracy


print acuracy.eval({x:input_X, y_:input_Y, keep_prob:1.0})
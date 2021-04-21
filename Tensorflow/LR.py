# -*- coding: utf-8 -*-#
'''
@Project    :   DeepLearning
@File       :   LR.py
@USER       :   ZZZZZ
@TIME       :   2021/4/21 10:57

深度学习一般套路
1. 加载数据
2. 构造输入
3. 搭建网络
4. 计算输出
5. 计算loss
6. 反向传播
7. 打完收工
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)

# ------------------------------------- 加载数据 -------------------------------------
mnist = tf.keras.datasets.mnist
(x_train_all,y_train_all),(x_test,y_test) = mnist.load_data()
x_valid,x_train = x_train_all[:5000],x_train_all[5000:]
y_valid,y_train = y_train_all[:5000],y_train_all[5000:]

# ------------------------------------- 搭建网络 -------------------------------------
# 参数设置
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# 计算图的输入
X = tf.placeholder(tf.float32,[None,784]) # mnist图片尺寸为28*28=784
Y = tf.placeholder(tf.float32,[None,10]) # 0-9共9个数字，10分类问题
# 模型权重
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# ------------------------------------- 计算输出 -------------------------------------
pred = tf.nn.softmax(tf.matmul(X, W) + b)


# ------------------------------------- 计算loss -------------------------------------
# 交叉熵
loss = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(pred), reduction_indices=1))


# ------------------------------------- 反向传播 -------------------------------------
# SGD
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
# 初始化
init = tf.global_variables_initializer()

# ------------------------------------- 网络运行 -------------------------------------
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        avg_loss = 0.
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, l = sess.run([optimizer, loss], feed_dict={X: batch_xs, Y: batch_ys})
            avg_loss += l / total_batch

        if (epoch + 1) % display_step == 0:
            logging.info("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(avg_loss))

    logging.info("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # logging.info("Accuracy:",sess.run([accuracy],feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
    logging.info("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))

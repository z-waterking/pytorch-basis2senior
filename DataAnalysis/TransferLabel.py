# -*- coding: utf-8 -*-#
'''
@Project    :   DeepLearning
@File       :   TransferLabel.py
@USER       :   ZZZZZ
@TIME       :   2020/11/28 19:37
'''

'''
将一列整数标记转化为onehot向量形式（不带表头和行号）
'''
import pandas as pd

t1 = pd.read_csv("../Data/Mnist/Mnist_Train_Label.csv", header = None)
print(len(t1))
t2 = []
for index in t1.iloc[:, 0].tolist():
    onehot = [0] * 10
    onehot[index] = 1
    t2.append(onehot)
t2 = pd.DataFrame(t2, columns=[1] * 10)
print(len(t2))
t2.to_csv("Mnist_Train_Label.csv", sep=',', index=False, header=False)
print(len(t2))
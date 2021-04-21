 [toc]
# DeepLearning

# 说明

本代码库所用的python版本为anaconda包中的3.6
pytorch环境为1.5.1: https://pytorch.org/get-started/previous-versions/

# 什么是好代码？

本库的目标在于写"笨代码"。
其间有两个要素，清晰的代码结构与完善的代码文档。各个目录下都有与此目录相关的README.md，来标明需要进行阅读的材料等。

# 深度学习一般套路

1. 加载数据
2. 构造输入
3. 搭建网络
4. 计算输出
5. 计算loss
6. 反向传播
7. 打完收工

# 环境依赖

主目录下执行：pipreqs . --encoding=utf-8 --force，生成requirements文件

本地执行：pip install -r requirements，安装所有的依赖

# Data

1. 抽象基类：https://blog.csdn.net/lengfengyuyu/article/details/85071835

2. 推荐数据集-Recommender Systems Datasets :https://cseweb.ucsd.edu/~jmcauley/datasets.html#market_bias

3. __init__.py的作用：https://www.cnblogs.com/tp1226/p/8453854.html

## 1. Criteo

```
# Criteo数据集-(nums, category) : label
1. 训练数据的格式：
    ['i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9', 'i10', 'i11', 'i12', 'i13'，
    'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13',
    'c14', 'c15', 'c16', 'c17', 'c18', 'c19', 'c20', 'c21', 'c22', 'c23', 'c24', 'c25', 'c26']
    i 代表数值类型的特征
    c 代表ID类型的特征
```

## 2. Mnist

```
# MINST-纯分类数据-(nums) : label
## MINST相关信息
1. 地址：http://yann.lecun.com/exdb/mnist/
    原版四个文件本地大小：
    * train-images-idx3-ubyte     :   47M
    * train-labels-idx3-ubyte     :   60K
    * t10k-images-idx3-ubyte      :   7.8M
    * t10k-labels-idx1-ubyte      :   10K
    
2. 处理MINST数据集：https://www.jianshu.com/p/e7c286530ab9
    由于原版数据集太大，不便于上传github。且其格式无法直接利用。
    因此将其进行采样，并处理为方便观看的向量格式，以作为各深度学习算法的基础数据源。
    处理结果分为四个csv文件：
    1. Mnist_Train_Data.csv
    2. Mnist_Train_Label.csv
    3. Mnist_Test_Data.csv
    4. Mnist_Test_Label.csv
```

## 3. MovieLens

pass

# DataAnalysis

## 1. Pandas对数据进行分析



## 2. Matplotlib对分析结果进行展示


## 3. PyEcharts对分析结果进行展示

# DataFlow

```
数据中间商

通过此包将Data中提供的数据向模型中进行batch或stream方式地传递

1. python生成器：https://www.cnblogs.com/liangmingshen/p/9706181.html
2. yield用法：https://blog.csdn.net/mieleizhi0522/article/details/82142856
4. 解决python模块调用时代码中使用相对路径访问的文件，提示文件不存在的问题:https://blog.csdn.net/cxx654/article/details/79371565
```

# Pytorch

**文件夹命名："{} / {}_{}".format(模型名, 模型名, 数据名)**

* pytorch教程：http://pytorch123.com/

* pytorch-handbook: https://github.com/zergtant/pytorch-handbook

* pytorch-tutorial: https://github.com/yunjey/pytorch-tutorial

# Tensorflow



# Tensorflow2

```
学习资料
1. 简单粗暴Tensorflow：https://tf.wiki/zh_hans
2. 动手学深度学习：https://zh.d2l.ai/index.html
```
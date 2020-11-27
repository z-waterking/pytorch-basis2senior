# MINST-纯分类数据-(nums) : label
## MINST相关信息
1. 地址：http://yann.lecun.com/exdb/mnist/

    a. 原版四个文件本地大小：
    
    * train-images-idx3-ubyte     :   47M
    
    * train-labels-idx3-ubyte     :   60K
    
    * t10k-images-idx3-ubyte      :   7.8M
    
    * t10k-labels-idx1-ubyte      :   10K
    
2. 处理MINST数据集：https://www.jianshu.com/p/e7c286530ab9

    由于原版数据集太大，不便于上传github。且其格式无法直接利用。
    
    因此将其进行采样，并处理为方便观看的向量格式，以作为各深度学习算法的基础数据源。
    
    处理结果分为四个csv文件：
    
    1. MNIST_Train_Data.csv
    
    2. MNIST_Train_Label.csv
    
    3. MNIST_Test_Data.csv
    
    4. MNIST_Test_Label.csv
    
3. Python模块中包__init__.py的作用：https://www.cnblogs.com/maseng/p/3580696.html
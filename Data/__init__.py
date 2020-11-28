# -*- coding: utf-8 -*-#
'''
@Project    :   DeepLearning
@File       :   __init__.py
@USER       :   ZZZZZ
@TIME       :   2020/11/27 10:05
'''
import sys
import os

'''
import all relevant Data Class
'''
from Data.Mnist.Mnist import Mnist
from Data.Criteo.Criteo import Criteo

'''
Insert path to sys.path
'''
DataDirs = ['Criteo', 'Mnist', 'MovieLens']
MainDirName = os.path.dirname(os.path.abspath(__file__))
for data_dir in DataDirs:
    sys.path.append(os.path.join(MainDirName, data_dir) + '/')
print(sys.path)
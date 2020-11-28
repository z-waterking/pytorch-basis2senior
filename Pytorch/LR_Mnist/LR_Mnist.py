# -*- coding: utf-8 -*-#
'''
@Project    :   DeepLearning
@File       :   LR_Mnist.py
@USER       :   ZZZZZ
@TIME       :   2020/11/26 22:25
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
from Data import *
from DataFlow import DataBatchFlow

batch_size = 200
learning_rate = 0.01
epochs = 10

print(torch.__version__)
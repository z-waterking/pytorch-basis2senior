# -*- coding: utf-8 -*-#
'''
@Project    :   DeepLearning
@File       :   DataAcquisition.py
@USER       :   ZZZZZ
@TIME       :   2020/11/26 11:14
'''
import abc

class DataAcquisition(abc.ABC):
    '''
    数据获取的抽象基类，所有数据均需实现此中的抽象方法。
    主要包含：获取训练数据、训练标记、测试数据、测试标记，进行数据长度描述，进行随机数据检查。
    ------------------------------------------------------------------------------
    Abstract base class for data acquisition, all data needs to implement the abstract methods in this.
    It mainly includes: obtaining training data, training marks, test data, and test marks, describing the data length, and checking random data.
    '''
    @abc.abstractmethod
    def GetAllTrainData(cls):
        pass

    @abc.abstractmethod
    def GetAllTrainLabel(self):
        pass

    @abc.abstractmethod
    def GetAllTestData(self):
        pass

    @abc.abstractmethod
    def GetAllTestLabel(self):
        pass

    @abc.abstractmethod
    def DescribeDataLength(self):
        pass

    @abc.abstractmethod
    def CheckRandomData(self):
        pass

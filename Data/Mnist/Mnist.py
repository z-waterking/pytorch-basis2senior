# -*- coding: utf-8 -*-#
"""
@Project    :   DeepLearning
@File       :   Criteo.py
@USER       :   ZZZZZ
@TIME       :   2020/11/25 19:25
"""
from Data.DataAcquisition import DataAcquisition
import numpy as np
import pandas as pd
import os
import random
import logging
import matplotlib.pyplot as plt
#Set loggging level to logging.DEBUG
logging.basicConfig(level=logging.DEBUG)

class Mnist(DataAcquisition):
    def __init__(self):
        '''
        Open All Files and Read All Datas
        '''
        '''
            Using Current path to make these data files can be visited by other .py
        '''
        self.Current_Path = os.path.dirname(__file__)
        logging.debug('Data Path: {}'.format(self.Current_Path))
        #FileNames
        self.Train_Data_FileName = os.path.join(self.Current_Path, "Mnist_Train_Data.csv")
        self.Train_Label_FileName = os.path.join(self.Current_Path, "Mnist_Train_Label.csv")
        self.Test_Data_FileName = os.path.join(self.Current_Path, "Mnist_Test_Data.csv")
        self.Test_Label_FileName = os.path.join(self.Current_Path, "Mnist_Test_Label.csv")

        #File Objects
        self.Train_Data_File = pd.read_csv(self.Train_Data_FileName, sep=',', header = None)
        self.Train_Label_File = pd.read_csv(self.Train_Label_FileName, header = None)
        self.Test_Data_File = pd.read_csv(self.Test_Data_FileName, sep=',', header = None)
        self.Test_Label_File = pd.read_csv(self.Test_Label_FileName, header = None)

        #File Datas
        #Please read the original datas or README.md to understand the process.
        self.Train_Data = self.Train_Data_File.values.tolist()
        self.Train_Label = self.Train_Label_File.values.tolist()
        self.Test_Data = self.Test_Data_File.values.tolist()
        self.Test_Label = self.Test_Label_File.values.tolist()

        #Shape of Train Datas
        self.Data_Shape_Length = 28
        self.Data_Shape_Wide = 28

        #Notation
        logging.info("Please call * GetDataLength() * method to get the length of datas")

    def GetAllTrainData(self):
        '''
        Return all Train_Data
        :return: generator[numpy.array()]
        '''
        return self.Train_Data

    def GetAllTrainLabel(self):
        '''
        Return all Train_Label
        :return: list[]
        '''
        return self.Train_Label

    def GetAllTestData(self):
        '''
        Return all Test_Data
        :return:list[np.array()]
        '''
        return self.Test_Data

    def GetAllTestLabel(self):
        '''
        Return all Test_Label
        :return:list[]
        '''
        return self.Test_Label

    def DescribeDataLength(self):
        '''
        Describe the length of Train Data Set and Test Data Set
        :return:
        '''
        logging.info('The Length of {} is {}'.format(self.Train_Data_FileName, len(self.Train_Data)))
        logging.info('The Length of {} is {}'.format(self.Train_Label_FileName, len(self.Train_Label)))
        logging.info('The Length of {} is {}'.format(self.Test_Data_FileName, len(self.Test_Data)))
        logging.info('The Length of {} is {}'.format(self.Test_Label_FileName, len(self.Test_Label)))

    def CheckRandomData(self, AssignedIndex = None, ToPicture = False):
        '''
        Select one train data and one test data to check if they are true
        :param AssignedIndex: check given index's data
        :param ToPicture: if need picture or not
        :return:
        '''

        #Select a random index respectively
        Random_Train_Index = random.randint(0, len(self.Train_Label))
        Random_Test_Index = random.randint(0, len(self.Test_Label))

        #Select a index which you need to verify
        if AssignedIndex != None:
            Random_Train_Index = AssignedIndex
            Random_Test_Index = AssignedIndex

        #Select relevant datas
        Train_Data_Selected = self.Train_Data[Random_Train_Index]
        Train_Label_Selected = self.Train_Label[Random_Train_Index]
        Test_Data_Selected = self.Test_Data[Random_Test_Index]
        Test_Label_Selected = self.Test_Label[Random_Test_Index]

        #Check Train Data And Test Data
        print('The Selected Train Data Index is {}, train data is as follows:'.format(Random_Train_Index))
        for index, value in enumerate(Train_Data_Selected):
            print(value, end = '\t')
            if index != 0 and index % self.Data_Shape_Length == 0:
                print()
        print('\nThe Train Label is {}'.format(Train_Label_Selected))

        print('The Selected Test Data Index is {}, test data is as follows:'.format(Random_Test_Index))
        for index, value in enumerate(Test_Data_Selected):
            print(value, end='\t')
            if index != 0 and index % self.Data_Shape_Length == 0:
                print()
        print('\nThe Test Label is {}'.format(Test_Label_Selected))

        #If need transfer Array to Image
        if ToPicture == True:
            Train_Data_Image = np.reshape(np.array(Train_Data_Selected), newshape = (self.Data_Shape_Length, self.Data_Shape_Wide))
            Test_Data_Image = np.reshape(np.array(Test_Data_Selected), newshape = (self.Data_Shape_Length, self.Data_Shape_Wide))

            #Using class Image to implement tranformation from array to image
            plt.imsave(self.Current_Path + '/{}_{}.jpg'.format("TrainData", Random_Train_Index), Train_Data_Image)
            plt.imsave(self.Current_Path + '/{}_{}.jpg'.format("TestData", Random_Test_Index), Test_Data_Image)

if __name__ == "__main__":
    mt = Mnist()
    mt.DescribeDataLength()
    mt.CheckRandomData(ToPicture=False)

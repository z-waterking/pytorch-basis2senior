# -*- coding: utf-8 -*-#
"""
@Project    :   DeepLearning
@File       :   Mnist.py
@USER       :   ZZZZZ
@TIME       :   2020/11/25 19:25
"""
import numpy as np
import os
import random
import logging
from PIL import Image
#设置logging级别为debug
logging.basicConfig(level=logging.DEBUG)

class Mnist():
    def __init__(self):
        '''
        Open All Files and Read All Datas
        '''
        #FileNames
        self.Train_Data_FileName = "MNIST_Train_Data.csv"
        self.Train_Label_FileName = "MNIST_Train_Label.csv"
        self.Test_Data_FileName = "MNIST_Test_Data.csv"
        self.Test_Label_FileName = "MNIST_Test_Label.csv"

        #File Objects
        self.Train_Data_File = open(self.Train_Data_FileName, 'r', encoding = 'utf-8')
        self.Train_Label_File = open(self.Train_Label_FileName, 'r', encoding = 'utf-8')
        self.Test_Data_File = open(self.Test_Data_FileName, 'r', encoding = 'utf-8')
        self.Test_Label_File = open(self.Test_Label_FileName, 'r', encoding = 'utf-8')

        #File Datas
        #Please read the original datas or README.md to understand the process.
        self.Train_Data = [np.array(line.split(','), dtype = np.uint8) for line in self.Train_Data_File.readlines()]
        self.Train_Label = [int(line.strip()) for line in self.Train_Label_File.readlines()]
        self.Test_Data = [np.array(line.split(','), dtype = np.uint8) for line in self.Test_Data_File.readlines()]
        self.Test_Label = [int(line.strip()) for line in self.Test_Label_File.readlines()]

        #Shape of Train Datas
        self.Data_Shape_Length = 28
        self.Data_Shape_Wide = 28

        #Notation
        logging.info("Please call * GetFileLength() * method to get the length of datas")

    def GetAllTrainData(self):
        return self.Train_Data

    def GetAllTrainLabel(self):
        return self.Train_Label

    def GetAllTestData(self):
        return self.Test_Data

    def GetAllTestLabel(self):
        return self.Test_Label

    def GetFileLength(self):
        '''
        Describe the length of Train Data Set and Test Data Set
        :return:
        '''
        logging.info('The Length of {} is {}'.format(self.Train_Data_FileName, len(self.Train_Data)))
        logging.info('The Length of {} is {}'.format(self.Train_Label_FileName, len(self.Train_Label)))
        logging.info('The Length of {} is {}'.format(self.Test_Data_FileName, len(self.Test_Data)))
        logging.info('The Length of {} is {}'.format(self.Test_Label_FileName, len(self.Test_Label)))

    def CheckRandomData(self, ToPicture = False):
        '''
        Select one train data and one test data to check if they are true
        :return:
        '''
        #Select a random index respectively
        Random_Train_Index = random.randint(0, len(self.Train_Label))
        Random_Test_Index = random.randint(0, len(self.Test_Label))

        #Select relevant datas
        Train_Data_Selected = self.Train_Data[Random_Train_Index]
        Train_Label_Selected = self.Train_Label[Random_Train_Index]
        Test_Data_Selected = self.Test_Data[Random_Test_Index]
        Test_Label_Selected = self.Test_Label[Random_Test_Index]

        #Check Train Data And Test Data
        print('The Selected Train Data Index is {}, train data is as follows:'.format(Random_Train_Index))
        for index, value in enumerate(Train_Data_Selected):
            print(value, end = '  ')
            if index != 0 and  index % self.Data_Shape_Length == 0:
                print()
        print('\nThe Train Label is {}'.format(Train_Label_Selected))

        print('The Selected Test Data Index is {}, test data is as follows:'.format(Random_Test_Index))
        for index, value in enumerate(Test_Data_Selected):
            print(value, end='  ')
            if index != 0 and  index % self.Data_Shape_Length == 0:
                print()
        print('\nThe Test Label is {}'.format(Test_Label_Selected))

        #If need transfer Array to Image
        if ToPicture == True:
            Train_Data_Image = np.reshape(Train_Data_Selected, newshape = (self.Data_Shape_Length, self.Data_Shape_Wide))
            Test_Data_Image = np.reshape(Test_Data_Selected, newshape = (self.Data_Shape_Length, self.Data_Shape_Wide))
            #Using class Image to implement tranformation from array to image
            Image.fromarray(Train_Data_Image).save('{}_{}.png'.format("TrainData", Random_Train_Index))
            Image.fromarray(Test_Data_Image).save('{}_{}.png'.format("TestData", Random_Test_Index))

if __name__ == "__main__":
    tm = Mnist()
    tm.GetFileLength()
    tm.CheckRandomData(True)

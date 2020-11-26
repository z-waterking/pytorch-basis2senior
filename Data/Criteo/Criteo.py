# -*- coding: utf-8 -*-#
"""
@Project    :   DeepLearning
@File       :   Criteo.py
@USER       :   ZZZZZ
@TIME       :   2020/11/25 19:25
"""
from Data.DataAcquisition import DataAcquisition
import random
import logging
import pandas as pd
#Set loggging level to logging.DEBUG
logging.basicConfig(level=logging.DEBUG)

class Criteo(DataAcquisition):
    def __init__(self):
        '''
        Open All Files and Read All Datas
        '''
        #FileNames
        self.Train_Data_FileName = "CRITEO_Train_Data.csv"
        self.Train_Label_FileName = "CRITEO_Train_Label.csv"
        self.Test_Data_FileName = "CRITEO_Test_Data.csv"
        self.Test_Label_FileName = "CRITEO_Test_Label.csv"

        #Data Columns
        self.DataNumColumns = ['i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9', 'i10', 'i11', 'i12', 'i13']
        self.DataCateColumns = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13',
                                'c14', 'c15', 'c16', 'c17', 'c18', 'c19', 'c20', 'c21', 'c22', 'c23', 'c24', 'c25', 'c26']
        self.DataColumns = self.DataNumColumns + self.DataCateColumns

        #File Objects
        self.Train_Data_File = pd.read_csv(self.Train_Data_FileName, sep = ',')
        self.Train_Label_File = pd.read_csv(self.Train_Label_FileName)
        self.Test_Data_File = pd.read_csv(self.Test_Data_FileName, sep = ',')
        self.Test_Label_File = pd.read_csv(self.Test_Label_FileName)

        #File Datas
        #Please read the original datas or README.md to understand the process.
        self.Train_Data = self.Train_Data_File.values.tolist()
        #Get the Train Label's First Column's value
        self.Train_Label = self.Train_Label_File[0].values.tolist()

        self.Test_Data = self.Test_Data_File.values.tolist()
        # Get the Test Label's First Column's value
        self.Test_Label = self.Test_Label_File[0].values.tolist()

        #Notation
        logging.info("Please call * GetFileLength() * method to get the length of datas")

    def GetAllTrainData(self):
        '''
        Return all Train_Data
        :return: list[numpy.array()]
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

    def GetFileLength(self):
        '''
        Describe the length of Train Data Set and Test Data Set
        :return:
        '''
        logging.info('The Length of {} is {}'.format(self.Train_Data_FileName, len(self.Train_Data)))
        logging.info('The Length of {} is {}'.format(self.Train_Label_FileName, len(self.Train_Label)))
        logging.info('The Length of {} is {}'.format(self.Test_Data_FileName, len(self.Test_Data)))
        logging.info('The Length of {} is {}'.format(self.Test_Label_FileName, len(self.Test_Label)))

    def CheckRandomData(self):
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
        print('The Selected Train Data Index is {}'.format(Random_Train_Index))
        print('The Train Data is {}'.format(Train_Data_Selected))
        print('The Train Label is {}'.format(Train_Label_Selected))

        print('The Selected Test Data Index is {}, test data is as follows:'.format(Random_Test_Index))
        print('The Test Data is {}'.format(Test_Data_Selected))
        print('The Test Label is {}'.format(Test_Label_Selected))

if __name__ == "__main__":
    tm = Criteo()
    tm.GetFileLength()
    tm.CheckRandomData()

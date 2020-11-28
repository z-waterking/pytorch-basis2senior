# -*- coding: utf-8 -*-#
'''
@Project    :   DeepLearning
@File       :   DataFlow.py
@USER       :   ZZZZZ
@TIME       :   2020/11/28 14:30
'''
import logging
import random
from Data.Mnist import Mnist
logging.basicConfig(level=logging.DEBUG)

class DataFlow():
    def __init__(self, DataSource):
        '''
        Get Datas from DataSource By Different Ways.
        :param DataSource: Need to be the Object: DataAcquisition
        '''
        #Get the Info of Data, and output the data length.
        self.DataSource = DataSource
        self.DataSource.GetDataLength()

        #Get the Data
        logging.info("Get Data Start!")

        self.BatchSize = 1
        self.TrainData = self.DataSource.GetAllTrainData()
        self.TrainLabel = self.DataSource.GetAllTrainLabel()
        self.TestData = self.DataSource.GetAllTestData()
        self.TestLabel = self.DataSource.GetAllTestLabel()

        #Get the length of Data
        self.TrainDataLen = len(self.TrainData)
        self.TestDataLen = len(self.TestData)

        logging.info("Get Data End!")

        logging.info("Shuffle the Data And Label Simultaneously Start!")

        logging.info("Shuffle Train Data")

        TempTrainDataLabel = list(zip(self.TrainData, self.TrainLabel))
        random.shuffle(TempTrainDataLabel)
        self.TrainData[:], self.TrainLabel[:] = zip(*TempTrainDataLabel)

        logging.info("Shuffle TestData")

        TempTestDataLabel = list(zip(self.TrainData, self.TrainLabel))
        random.shuffle(TempTestDataLabel)
        self.TestData[:], self.TestLabel[:] = zip(*TempTestDataLabel)

        logging.info("Shuffle the Data And Label Simultaneously End!")

        logging.info("The default BatchSize is {}, Please Set the Data BatchSize!".format(self.BatchSize))

    def SetBatchSize(self, BatchSize):
        '''
        :param BatchSize: A num which is smaller than DataSize and larger than 0
        :return:None
        '''

        #Judge the if the argument is valid
        ValidArg = True
        if not isinstance(BatchSize, int):
            logging.error("BatchSize must be a Interger!")
            ValidArg = False
        else:
            if BatchSize <= 0:
                logging.error("Batch Size need to be larger than 0!")
            else:
                logging.error("Batch Size need to be smaller than the data length!")
        if ValidArg == True:
            self.BatchSize = BatchSize

    def GetBatchSize(self):
        return self.BatchSize

    def GetTrainDataBatch(self):
        pass

    def GetTrainLabelBatch(self):
        pass

    def GetTestDataBatch(self):
        pass

    def GetTestLabelBatch(self):
        pass

if __name__ == "__main__":
    pass

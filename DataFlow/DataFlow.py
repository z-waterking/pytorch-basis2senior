# -*- coding: utf-8 -*-#
'''
@Project    :   DeepLearning
@File       :   DataFlow.py
@USER       :   ZZZZZ
@TIME       :   2020/11/28 14:30
'''
import logging
import random
from Data import *
logging.basicConfig(level=logging.DEBUG)

class DataFlow():
    def __init__(self, DataSource):
        '''
        Get Datas from DataSource By Different Ways.
        :param DataSource: Need to be the Object: DataAcquisition
        '''
        #Get the Info of Data, and output the data length.
        self.DataSource = DataSource
        self.DataSource.DescribeDataLength()

        #Get the Data And Shuffle them
        logging.info("Get Data Start!")
        self.TrainData = self.DataSource.GetAllTrainData()
        self.TrainLabel = self.DataSource.GetAllTrainLabel()
        self.TestData = self.DataSource.GetAllTestData()
        self.TestLabel = self.DataSource.GetAllTestLabel()
        self.ShuffleData()

        #Get the length of Data
        self.TrainDataLen = len(self.TrainData)
        self.TestDataLen = len(self.TestData)
        logging.info("Get Data End!")

        #Set the BatchSize as default 1.
        self.MaxBatchSize = max(self.TrainDataLen, self.TestDataLen)
        self.BatchSize = 1
        logging.info("The default BatchSize is {}, Please Set the Data BatchSize!".format(self.BatchSize))

    def ShuffleData(self):
        '''
            Shuffle the Data And Label Simultaneously
        :return: None
        '''
        logging.info("Shuffle the Data And Label Simultaneously Start!")

        logging.info("Shuffle Train Data")
        TempTrainDataLabel = list(zip(self.TrainData, self.TrainLabel))
        random.shuffle(TempTrainDataLabel)
        self.TrainData[:], self.TrainLabel[:] = zip(*TempTrainDataLabel)

        logging.info("Shuffle TestData")
        TempTestDataLabel = list(zip(self.TestData, self.TestLabel))
        random.shuffle(TempTestDataLabel)
        self.TestData[:], self.TestLabel[:] = zip(*TempTestDataLabel)

        logging.info("Shuffle the Data And Label Simultaneously End!")

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
            elif BatchSize > self.MaxBatchSize:
                logging.error("Batch Size need to be smaller than the data length!")
        if ValidArg == True:
            self.BatchSize = BatchSize
        else:
           raise Exception("BatchSize Invalid!")

        #ReCut the data
        self.CutDataForGenerator()

    def GetBatchSize(self):
        return self.BatchSize

    def CutDataForGenerator(self):
        '''
        Cut the datas by BatchSize
        if BatchSize == 2:
            [a, b, c, d] ---> [[a, b], [c, d]]
        :return: None
        '''
        self.TrainData = [self.TrainData[CutStart : CutStart + self.BatchSize] for CutStart in range(0, len(self.TrainData), self.BatchSize)]
        self.TrainLabel = [self.TrainLabel[CutStart: CutStart + self.BatchSize] for CutStart in range(0, len(self.TrainLabel), self.BatchSize)]

        self.TestData = [self.TestData[CutStart: CutStart + self.BatchSize] for CutStart in range(0, len(self.TestData), self.BatchSize)]
        self.TestLabel = [self.TestLabel[CutStart: CutStart + self.BatchSize] for CutStart in range(0, len(self.TestLabel), self.BatchSize)]

    def GetTrainDataBatch(self):
        for TrainDataBatch in self.TrainData:
            yield TrainDataBatch

    def GetTrainLabelBatch(self):
        for TrainLabelBatch in self.TrainLabel:
            yield TrainLabelBatch

    def GetTestDataBatch(self):
        for TestDataBatch in self.TestData:
            yield TestDataBatch

    def GetTestLabelBatch(self):
        for TestLabelBatch in self.TestLabel:
            yield TestLabelBatch

if __name__ == "__main__":
    #Construct DataAcquisition Object First
    mt = Mnist()
    df = DataFlow(mt)

    #Transfer DataAcquisition Object as parameter
    # dg = df.GetTrainLabelBatch()
    # TempData = next(dg)
    # print('a batch data len is {}, \nthe data is {}'.format(len(TempData), TempData))
    # #Test the SetBatchSize

    print('-----------------------------')
    df.SetBatchSize(3)
    dg = df.GetTestLabelBatch()
    TempData = next(dg)
    print('set batch size as 3, a batch data len is {}, \nthe data is {}'.format(len(TempData), TempData))
    #Testing the DataFlow
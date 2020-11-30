# -*- coding: utf-8 -*-#
'''
@Project    :   DeepLearning
@File       :   LR.py
@USER       :   ZZZZZ
@TIME       :   2020/11/26 22:25
'''
import torch
import torch.nn as nn
import numpy as np
from Data import *
from DataFlow import DataBatchFlow

epochs = 10

class LR(nn.Module):
    def __init__(self):
        '''
        Must call the super class's __init__ method
        '''
        super(LR, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        '''
        forward deliver
        :return: the output of the net
        '''
        #Linear Input
        out = self.fc(x)
        # #Sigmoid Function
        # out = torch.sigmoid(out)
        return out

if __name__ == "__main__":
    '''
    An Example for calling LR Model
    '''
    #Init Data and DataFlow Class
    MnistData = Mnist()
    DataFlow = DataBatchFlow(MnistData)
    DataFlow.SetBatchSize(2)

    #Construct Model, Loss and Optimizer
    Model = nn.Linear(784, 10)
    Criterion = nn.MSELoss()
    Optimizer = torch.optim.Adam(Model.parameters())

    #Some other parameters
    Epochs = 10

    # Get Test Data
    TestData = np.array(list(DataFlow.GetTestDataBatch()))
    TestLabel = np.array(list(DataFlow.GetTestLabelBatch()))

    #Training the Model
    for epoch in range(10):
        #Set the Model to Train mode
        Model.train()

        # Get the Input Data and Output Data
        TrainDataGenerator = DataFlow.GetTrainDataBatch()
        TrainLabeGenerator = DataFlow.GetTrainLabelBatch()

        #Train Model Once
        try:
            #Reading Generator until it raise Exception
            while True:
                #Input and Output Data must convert to torch.Tensor
                TrainDataBatch = torch.from_numpy(np.array(next(TrainDataGenerator))).float()
                TrainLabelBatch = torch.from_numpy(np.array(next(TrainLabeGenerator))).float()
                TrainOutput = Model(TrainDataBatch)

                #Computing Loss
                Loss = Criterion(TrainOutput, TrainLabelBatch)

                #Clear Gradient
                Optimizer.zero_grad()
                Loss.backward()
                Optimizer.step()

        except Exception as e:
            print(epoch, e)

        #Evaluate the Performance of Model
        DataAllNums = 0
        DataCorrectNums = 0
        for (TData, TLabel) in zip(TestData, TestLabel):
            TData = torch.from_numpy(np.array(TData)).float()
            TLabel = torch.from_numpy(np.array(TLabel)).float()
            TOutput = Model(TData)
            if TOutput.equal(TLabel):
                DataCorrectNums += 1
            DataAllNums += 1
        print(DataCorrectNums / DataAllNums)

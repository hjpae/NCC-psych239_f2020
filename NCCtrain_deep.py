import json
import numpy as np
import torch
import random
import tensorboardX
import matplotlib as plt
from torch.utils.data import Dataset, DataLoader, Sampler
from torch import nn
from tensorboardX import SummaryWriter

from NCCTest_deep import testNCC # check script name

class NCC(nn.Module):
    def __init__(self, trainSplitRatio=0.7, sizeEmbLayer=100, sizeClassfLayer=100, dropOutRatio=0.75,):
        super(NCC, self).__init__()
        self.trainSplitRatio = trainSplitRatio
        self.sizeEmbLayer = sizeEmbLayer
        self.sizeClassfLayer = sizeClassfLayer
        self.dropOutRatio = dropOutRatio

        self.e1Linear1 = nn.Linear(2, self.sizeEmbLayer)
        self.e1BatchNorm1 = nn.BatchNorm1d(self.sizeEmbLayer)
        self.e1ReLu1 = nn.ReLU()
        self.e2Drouput1 = nn.Dropout(self.dropOutRatio)
        
        self.e1Linear2 = nn.Linear(self.sizeEmbLayer, self.sizeEmbLayer)
        self.e1BatchNorm2 = nn.BatchNorm1d(self.sizeEmbLayer)
        self.e1ReLu2 = nn.ReLU()
        self.e2Drouput2 = nn.Dropout(self.dropOutRatio)

        self.classLayer = nn.Sequential(
            nn.Linear(self.sizeEmbLayer, self.sizeClassfLayer),
            nn.BatchNorm1d(self.sizeClassfLayer),
            nn.ReLU(),
            nn.Dropout(self.dropOutRatio),
            nn.Linear(self.sizeClassfLayer, self.sizeClassfLayer),
            nn.BatchNorm1d(self.sizeClassfLayer),
            nn.ReLU(),
            nn.Dropout(self.dropOutRatio),
        )
        
        self.deepclass = nn.Sequential(
            nn.Linear(self.sizeEmbLayer, self.sizeClassfLayer),
            nn.BatchNorm1d(self.sizeClassfLayer),
            nn.ReLU(),
            nn.Dropout(self.dropOutRatio),
            nn.Linear(self.sizeClassfLayer, self.sizeClassfLayer),
            nn.BatchNorm1d(self.sizeClassfLayer),
            nn.ReLU(),
            nn.Dropout(self.dropOutRatio),
            nn.Linear(self.sizeClassfLayer, self.sizeClassfLayer),
            nn.BatchNorm1d(self.sizeClassfLayer),
            nn.ReLU(),
            nn.Dropout(self.dropOutRatio),
            nn.Linear(self.sizeClassfLayer, self.sizeClassfLayer),
            nn.BatchNorm1d(self.sizeClassfLayer),
            nn.ReLU(),
            nn.Dropout(self.dropOutRatio), # 4*4
        )
        
        self.logits = nn.Linear(self.sizeClassfLayer, 1)

    def forward(self, xVal, yVal):
        xyval = torch.cat([xVal, yVal], 2)
        BatchSize = xyval.shape[0]
        DataSize = xyval.shape[1]
        
        # deeper embed layer 
        # need to define one by one to keep dimension
        e1L1 = self.e1Linear1(xyval).view(BatchSize, self.sizeEmbLayer, DataSize)
        e1B1 = self.e1BatchNorm1(e1L1).view(BatchSize, DataSize, self.sizeEmbLayer)
        e1R1 = self.e1ReLu1(e1B1)
        e1D1 = self.e2Drouput1(e1R1) #1

        e1L2 = self.e1Linear2(e1D1).view(BatchSize, self.sizeEmbLayer, DataSize)
        e1B2 = self.e1BatchNorm2(e1L2).view(BatchSize, DataSize, self.sizeEmbLayer)                        
        e1R2 = self.e1ReLu2(e1B2)
        e1D2 = self.e2Drouput2(e1R2) #2

        e1L3 = self.e1Linear2(e1D2).view(BatchSize, self.sizeEmbLayer, DataSize)
        e1B3 = self.e1BatchNorm2(e1L3).view(BatchSize, DataSize, self.sizeEmbLayer)                        
        e1R3 = self.e1ReLu2(e1B3)
        e1D3 = self.e2Drouput2(e1R3) #3

        e1L4 = self.e1Linear2(e1D3).view(BatchSize, self.sizeEmbLayer, DataSize)
        e1B4 = self.e1BatchNorm2(e1L4).view(BatchSize, DataSize, self.sizeEmbLayer)                        
        e1R4 = self.e1ReLu2(e1B4)
        e1D4 = self.e2Drouput2(e1R4) #4

        e1L5 = self.e1Linear2(e1D4).view(BatchSize, self.sizeEmbLayer, DataSize)
        e1B5 = self.e1BatchNorm2(e1L5).view(BatchSize, DataSize, self.sizeEmbLayer)                        
        e1R5 = self.e1ReLu2(e1B5)
        e1D5 = self.e2Drouput2(e1R5) #5

        e1L6 = self.e1Linear2(e1D5).view(BatchSize, self.sizeEmbLayer, DataSize)
        e1B6 = self.e1BatchNorm2(e1L6).view(BatchSize, DataSize, self.sizeEmbLayer)                        
        e1R6 = self.e1ReLu2(e1B6)
        e1D6 = self.e2Drouput2(e1R6) #6

        e1L7 = self.e1Linear2(e1D6).view(BatchSize, self.sizeEmbLayer, DataSize)
        e1B7 = self.e1BatchNorm2(e1L7).view(BatchSize, DataSize, self.sizeEmbLayer)                        
        e1R7 = self.e1ReLu2(e1B7)
        e1D7 = self.e2Drouput2(e1R7) #7

        e1L8 = self.e1Linear2(e1D7).view(BatchSize, self.sizeEmbLayer, DataSize)
        e1B8 = self.e1BatchNorm2(e1L8).view(BatchSize, DataSize, self.sizeEmbLayer)                        
        e1R8 = self.e1ReLu2(e1B8)
        e1D8 = self.e2Drouput2(e1R8) #8
        
        # mean of embedlayer
        finalEmbLayer = torch.mean(e1D4, 1)
        
        # deeper classification layer
        deepclass1 = self.deepclass(finalEmbLayer) #8+4=12
        deepclass2 = self.deepclass(deepclass1) #8+4+4=16
        
        # softmax output 
        logits = self.logits(deepclass2)
        prob = torch.sigmoid(logits)
        return logits, prob

def returnTorch(listObj):
    XList = []
    YList = []
    LabelList = []
    for obj in listObj:
        tempX = np.array(obj["trainX"])
        tempX = tempX[np.newaxis, ...]
        XList.append(tempX)

        tempY = np.array(obj["trainY"])
        tempY = tempY[np.newaxis, ...]
        YList.append(tempY)

        tempLabel = np.array([obj["label"]])
        tempLabel = tempLabel[np.newaxis, ...]
        LabelList.append(tempLabel)

    ## format the data into np.float32 and single array
    X = np.concatenate(XList, axis=0)
    Y = np.concatenate(YList, axis=0)
    Label = np.concatenate(LabelList, axis=0)
    X = X[..., np.newaxis]
    Y = Y[..., np.newaxis]
    return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(Label)

if __name__ == '__main__':
    batchSize = 256 # whole data size = 30000 / minibatch size 2n = 32
    fileName = "./data/causal-data-gen-30K.json-original"
    trainSplitRatio = 0.7
    iterVal = 25 # epoch iteration
    intLrRate = 0.0001

    ''' *************************  '''
    with open(fileName, 'r') as ReadFile:
        dataset = {}
        for line in ReadFile:
            data = json.loads(line)
            if data["size"] not in dataset:
                dataset[data["size"]] = [data]
            else:
                dataset[data["size"]].append(data)
    train_dataset = {}
    test_dataset = {}

    for size, data in dataset.items():
        random.shuffle(data)
        idx = int(np.floor(trainSplitRatio * len(data)))
        train_dataset[size] = data[:idx]
        test_dataset[size] = data[idx:]

    ''' *************************  '''

    model = NCC()
    model = model.cuda()

    writer = SummaryWriter('runs_new/NCCdeep16')

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.RMSprop(params=model.parameters(), lr=intLrRate)
    ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    model.train()
    test_lowest_loss = 1
    count = 0
    highest_acc = 0
    for itr in range(iterVal):
        model.train()
        for size, data in train_dataset.items():
            random.shuffle(data) # batch shuffle(?)
            for idx in range(0, len(data), batchSize):
                trainX, trainY, labels = returnTorch(data[idx: idx + batchSize])

                optimizer.zero_grad()

                logits, prob = model(trainX.cuda().float(), trainY.cuda().float())
                loss = criterion(prob.cuda().float(), labels.cuda().float())
                writer.add_scalar('Train loss', loss.item(), count)
                loss.backward()
                ExpLR.step()

                print("itr: ", itr, "count: ", count, "loss: ", loss)
                count += 1

        with torch.no_grad():
            model.eval()
            losslist = []
            for size, data in test_dataset.items():
                for idx in range(0, len(data), batchSize):
                    testX, testY, labels = returnTorch(data[idx: idx + batchSize])
                    logits, prob = model(testX.cuda().float(), testY.cuda().float())
                    loss = criterion(prob.cuda().float(), labels.cuda().float())
                    losslist.append(loss.item())
            print("itr: ", itr, "loss: ", np.mean(losslist))
            writer.add_scalar("Test loss", np.mean(losslist), itr)
            
            if np.mean(losslist) < test_lowest_loss:
                test_lowest_loss = np.mean(losslist)
                print("test_lowest_loss:", test_lowest_loss, " saving model ..")
                torch.save(model, './model/NCCdeep16.pt')
                writer.add_scalar("Tubingen Test accuracy", testNCC(), itr)
                
                if testNCC() > highest_acc:
                    torch.save(model, './model/NCCdeep16.pt')

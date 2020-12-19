import json
import numpy as np
import torch
import random
import tensorboardX
import matplotlib as plt
from torch.utils.data import Dataset, DataLoader, Sampler
from torch import nn
from tensorboardX import SummaryWriter

from NCCTest_resconn import testNCC # check script name

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

        self.embedLayer = nn.Sequential(
            nn.Linear(2, self.sizeEmbLayer),
            nn.ReLU(),
            nn.Dropout(self.dropOutRatio),
            nn.Linear(self.sizeEmbLayer, self.sizeEmbLayer),
            nn.ReLU(),
            nn.Dropout(self.dropOutRatio),
        )
        
        self.resembedLayer = nn.Sequential(
            nn.Linear(self.sizeEmbLayer, self.sizeEmbLayer),
            nn.ReLU(),
            nn.Dropout(self.dropOutRatio),
        )

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

        self.resclassLayer = nn.Sequential(
            nn.Linear(self.sizeEmbLayer, self.sizeClassfLayer),
            nn.BatchNorm1d(self.sizeClassfLayer),
            nn.ReLU(),
            nn.Dropout(self.dropOutRatio),
        )

        self.logits = nn.Linear(self.sizeClassfLayer, 1)

    def forward(self, xVal, yVal):
        xyval = torch.cat([xVal, yVal], 2)
        BatchSize = xyval.shape[0]
        DataSize = xyval.shape[1]
        x = self.e1Linear1(xyval).view(BatchSize, self.sizeEmbLayer, DataSize)
        x = self.e1BatchNorm1(x).view(BatchSize, DataSize, self.sizeEmbLayer)
        x = self.e1ReLu1(x)
        x = self.e2Drouput1(x)
        
        # residual block - embedded layer 
        res_emb = x
        x = self.resembedLayer(x)
        x = x + res_emb
        out = self.e1ReLu1(x)
        
        # averaging output from embed layer 
        out_emb = torch.mean(out, 1)

        k = self.classLayer(out_emb)

        # residual block - classification layer
        res_class = k
        k = self.resclassLayer(k)
        k = k + res_class
        out_c = self.e1ReLu1(k)

        # softmax layer(logit to prob) - output layer 
        logits = self.logits(out_c)
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
    fileName = "C:/Users/Kardien/Documents/python/psych239_f2020/data/causal-data-gen-30K.json-original"
    trainSplitRatio = 0.7
    iterVal = 25 # should be 10000 ... 85.3 epochs
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

    writer = SummaryWriter('runs/NCCres2')

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
                print("test_lowest_lost:", test_lowest_loss, " saving model ..")
                torch.save(model, './model/NCC_model_res2.pt')
                writer.add_scalar("Tubengen accuracy", testNCC(), itr)

                if testNCC() > highest_acc:
                    torch.save(model, './model/NCC_model_res2.pt')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter

from sklearn.metrics import confusion_matrix

#%% creating personal dataloader

train_mnist = pd.read_csv("../input/fashion-mnist_train.csv")
test_mnist = pd.read_csv("../input/fashion-mnist_test.csv")

class FashionDataset(Dataset):
   
    def __init__(self, data, transform = None):
        """Method to initilaize variables.""" 
        self.MNIST = list(data.values)
        self.transform = transform
        
        label = []
        image = []
        
        for i in self.MNIST:
            label.append(i[0]) # first column
            image.append(i[1:])
        self.labels = np.asarray(label)
        # dim = 28 * 28 * 1 = h * w * color_channels
        self.images = np.asarray(image).reshape(-1, 28, 28, 1).astype('float32')

    def __getitem__(self, index):
        label = self.labels[index]
        image = self.images[index]
        
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)

# transform data into tensor, range from 0 to 1
train_set = FashionDataset(train_mnist, transform=transforms.Compose([transforms.ToTensor()]))
test_set = FashionDataset(test_mnist, transform=transforms.Compose([transforms.ToTensor()]))

train_loader = DataLoader(train_set, batch_size=100)
test_loader = DataLoader(train_set, batch_size=100) # parsing into minibatch


#%% dataloader using torchvision 

train_set = torchvision.datasets.FashionMNIST("./data", download=True, train=True, transform=
                                                transforms.Compose([transforms.ToTensor()]))
test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=
                                               transforms.Compose([transforms.ToTensor()]))              

train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100)

#%% define the label 
def output_label(label):
    output_mapping = {
                 0: "T-shirt/Top",
                 1: "Trouser",
                 2: "Pullover",
                 3: "Dress",
                 4: "Coat", 
                 5: "Sandal", 
                 6: "Shirt",
                 7: "Sneaker",
                 8: "Bag",
                 9: "Ankle Boot"
                 }
    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]


#%% confirm the data with imshow() 
a = next(iter(train_loader))
print(a[0].size())

print(len(train_set))

image, label = next(iter(train_set))
plt.imshow(image.squeeze(), cmap="gray")
print(label)
print(output_label(label))

#%% CNN Model 
class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out

model = CNN()
model = model.cuda()

writer = SummaryWriter('runs/CNNmnist')

error = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print(model)



#%% LSTM model 



#%% NCC model 

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
        self.logits = nn.Linear(self.sizeClassfLayer, 1)

    def forward(self, xVal, yVal):
        xyval = torch.cat([xVal, yVal], 2)
        BatchSize = xyval.shape[0]
        DataSize = xyval.shape[1]
        e1L1 = self.e1Linear1(xyval).view(BatchSize, self.sizeEmbLayer, DataSize)
        e1B1 = self.e1BatchNorm1(e1L1).view(BatchSize, DataSize, self.sizeEmbLayer)
        e1R1 = self.e1ReLu1(e1B1)
        e1D1 = self.e2Drouput1(e1R1)
        e1L2 = self.e1Linear2(e1D1).view(BatchSize, self.sizeClassfLayer, DataSize)
        e1B2 = self.e1BatchNorm2(e1L2).view(BatchSize, DataSize, self.sizeClassfLayer)                        
        e1R2 = self.e1ReLu2(e1B2)
        e1D2 = self.e2Drouput2(e1R2)

        finalEmbLayer = torch.mean(e1D2, 1)
        # r = self.embedLayer(xyval)
        # finalEmbLayer = torch.mean(r, 1)

        classLayer = self.classLayer(finalEmbLayer)
        logits = self.logits(classLayer)
        prob = torch.sigmoid(logits)
        return logits, prob

model = NCC()
model = model.cuda()

writer = SummaryWriter('runs/NCCmnist')

error = torch.nn.BCELoss()
optimizer = torch.optim.RMSprop(params=model.parameters(), lr=0.001)
print(model)

model.train()
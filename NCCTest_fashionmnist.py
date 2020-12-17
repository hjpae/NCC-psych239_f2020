import torch
import json
import numpy as np
from torch import nn
import pickle

from NCCtrain_linear import NCC
from NCCtrain_linear import returnTorch

def testNCC():
    tubDataset = "./data/tubehengenDataFormat.json"
    model = torch.load('./model/NCC_model_final.pt')
    model.eval()
    with torch.no_grad():
        with open(tubDataset, "r") as tubDataReader:
            count = 0
            correct = 0
            for line in tubDataReader:
                data = json.loads(line)
                testInputX, testInputY, testLabel = returnTorch([data])
                _, prob = model(testInputX.cuda().float(), testInputY.cuda().float())

                if prob[0][0] > (1 - prob[0][0]):
                    prediction = 1
                else:
                    prediction = 0
                count += 1

                if prediction == testLabel[0][0]:
                    correct += 1
                else:
                    print("wrong Prediction : prob : %f label : %f" % (prob[0][0], testLabel[0][0]))

                print("count : ", count, "correct : ", correct)
            print("accuracy: ", correct / count)
            return correct / count

classLabel = {'person': 0, 'chair': 1, 'car': 2, 'dog': 3, 'bottle': 4, 'cat': 5, 'bird': 6,
                           'pottedplant': 7, 'sheep': 8, 'boat': 9, 'aeroplane': 10, 'tvmonitor': 11, 'sofa': 12,
                           'bicycle': 13, 'horse': 14, 'motorbike': 15, 'diningtable': 16, 'cow': 17, 'train': 18,
                           'bus': 19}

if __name__ == '__main__':
    testNCC()
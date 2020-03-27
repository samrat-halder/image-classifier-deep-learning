import os
import pandas as pd
import numpy as np
import cv2
import pickle
import random
import time
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from func import *

class LeNet(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3,
                               out_channels = 6,
                               kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 6,
                               out_channels = 10,
                               kernel_size = 5)
        self.conv3 = nn.Conv2d(in_channels = 10,
                               out_channels = 18,
                               kernel_size = 5)

        self.fc1 = nn.Linear(18 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size = 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size = 2)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size = 5)

        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)

        return x

class AlexNet(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), #in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), #kernel_size
            nn.Conv2d(16, 64, 3, padding = 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 256, 3, padding = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, 3, padding = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding = 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(32 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

if __name__=='__main__':
    train_data_folder = './modified_kaggle_HW2/kaggle_train_128/train_128/'
    test_data_folder = './modified_kaggle_HW2/kaggle_test_128/test_128/'
    labels = pd.read_csv('./modified_kaggle_HW2/modified_prediction_labels_template.csv')
    #five output classes
    OUTPUT_DIM=5
    classes = load_labels()
    trainset = []
    with open('trainset.pkl', 'rb') as f:
      for class_ in classes:
        trainset.append(pickle.load(f))
    trainset = sum(trainset,[])
    trainset_data = np.asarray([img[0] for img in trainset])
    means_train = trainset_data.mean(axis = (0,1,2)) / 255
    stds_train = trainset_data.std(axis = (0,1,2)) / 255
    print(f'Calculated means: {means_train}')
    print(f'Calculated stds: {stds_train}')
    del trainset_data

    with open('testset.pkl', 'rb') as f:
    testset = pickle.load(f)
    testset_data = np.asarray([img[0] for img in testset]) 
    means_test = testset_data.mean(axis = (0,1,2)) / 255
    stds_test = testset_data.std(axis = (0,1,2)) / 255
    print(f'Calculated means: {means_test}')
    print(f'Calculated stds: {stds_test}')
    del testset_data

    train_transforms = transforms.Compose([
                          transforms.RandomHorizontalFlip(),
                          transforms.RandomRotation(10),
                          #transforms.CenterCrop(64),
                          transforms.ToTensor(),
                          #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                          transforms.Normalize(mean = means_train, 
                                              std = stds_train)
                       ])
    #Torch tensor to PIL image
    trans = transforms.ToPILImage()
    trainset_transformed = []
    for i in range(len(trainset)):
      trainset_transformed.append((train_transforms(trans(trainset[i][0])), trainset[i][1]))
      if i%400 == 0:
        print(i, ' images are processed')
    del trainset
    test_transforms = transforms.Compose([
                          transforms.RandomHorizontalFlip(),
                          transforms.RandomRotation(10),
                          #transforms.CenterCrop(64),
                          transforms.ToTensor(),
                          #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                          transforms.Normalize(mean = means_test, 
                                              std = stds_test)
                       ])
    trans = transforms.ToPILImage()
    testset_transformed = []
    for i in range(len(testset)):
      testset_transformed.append((test_transforms(trans(testset[i][0])), testset[i][1]))
      if i%400 == 0:
        print(i, ' images are processed')
    del testset


    n_train_examples = int(len(trainset_transformed)*0.9)
    n_valid_examples = len(trainset_transformed) - n_train_examples
    train_data, valid_data = torch.utils.data.random_split(trainset_transformed, 
                                                           [n_train_examples, n_valid_examples])
    print(f'Number of training examples: {len(train_data)}')
    print(f'Number of validation examples: {len(valid_data)}')
    print(f'Number of testing examples: {len(testset_transformed)}')

    BATCH_SIZE = 128
    train_iterator = torch.utils.data.DataLoader(train_data, 
                                                 shuffle = True, 
                                                 batch_size = BATCH_SIZE)
    valid_iterator = torch.utils.data.DataLoader(valid_data, 
                                                 batch_size = BATCH_SIZE)

    test_iterator = torch.utils.data.DataLoader(testset_transformed,
                                               batch_size = 1)

    model = LeNet(OUTPUT_DIM)
    print(model)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    train_model(model, EPOCHS=15, OUTPUT_DIM=5, outputModel='bottle-LeNet.pt')

    model = AlexNet(OUTPUT_DIM)
    print(model)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    train_model(model, EPOCHS=15, OUTPUT_DIM=5, outputModel='bottle-AlexNet.pt')

    print('Class accuracy with LeNet: ')
    class_accuracy(model=LeNet(OUTPUT_DIM), outputModel='bottle-LeNet.pt')
    print("====================")
    print('Class accuracy with AlexNet: ')
    class_accuracy(model=AlexNet(OUTPUT_DIM), outputModel='bottle-AlexNet.pt')

    #prediction for test set
    net = AlexNet(OUTPUT_DIM)
    net.load_state_dict(torch.load('bottle-AlexNet.pt'))
    pred = []
    with torch.no_grad():
        for data in test_iterator:
            images, png = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            #print(png[0], ' ', predicted.item())
            pred.append((png[0], predicted.item()))
    df = pd.DataFrame(pred, columns=['Id', 'label'])
    df.to_csv('submission.csv', index=None)
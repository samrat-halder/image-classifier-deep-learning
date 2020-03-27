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

#load the labels
def load_labels():
  return ['0', '1', '2', '3', '4']

#read train data in CIFAR10 data format
def image_train_read(folder, class_, ls_):
  count = 1
  #read all the files in train directory
  for image in os.listdir(os.path.join(folder, class_)):
    print('Class : ', class_, ' Image count : ', count)
    img = cv2.imread(train_data_folder + class_ + '/' + image)
    #plt.imshow(img)
    ls_.append((img, int(class_)))
    count += 1

#read prediction/ test data in CIFAR10 data format
def image_test_read(folder, ls_):
  count = 1
  for image in os.listdir(folder):
    print('Image count : ', count)
    img = cv2.imread(test_data_folder + image)
    #plt.imshow(img)

    ls_.append((img, str(image)))
    count += 1

#calculate accuracy for predictions
def calculate_accuracy(fx, y):
    preds = fx.argmax(1, keepdim=True)
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc

def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0 
    model.train()
    for (x, y) in iterator:    
        x = x.to(device)
        y = y.to(device)    
        optimizer.zero_grad()
        fx = model(x)
        loss = criterion(fx, y)
        acc = calculate_accuracy(fx, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device): 
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)
            fx = model(x)
            loss = criterion(fx, y)
            acc = calculate_accuracy(fx, y)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, EPOCHS=10, OUTPUT_DIM=5, outputModel='bottle-model-LeNet.pt'):
  best_valid_loss = float('inf')
  optimizer = optim.Adam(model.parameters())
  criterion = nn.CrossEntropyLoss()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device)
  criterion = criterion.to(device)
  for epoch in range(EPOCHS):  
      start_time = time.time()
      train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
      valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)
      if valid_loss < best_valid_loss:
          best_valid_loss = valid_loss
          torch.save(model.state_dict(), outputModel)
      end_time = time.time()
      epoch_mins, epoch_secs = epoch_time(start_time, end_time)
      print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
      print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
      print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

def class_accuracy(model, outputModel=''):
  net = model
  net.load_state_dict(torch.load(outputModel))
  class_correct = list(0. for i in range(OUTPUT_DIM))
  class_total = list(0. for i in range(OUTPUT_DIM))
  with torch.no_grad():
      for data in valid_iterator:
          images, labels = data
          outputs = net(images)
          _, predicted = torch.max(outputs, 1)
          c = (predicted == labels).squeeze()
          for i in range(OUTPUT_DIM):
              label = labels[i]
              class_correct[label] += c[i].item()
              class_total[label] += 1

      for i in range(5):
        print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

# -*- coding: utf-8 -*-
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from KITTI_ground_dataset import KITTIGroundDataset
from unet import Unet

def train(trainloader, net, criterion, optimizer, device, epoch):
    '''
    Function for training.
    '''
    start = time.time()
    running_loss = 0.0
    net = net.train()
    iter_loss = []
    for images, labels in tqdm(trainloader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = net(images)
        point_mask = torch.sum(images,axis=1,keepdim=True).type(torch.BoolTensor)
        loss = criterion(output[point_mask], labels[point_mask])
        iter_loss.append(loss)
        loss.backward()
        optimizer.step()
        running_loss = loss.item()
    end = time.time()
    print('[epoch %d] loss: %.3f elapsed time %.3f' %
          (epoch, running_loss, end-start))
    return running_loss, iter_loss

def test(testloader, net, criterion, device):
    '''
    Function for testing.
    '''
    losses = 0.
    cnt = 0
    with torch.no_grad():
        net = net.eval()
        for images, labels in tqdm(testloader):
            images = images.to(device)
            labels = labels.to(device)
            output = net(images)
            point_mask = torch.sum(images,axis=1,keepdim=True).type(torch.BoolTensor)
            loss = criterion(output[point_mask], labels[point_mask])
            losses += loss.item()
            cnt += 1
    print(losses / cnt)
    return (losses/cnt)

# TODO change data_range to include all train/evaluation/test data.
# TODO adjust batch_size.
data_dir = '../../KITTI/KITTI_odometry/dataset/sequences/'
train_data = KITTIGroundDataset(data_dir, np.arange(1,2))
train_loader = DataLoader(train_data, batch_size=4)
eval_data = KITTIGroundDataset(data_dir, np.arange(10,11))
eval_loader = DataLoader(eval_data, batch_size=4)

name = 'ground_estimation_net'
net = Unet(32,1,5,32,concat=False)
use_multi_GPU = True
if use_multi_GPU:
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

num_epochs = 5
train_loss = np.zeros(num_epochs)
validation_loss = np.zeros(num_epochs)
train_iteration_loss = []

print('\nStart training')
for epoch in range(num_epochs): #TODO decide epochs
    print('-----------------Epoch = %d-----------------' % (epoch+1))
    train_loss[epoch], iter_loss = train(train_loader, net, criterion, optimizer, device, epoch+1)
    train_iteration_loss.extend(iter_loss)

    # TODO create your evaluation set, load the evaluation set and test on evaluation set
    validation_loss[epoch] = test(eval_loader, net, criterion, device)

plt.plot(np.arange(num_epochs)+1, train_loss,label='Training loss')
plt.plot(np.arange(num_epochs)+1, validation_loss,label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('MSE vs Epoch')
plt.legend()
plt.savefig('training_loss.png')
# plt.show()
# plt.close()

print('\nFinished Training, Testing on test set')
# test(test_loader, net, criterion, device)

torch.save(net.state_dict(), './models/model_{}.pth'.format(name))
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data.dataset import Subset

import matplotlib.pyplot as plt
import numpy as np

from net import Net

max_epoch=5000
batch_size=32
n_node=10

torch.random.manual_seed(1)

trainset = torchvision.datasets.MNIST(root='./data',train=True,
                                      download=True, transform=transforms.ToTensor())

indices=torch.load('./noniid_filter/filter_r90_s01.pt')
subset=[Subset(trainset,indices[i]) for i in range(10)]

trainloader = [torch.utils.data.DataLoader(subset[i], batch_size=batch_size,
                                           shuffle=False, num_workers=2) for i in range(10)]

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print('using device', device)
net=[Net().to(device) for i in range(10)]
local_model=[{} for i in range(10)]
recv_models=[[] for i in range(10)]

fl_coefficiency=0.1

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = [optim.Adam(net[i].parameters(), lr=0.001) for i in range(10)]

# pre-self training
for epoch in range(50) :
    for n in range(10) :
        running_loss = 0.0
        for i, data in enumerate(trainloader[n], 0) :

            # get the inputs; data is a list of [x_train, y_train]
            x_train, y_train = data
            x_train = x_train.to(device)
            y_train = y_train.to(device)

            # zero the parameter gradients
            optimizer[n].zero_grad()

            # forward + backward + optimize
            y_output = net[n](x_train)
            loss = criterion(y_output, y_train)
            loss.backward()
            optimizer[n].step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print(f'Pre-self training: [{n}, {epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.5f}')
                running_loss = 0.0
    
import json

contact_list=[]

filename=f'./contact_pattern/rwp_n10_a0500_r100_p10_s01.json'
# filename=f'./contact_pattern/cse_n10_c10_b02_tt10_tp5_s01.json'
print(f'Loading ... {filename}')
with open(filename) as f :
    contact_list=json.load(f)

for epoch in range(max_epoch):  # loop over the dataset multiple times
    
    #print(global_model['fc2.bias'][1])
    contact = contact_list[epoch]
    print(f'at t={epoch} : ', contact)

    # receive local_models from contacts
    for n in range(10) :
        local_model[n]=net[n].state_dict()
        nbr = contact[str(n)]
        recv_models[n]=[]
        for k in nbr :
            recv_models[n].append(net[k].state_dict())

    for n in range(10) :
        update_model=recv_models[n]
        n_nbr=len(update_model)
        print(f'at {n} n_nbr={n_nbr}')
        for k in range(n_nbr) :
            for key in update_model[k] :
                update_model[k][key]=recv_models[n][k][key]-local_model[n][key]
        
        for k in range(n_nbr) :
            for key in update_model[k] :
                local_model[n][key] += update_model[k][key]*fl_coefficiency/(n_nbr+1)

    for n in range(10) :
        nbr = contact[str(n)]
        if len(nbr)>0 :
            net[n].load_state_dict(local_model[n])

    for n in range(10) :
        nbr = contact[str(n)]
        if len(nbr)==0 :
            continue

        running_loss = 0.0
        for i, data in enumerate(trainloader[n], 0) :

            # get the inputs; data is a list of [x_train, y_train]
            x_train, y_train = data
            x_train = x_train.to(device)
            y_train = y_train.to(device)

            # zero the parameter gradients
            optimizer[n].zero_grad()

            # forward + backward + optimize
            y_output = net[n](x_train)
            loss = criterion(y_output, y_train)
            loss.backward()
            optimizer[n].step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print(f'[{n}, {epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.5f}')
                running_loss = 0.0
    
    #print(net.state_dict())
    print(f'Model saving ... at {epoch+1}')
    for n in range(10) :
        torch.save(net[n].state_dict(),f'./trained_net/mnist_net_{n}_{epoch+1:04d}.pth')
    
print('Finished Training')

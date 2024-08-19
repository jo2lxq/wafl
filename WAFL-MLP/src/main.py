import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from net import Net
from torch.utils.data.dataset import Subset

#
# Configurations
#

# experiment name -- recommended names are rwp0500, rwp1000, rwp2000, cse02, cse04, cse08. line, tree, ringstar, dense
experiment_case = "rwp0500"

# specify the contact pattern file.
#   recommended files are:
#    for rwp0500  -- rwp_n10_a0500_r100_p10_s01.json
#    for rwp1000  -- rwp_n10_a1000_r100_p10_s01.json
#    for rwp2000  -- rwp_n10_a2000_r100_p10_s01.json
#    for cse02    -- cse_n10_c10_b02_tt10_tp5_s01.json
#    for cse04    -- cse_n10_c10_b04_tt10_tp5_s01.json
#    for cse08    -- cse_n10_c10_b08_tt10_tp5_s01.json
#    for line     -- static_line_n10.json
#    for tree     -- static_tree_n10.json
#    for ringstar -- static_ringstar_n10.json
#    for dense    -- static_dense_n10.json
cp_filename = f"../data/contact_pattern/rwp_n10_a0500_r100_p10_s01.json"

# Hyperparameters
self_train_epoch = 50  # default 50
max_epoch = 5000  # default 5000
n_device = 10  # use 10

batch_size = 32  # default 32
learning_rate = 0.001  # default 0.001
fl_coefficiency = 1.0  # defaulr 1.0  (WAFL's aggregation co efficiency)

# Fix the seed
torch.random.manual_seed(1)

# Prepare the model folder (generated from the specified experiment_case)
TRAINED_NET_PATH = f"../trained_net/{experiment_case}"
if not os.path.exists(TRAINED_NET_PATH):
    os.makedirs(TRAINED_NET_PATH)

#
# Setting up Train DataSet Loader
#
trainset = torchvision.datasets.MNIST(
    root="../data/MNIST/", train=True, download=True, transform=transforms.ToTensor()
)
indices = torch.load("../data/noniid_filter/filter_r90_s01.pt")
subset = [Subset(trainset, indices[i]) for i in range(n_device)]
trainloader = [
    torch.utils.data.DataLoader(
        subset[i], batch_size=batch_size, shuffle=False, num_workers=2
    )
    for i in range(n_device)
]

# Processor setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using device", device)

# Model Parameter Configuration
net = [Net().to(device) for i in range(n_device)]
local_model = [{} for i in range(n_device)]
recv_models = [[] for i in range(n_device)]

# Setting up Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = [optim.Adam(net[i].parameters(), lr=learning_rate) for i in range(n_device)]

#
# Run "pre-self training"
#
for epoch in range(self_train_epoch):
    for n in range(n_device):
        running_loss = 0.0
        for i, data in enumerate(trainloader[n], 0):

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
            if i % 100 == 99:  # print every 100 mini-batches
                print(
                    f"Pre-self training: [{n}, {epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.5f}"
                )
                running_loss = 0.0

#
# Load Contact list for mobility / static network simulation
#
contact_list = []
print(f"Loading ... contact pattern file: {cp_filename}")
with open(cp_filename) as f:
    contact_list = json.load(f)

#
# WAFL Training
#
for epoch in range(max_epoch):  # loop over the dataset multiple times

    # print(global_model['fc2.bias'][1])
    contact = contact_list[epoch]
    print(f"at t={epoch} : ", contact)

    # receive local_models from contacts
    for n in range(n_device):
        local_model[n] = net[n].state_dict()
        nbr = contact[str(n)]
        recv_models[n] = []
        for k in nbr:
            recv_models[n].append(net[k].state_dict())

    for n in range(n_device):
        update_model = recv_models[n]
        n_nbr = len(update_model)
        print(f"at {n} n_nbr={n_nbr}")
        for k in range(n_nbr):
            for key in update_model[k]:
                update_model[k][key] = recv_models[n][k][key] - local_model[n][key]

        for k in range(n_nbr):
            for key in update_model[k]:
                local_model[n][key] += (
                    update_model[k][key] * fl_coefficiency / (n_nbr + 1)
                )

    for n in range(n_device):
        nbr = contact[str(n)]
        if len(nbr) > 0:
            net[n].load_state_dict(local_model[n])

    for n in range(n_device):
        nbr = contact[str(n)]
        if len(nbr) == 0:
            continue

        running_loss = 0.0
        for i, data in enumerate(trainloader[n], 0):

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
            if i % 100 == 99:  # print every 100 mini-batches
                print(f"[{n}, {epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.5f}")
                running_loss = 0.0

    # print(net.state_dict())
    print(f"Model saving ... at {epoch+1}")
    for n in range(n_device):
        torch.save(
            net[n].state_dict(), f"{TRAINED_NET_PATH}/mnist_net_{n}_{epoch+1:04d}.pth"
        )

print(
    f"Finished Training. Models were saved in {TRAINED_NET_PATH}. Next, run draw_trend_graph.py and draw_confusion_matrix.py"
)

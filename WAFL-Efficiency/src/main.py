import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision
from torchvision import datasets, transforms
import timm
import random
import os
import copy
import time
import sys
import argparse

from functions.eval import evaluate
from functions.model_exchange import update_nets
from functions.subset import create_subsets

parser = argparse.ArgumentParser(description="Hyper Parameters")
parser.add_argument("--pre_epoch", default = 10, type=int)
parser.add_argument("--max_epoch", default = 1000, type=int)
parser.add_argument("--batch_size", default = 32, type=int)
parser.add_argument("--steps_per_epoch", default = 5, type=int)
parser.add_argument("--noniid", default = 0.9, type=float)
parser.add_argument("--zero_initialization", default = False, type=bool)
parser.add_argument("--reduce", default = "topk_ds_dq", type=str, help = "set topk_ds, topk_ds_dq, or None") 
parser.add_argument("--K", default = 0.1, type=float, help = "sparsification rate")
parser.add_argument("--Q", default = 1, type=float, help = "quatization bit")

args = parser.parse_args()
pre_epoch = args.pre_epoch
max_epoch = args.max_epoch
batch_size = args.batch_size
steps_per_epoch = args.steps_per_epoch
noniid = args.noniid
zero_initialization = args.zero_initialization
reduce = args.reduce
K = args.K
Q = args.Q

fl_coefficiency = 1
class_num = 100
node_num = 10

# setup
start_time = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.random.manual_seed(1)
random.seed(1)

# directory for storing the prev_net data
tmp_dir = "../tmp/" 
if os.path.isdir(tmp_dir):
	for filename in os.listdir(tmp_dir):
		os.remove(tmp_dir + filename)
else:
	os.makedirs(tmp_dir)
save_dir = "../trained_net/" 
if not os.path.isdir(tmp_dir):
	os.makedirs(save_dir)

# prepare dataloader
train_transform = transforms.Compose([
	transforms.Resize(224),  
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
])
val_transform = transforms.Compose([
	transforms.Resize(224),  
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
])

trainset = torchvision.datasets.CIFAR100(root='../data', train = True, download=True, transform=train_transform)
trainloaders = []

print("Data loaded")

if noniid == 0: # iid
	trainset_size = int(len(trainset) / node_num)
	for n in range(node_num):
		indices = list(range(n * trainset_size, (n + 1) * trainset_size))
		traindataset = Subset(trainset, indices)
		trainloaders.append(torch.utils.data.DataLoader(traindataset, batch_size = batch_size, shuffle = True))
elif 0 < noniid and noniid <= 1: # noniid
	class_to_node = {cls: random.randint(0, node_num - 1) for cls in range(class_num)}
	subsets = create_subsets(trainset, num_nodes = node_num , class_to_node = class_to_node, bias_ratio = noniid)
	for n in range(node_num):
		trainloaders.append(torch.utils.data.DataLoader(subsets[n], batch_size = batch_size, shuffle = True))
else:
	print(f"argument error : noniid = {noniid}")
	sys.exit()

valset = torchvision.datasets.CIFAR100(root='../data', train = False, download=True, transform=val_transform)
valloader = torch.utils.data.DataLoader(valset, batch_size = batch_size, shuffle=True)

del trainset
del valset

nets = [timm.create_model('vit_base_patch16_224', pretrained=True) for _ in range(node_num)]
for n in range(node_num):
	nets[n].head = nn.Linear(nets[n].head.in_features, 100)  

prev_net = [[None] * node_num for _ in range(node_num)]
init_net = None
if zero_initialization == True: # initializa the prev_nets with zero
	init_net = copy.deepcopy(nets[0].state_dict())
	init_net["head.weight"] = torch.zeros_like(init_net["head.weight"])
	init_net["head.bias"] = torch.zeros_like(init_net["head.bias"])
for net in nets:
	net.to(device)

print("Model loaded")

optimizers = [optim.AdamW([
	{'params': [nets[i].cls_token], 'lr': 1e-5},  
	{'params': [nets[i].pos_embed], 'lr': 1e-5},  
	{'params':nets[i].patch_embed.parameters(), 'lr': 1e-5},  
	{'params': nets[i].blocks.parameters(), 'lr': 1e-5},  
	{'params': nets[i].norm.parameters(), 'lr': 1e-5},	
	{'params': nets[i].head.parameters(), 'lr': 1e-3},	
]) for i in range(node_num)]
criterion = nn.CrossEntropyLoss()

print("Start Pre-self Training")

# pre_self training
for epoch in range(pre_epoch):
	for n in range(node_num):
		net = nets[n]
		optimizer = optimizers[n]
		trainloader = trainloaders[n]

		net.train()
		running_loss = 0.0
		correct = 0
		total = 0

		for batch_index, (inputs, labels) in enumerate(trainloader):
			if batch_index == steps_per_epoch:
				break
			inputs, labels = inputs.to(device), labels.to(device)
			optimizer.zero_grad()
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
			_, predicted = torch.max(outputs, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
		print(f"Epoch [{epoch+1}/{pre_epoch}] Node {n} Loss {running_loss/len(trainloader):.4f} Accuracy {100 * correct/total:.2f}%")

# save (after pre self traininig)
for n in range(node_num):
	torch.save(nets[n].state_dict(), f"{save_dir}/prenet_e{pre_epoch}_n{n}.pth")

for n in range(node_num):
	acc = evaluate(nets[n], valloader, device)
	print(f"Node {n} Validation Accuracy {acc:.2f}%")

print("Start WAFL")

# WAFL
import json
contact_list = []
filename=f'./contact_pattern/rwp_n10_a0500_r100_p10_s01.json'
print(f'Loading ... {filename}')
with open(filename) as f :
	contact_list=json.load(f)

for epoch in range(max_epoch):
	contact = contact_list[epoch]
	update_nets(nets, contact, fl_coefficient=1, reduce=reduce, K=K, Q=Q, zero_initialization = zero_initialization, prev_net=prev_net, init_net = init_net, device = device)

	for n in range(node_num):
		nbr = contact[str(n)]
		if len(nbr) == 0:
			continue

		net = nets[n]
		optimizer = optimizers[n]
		trainloader = trainloaders[n]

		net.train()
		running_loss = 0.0
		correct = 0
		total = 0

		for batch_index, (inputs, labels) in enumerate(trainloader):
			if batch_index == steps_per_epoch:
				break
			inputs, labels = inputs.to(device), labels.to(device)
			optimizer.zero_grad()
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
			_, predicted = torch.max(outputs, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
		print(f"Epoch [{epoch+1}/{max_epoch}] Node {n} Loss {running_loss/len(trainloader):.4f} Accuracy {100 * correct/total:.2f}%")

	# evaluation
	sum = 0
	for n in range(node_num):
		acc = evaluate(nets[n], valloader, device)
		print(f"Epoch [{epoch+1}/{max_epoch}] Node {n} Validation Accuracy {acc:.2f}%")
		sum += acc
	print(f"Epoch [{epoch+1}/{max_epoch}] Avg Validation Accuracy {sum / node_num:.2f}%")

	# save
	if (epoch + 1) == max_epoch:
		for n in range(node_num):
			torch.save(nets[n].state_dict(), f"{save_dir}/net_e{epoch+1}_n{n}.pth")
		
for filename in os.listdir(tmp_dir):
	os.remove(file_path)
print(f"time : {time.time() - start_time}")

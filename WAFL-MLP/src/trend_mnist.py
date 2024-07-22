import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import statistics
import math

from net import Net

#output_code="static"
#exp_codes=['static_line', 'static_line_cof01', 'static_tree', 'static_tree_cof01', 'static_ringstar', 'static_ringstar_cof01']
output_code="cse"
exp_codes=['self-train', 'federated', 'cse02','cse04','cse08']

def save_accuracy_trend(x,acc,classes,
                        title='Accuracy Trend',
                        save_path=None) :

    plt.clf()
    #plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.title(title)
    for i in range(len(classes)) :
        y=acc[i] 
        plt.plot(x,y,label=classes[i])

    if(len(classes)>1):
        plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    if(save_path==None):
        plt.show()
    else:
        plt.savefig(save_path,bbox_inches='tight')

def save_accuracy_trend_with_eb(x,acc,x2,acc2,acc_err,classes,
                        title='Accuracy Trend',
                        save_path=None) :

    plt.clf()
    #plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.title(title)
    for i in range(len(classes)) :
        y=acc[i] 
        y2=acc2[i]
        #plt.errorbar(x2, y2, yerr = acc_err, capsize=5, fmt='o')
        plt.errorbar(x2, y2, yerr = acc_err, capsize=2, fmt='o', markersize=5, ecolor='black', markeredgecolor = "black", color='w')
        plt.plot(x,y,label=classes[i])

    if(len(classes)>1):
        plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    if(save_path==None):
        plt.show()
    else:
        plt.savefig(save_path,bbox_inches='tight')


max_epoch=5000       # trained epoch to load
batch_size=512

# nodes=['0','1','2','3','4','5','6','7','8','9']

x=np.linspace(1,max_epoch,max_epoch)
x2=np.linspace(max_epoch/10,max_epoch,10)

print(x2)

acc=[ np.array([i for i in range(max_epoch)],dtype='float32')
        for i in range(len(exp_codes))]
acc2=[ np.array([i for i in range(10)],dtype='float32')
        for i in range(len(exp_codes))]
acc_err=[ np.array([i for i in range(10)],dtype='float32')
        for i in range(len(exp_codes))]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('using device', device)

testset = torchvision.datasets.MNIST(root='.data',train=False,
                                        download=True, transform=transforms.ToTensor())

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)


net=Net().to(device)

for epoch in range(1,max_epoch+1) :
    print(epoch)

    for e in range(len(exp_codes)) : 
        exp_code=exp_codes[e] 

        accuracies = []
        for n in range(10) :
            net.load_state_dict(torch.load(f'./{exp_code}/mnist_net_{n}_{epoch:04d}.pth',map_location=torch.device('cpu')))

            correct = 0
            total = 0

            # since we're not training, we don't need to calculate the gradients for our outputs
            with torch.no_grad():
                for data in testloader:
                    x_test, y_test = data
                    x_test = x_test.to(device)
                    y_test = y_test.to(device)

                    # calculate outputs by running images through the network
                    y_output = net(x_test)
                    # the class with the highest energy is what we choose as prediction
                    _, y_pred = torch.max(y_output.data, 1)

                    # collect the correct predictions for each class
                    for groundtruth, prediction in zip(y_test, y_pred):
                        if groundtruth == prediction:
                            correct += 1
                        total += 1
    
            # print accuracy for each node
            accuracy = float(correct) / float(total)
            accuracies.append(accuracy)
            print(f'epoch={epoch}, node={n}, exp_code={exp_code}, accuracy={accuracy}')

        acc[e][epoch-1]=statistics.mean(accuracies)
        if epoch in x2 :
            i=x2.tolist().index(epoch)
            acc2[e][i]=statistics.mean(accuracies)
            acc_err[e][i]=statistics.stdev(accuracies)


save_accuracy_trend(x,acc,exp_codes,title=f'',save_path=f'./accuracy/accuracy_{output_code}.png')
#save_accuracy_trend_with_eb(x,acc,x2,acc2,acc_err,exp_codes,title='',save_path=f'./accuracy/accuracy_eb_{output_code}.png')

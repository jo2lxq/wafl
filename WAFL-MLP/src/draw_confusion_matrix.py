import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from net import Net

from matplotlib import cm
from sklearn.metrics import confusion_matrix
import itertools

experiment_case = 'cse08'
epochs = [1, 100, 200, 1000, 5000]
nodes = [9]

batch_size=256

def save_confusion_matrix(cm,classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          save_path=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if(save_path==None):
        plt.show()
    else:
        plt.savefig(save_path,bbox_inches='tight')


for epoch in epochs :

    for n in nodes :

        testset = torchvision.datasets.MNIST(root='.data',train=False,
                                        download=True, transform=transforms.ToTensor())

        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

        net=Net()
        net.load_state_dict(torch.load(f'./{experiment_case}/mnist_net_{n}_{epoch:04d}.pth',map_location=torch.device('cpu')))

        y_preds = []
        y_tests = []

        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                x_test, y_test = data
                # calculate outputs by running images through the network
                y_output = net(x_test)
                # the class with the highest energy is what we choose as prediction
                _, y_pred = torch.max(y_output.data, 1)
                y_preds.extend(y_pred.tolist())
                y_tests.extend(y_test.tolist())


        confusion_mtx = confusion_matrix(y_tests, y_preds) 
        #save_confusion_matrix(confusion_mtx, 
        #        classes = range(10),
        #        normalize = False,
        #        title=f'{experiment_case} (node={n}, epoch={epoch:d})', 
        #        cmap=plt.cm.Reds,
        #        save_path=f'./confusion_matrix/mnist_cm_{experiment_case}_count_{n}_{epoch:04d}.png')

        save_confusion_matrix(confusion_mtx, 
                classes = range(10),
                normalize = True,
                title=f'{experiment_case} (node={n}, epoch={epoch:d})', 
                cmap=plt.cm.Reds,
                save_path=f'./confusion_matrix/mnist_cm_normalize_{experiment_case}_{n}_{epoch:04d}.png')
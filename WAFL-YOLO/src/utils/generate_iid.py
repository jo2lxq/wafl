import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import random


def generate_iid(dl, batch_size, num_clients, num_classes, filename, random_seed=42):
    random.seed(random_seed)

    print(f'Generating IID filter...')

    indices = [[] for i in range(num_clients)]

    dists = np.zeros((num_clients, num_classes), dtype=np.int64)

    index = 0
    for imgs, targets, path, _ in dl:
        labels = []
        i = targets[:, 0]
        u, counts = i.unique(return_counts=True)
        for j in range(len(u)):
            matches = i == j
            target = targets[matches, 1:]
            if target.shape[0] > 1:
                arr = target[:, 0].reshape(-1).to('cpu').detach().numpy().copy()
                unique, freq = np.unique(arr, return_counts=True)
                mode = unique[np.argmax(freq)]
                labels.append(int(mode))
            elif target.shape[0] == 1:
                labels.append(int(target[0, 0].item()))
            else:
                raise RuntimeError("There is a data with no label")

        for i in range(len(labels)):
            n = random.randint(0, 9)
            indices[n].append(index + i)
            dists[n][labels[i]] += 1

        index += batch_size

    torch.save(indices, filename)
    print('Done!')
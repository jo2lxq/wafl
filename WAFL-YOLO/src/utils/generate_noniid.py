import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import random


def generate_noniid(dl, batch_size, num_clients, num_classes, filename, ratio=90, random_seed=42):
    random.seed(random_seed)

    print(f'Generating NonIID filter...')

    indices = [[] for i in range(num_clients)]

    dists = np.zeros((num_clients, num_classes), dtype=np.int64)

    # anno_dists = np.zeros((num_classes, num_classes), dtype=np.int64)

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
                # for cat in target['labels'].tolist():
                #     anno_dists[mode][cat] += 1
            elif target.shape[0] == 1:
                labels.append(int(target[0, 0].item()))
                # anno_dists[target['labels'].item()][target['labels'].item()] += 1
            else:
                raise RuntimeError("There is a data with no label")

        for i in range(len(labels)):
            if random.randint(0, 99) < ratio: 
                indices[labels[i]].append(index + i)
                dists[labels[i]][labels[i]] += 1
            else:
                n = random.randint(0, 8)
                if labels[i] <= n:
                    n += 1
                indices[n].append(index + i)
                dists[n][labels[i]] += 1

        index += batch_size

    print(dists)
    torch.save(indices, filename)
    raise RuntimeError("test completed")
    print('Done!')


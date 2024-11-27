import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset
import random
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset


def generate(dataset, batch_size, num_clients, num_classes, filename, random_seed=42):
    random.seed(random_seed)

    print(f'Generating IID filter...')

    sampler = torch.utils.data.SequentialSampler(dataset)
    batch_sampler = torch.utils.data.BatchSampler(
    sampler, batch_size, drop_last=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler,
                                            collate_fn=utils.collate_fn, num_workers=2)

    indices = [[] for i in range(num_clients)]

    dists = np.zeros((num_clients, num_classes), dtype=np.int64)

    # anno_dists = np.zeros((num_classes, num_classes), dtype=np.int64)

    index = 0
    for imgs, targets in data_loader:
        labels = []
        for target in targets:
            assert target['labels'].numel() != 0, "detect some labels are excluded"
            if target['labels'].numel() > 1:
                arr = target['labels'].to('cpu').detach().numpy().copy()
                unique, freq = np.unique(arr, return_counts=True)
                mode = unique[np.argmax(freq)]
                labels.append(mode)
                # for cat in target['labels'].tolist():
                #     anno_dists[mode][cat] += 1
            else:
                labels.append(target['labels'].item())
                # anno_dists[target['labels'].item()][target['labels'].item()] += 1

        for i in range(len(labels)):
            n = random.randint(0, 9)
            indices[n].append(index + i)
            dists[n][labels[i]] += 1

        index += batch_size

    torch.save(indices, filename)
    print('Done!')
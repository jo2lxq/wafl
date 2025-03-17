# Description
"""
- `ratio(r)` is the probability. `r` proportion of data with label i will be assigned to node i.
- `randomseed(s)` is the seed value used for data allocation.
"""

import os
import random
import sys

import torch
import tqdm
# import torchvision.datasets as datasets
import torchvision.transforms as transforms

sys.path.append("./")  # to import Mydataset
from functions.mydataset import Mydataset

if __name__ == "__main__":
    ## 1. Modifiable parameters
    randomseed = 1
    ratio = 90  # the rate that the n-th node has n-labeled picture
    n_node = 10
    n_output = 10 # label num

    ## 2. Other parameters and settings
    batch_size = 16
    random.seed(randomseed)

    # Specify the data and output file paths
    current_path = os.path.dirname(os.path.abspath(__file__))  # WAFL-ViT/src/utils
    data_dir = os.path.normpath(os.path.join(current_path, "../../data"))
    filename = os.path.join(
        data_dir, f"non-IID_filter/filter_r{ratio:02d}_s{randomseed:02d}.pt"
    )
    meanfile = os.path.join(
        data_dir, f"non-IID_filter/mean_r{ratio:02d}_s{randomseed:02d}.pt"
    )
    stdfile = os.path.join(
        data_dir, f"non-IID_filter/std_r{ratio:02d}_s{randomseed:02d}.pt"
    )
    print(f"Generating Non-IID filter ... {filename}")

    train_dir = os.path.join(data_dir, "train")  # Path to the dataset
    tmp_transform = transforms.Compose(
        [
            #transforms.Resize((256,256)),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    # trainset = datasets.ImageFolder(train_dir, transform=tmp_transform)
    trainset = Mydataset(train_dir, n_output, transform=tmp_transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, num_workers=4, pin_memory=True
    )

    indices = [
        [] for _ in range(0, n_node)
    ]  # indices[i] represents the data of the i-th node
    means = [torch.zeros(3) for _ in range(n_node)]
    stds = [torch.zeros(3) for _ in range(n_node)]

    index = 0

    ## 3. Creating the Non-IID filter (creating a list that represents which index of data each node has in the entire dataset)
    for data in trainloader:
        
        images, labels = data
        print(f"Batch images shape: {images.shape}, labels shape: {labels.shape}")

        batch_size = len(labels)
        labels = labels.tolist()

        for i in range(batch_size):
            if random.randint(0, 99) < ratio:
                print(labels[i])
                indices[labels[i]%n_node].append(index + i)
                means[labels[i]%n_node] += images[i].mean(dim=(1, 2))
                stds[labels[i]] += images[i].std(dim=(1, 2))
            else:
                n = random.randint(0, n_node - 2)
                if labels[i] <= n:
                    n += 1
                indices[n].append(index + i)
                means[n] += images[i].mean(dim=(1, 2))
                stds[n] += images[i].std(dim=(1, 2))

        index += batch_size

    ## 4. Calculating the mean and standard deviation of each node
    for i in range(len(indices)):
        means[i] /= len(indices[i])
        stds[i] /= len(indices[i])
        means[i] = means[i].to("cpu")
        stds[i] = stds[i].to("cpu")
        print(f"node_{i}:{indices[i]}\n")

    torch.save(indices, filename)
    torch.save(means, meanfile)
    torch.save(stds, stdfile)
    print("Done")

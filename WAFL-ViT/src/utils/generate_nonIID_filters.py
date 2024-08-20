# Description
"""
- `ratio(r)` is the probability. `r` proportion of data with label i will be assigned to node i.
- `randomseed(s)` is the seed value used for data allocation.
"""

import os
import random

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

if __name__ == "__main__":
    ## 1. Modifiable parameters
    randomseed = 1
    ratio = 50  # the rate that the n-th node has n-labeled picture
    n_node = 10

    ## 2. Other parameters and settings
    batch_size = 20
    random.seed(randomseed)

    # Specify the data and output file paths
    current_path = os.path.dirname(os.path.abspath(__file__))  # WAFL-ViT/src/utils
    data_dir = os.path.normpath(os.path.join(current_path, "../../data"))
    filename = os.path.join(
        data_dir, f"non-IID_filter/filter_r{ratio:02d}_s{randomseed:02d}.pt"
    )
    print(filename)
    print(f"Generating Non-IID filter ... {filename}")

    train_dir = os.path.join(data_dir, "train")  # Path to the dataset
    tmp_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    trainset = datasets.ImageFolder(train_dir, transform=tmp_transform)
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
        image, label = data
        batch_size = len(label)
        label = label.tolist()

        for i in range(len(label)):
            if random.randint(0, 99) < ratio:
                indices[label[i]].append(index + i)
            else:
                n = random.randint(0, 8)
                if label[i] <= n:
                    n += 1
                indices[n].append(index + i)

        index += batch_size

    torch.save(indices, filename)
    print("Done")

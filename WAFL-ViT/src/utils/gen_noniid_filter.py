import os
import random

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

batch_size = 16
randomseed = 1
ratio = 70  # the rate that n-th node has n-labeled picture
data_dir = "../data-raid/data/UTokyoE_building_dataset"

random.seed(randomseed)

filename = os.path.join(
    data_dir, f"noniid_filter/filter_r{ratio:02d}_s{randomseed:02d}.pt"
)
meanfile = os.path.join(
    data_dir, f"noniid_filter/mean_r{ratio:02d}_s{randomseed:02d}.pt"
)
stdfile = os.path.join(data_dir, f"noniid_filter/std_r{ratio:02d}_s{randomseed:02d}.pt")
print(f"Generating NonIID filter ... {filename}")

train_dir = os.path.join(data_dir, "train")

tmp_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)
trainset = datasets.ImageFolder(train_dir, transform=tmp_transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, num_workers=50, pin_memory=True
)

indices = [[], [], [], [], [], [], [], [], [], []]  # indices[i]はi番目のノードのデータ

means = [torch.zeros(3) for i in range(10)]
stds = [torch.zeros(3) for i in range(10)]

index = 0
for data in trainloader:
    x, y = data
    batch_size = len(y)
    y = y.tolist()

    for i in range(len(y)):
        if random.randint(0, 99) < ratio:
            indices[y[i]].append(index + i)
            means[y[i]] += x[i].mean(dim=(1, 2))
            stds[y[i]] += x[i].std(dim=(1, 2))
        else:
            n = random.randint(0, 8)
            if y[i] <= n:
                n += 1
            indices[n].append(index + i)
            means[n] += x[i].mean(dim=(1, 2))
            stds[n] += x[i].std(dim=(1, 2))

    index += batch_size

for i in range(len(indices)):
    means[i] /= len(indices[i])
    stds[i] /= len(indices[i])
    # print(indices[i])

print(f"means:\n{means}\nstds:\n{stds}")
torch.save(indices, filename)
torch.save(means, meanfile)
torch.save(stds, stdfile)
print("Done")

# for checking
# subset = Subset(trainset,indices[1])
# subloader = torch.utils.data.DataLoader(subset, batch_size=batch_size,
#                                          num_workers=2)
# for data in subloader :
#     x, y= data
#     print(y)

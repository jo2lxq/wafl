# 説明
"""
- `ratio(r)`は確率。ラベルiのデータうち`r`がノードiに割り振られる。
- `randomseed(s)`はデータの割り振りに使うseedの値。
"""

import os
import random

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

## 1. 変更可能なパラメータ
randomseed = 1
ratio = 70  # the rate that n-th node has n-labeled picture
n_node = 10

## 2. その他パラメータや設定
batch_size = 20
random.seed(randomseed)
# データや出力ファイルのPath指定
data_dir = ".."
filename = os.path.join(
    data_dir, f"non-IID_filter/filter_r{ratio:02d}_s{randomseed:02d}.pt"
)
print(filename)
print(f"Generating NonIID filter ... {filename}")

train_dir = os.path.join(
    data_dir, "train"
)  # データセットのPath
tmp_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)
trainset = datasets.ImageFolder(train_dir, transform=tmp_transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, num_workers=16, pin_memory=True
)

indices = [[] for _ in range(0, n_node)]  # indices[i]はi番目のノードのデータ

means = [torch.zeros(3) for _ in range(n_node)]
stds = [torch.zeros(3) for _ in range(n_node)]

index = 0

## 3. Non-IID filterの作成（各ノードが、全体データの中でどのindexのデータを持つかを表すリストを作成）
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

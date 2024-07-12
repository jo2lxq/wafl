import os

import torch
import torchvision.io as io
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class FromSubsetDataset(Dataset):  # subset -> dataset
    def __init__(self, data_list, device, transform=None, pre_transform=None, useGPUinTrans=None):
        new_data_list = []
        for i in range(len(data_list)):
            image, label = data_list[i]
            if useGPUinTrans is True:
                image = pre_transform(image).to(device)
                label = torch.tensor(label).to(device)
            elif useGPUinTrans is False:
                image = pre_transform(image)
                label = torch.tensor(label)
            new_data_list.append([image, label])
        self.transform = transform
        self.data_list = new_data_list

    def __getitem__(self, index):
        data, label = self.data_list[index]
        if self.transform is not None:
            data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.data_list)


class MyGPUdataset(Dataset):  # use when put data on GPU in __init__
    def __init__(self, root, device, transform=None, pre_transform=None):
        self.data = []
        self.labels = []
        self.transform = transform
        for i in range(0, 10):  # labelが0~n_output-1なので、labelごとにデータを追加していく.
            dir = os.path.join(root, str(i))
            images_path = os.listdir(dir)
            images_path.sort()
            for image_path in images_path:
                full_path = os.path.join(dir, image_path)
                image_buf = io.read_image(full_path).to(device)
                image_buf = pre_transform(image_buf)
                self.data.append(image_buf)  # 画像をtorch.Tensorにかえてデータリストに追加. データをGPUに送る
                self.labels.append((torch.tensor(i)).to(device))

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        if self.transform is not None:
            data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.data)

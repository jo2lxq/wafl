import os
import pickle

import torch
import torchvision


def materialize_dataset(dataset):
    """
    Load all the samples into memory for full serialization.
    This also ensures that the generated subset only contains
    records from that subset (By default, Subset contains all
    the data (the Subset.dataset attribute returns the whole dataset)).
    """
    return [(dataset[i][0].clone().detach(), torch.tensor(dataset[i][1]).clone().detach()) for i in range(len(dataset))]


class PickledDataset(torch.utils.data.Dataset):
    """
    Wrapper for materializing the Dataset class.
    Instead of using pickled datasets, we could also transfer the entire MNIST directory
    to the WAFL agents and use the filter file to only load data belonging to the current subset.
    But that would mean having the entire dataset on all the devices, even if only a subset of it
    would be accessed. Using fully materialized dataset pickles ensures that only the subset's
    raw data is present on the WAFL agent.
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Number of WAFL Agents.
N = 10
# Path to the Filter File.
FILTER_PATH = "###.pt"
# Path to the Original Dataset (MNIST).
DATA_PATH = "###/MNIST/"
# Path to the 'wafl/dataset/' directory.
DATASET_PATH = "../wafl/dataset/"

trainset = torchvision.datasets.MNIST(
    root=DATA_PATH,
    train=True,
    download=True,
    transform=torchvision.transforms.transforms.ToTensor(),
)
indices = torch.load(FILTER_PATH)

for i in range(N):
    # Materializing the Dataset.
    subset = materialize_dataset(torch.utils.data.Subset(trainset, indices[i]))
    # Wrapping it in our Dataset Class.
    subset = PickledDataset(subset)
    device_dataset_dict = DATASET_PATH + f"{str(i)}/"
    os.makedirs(device_dataset_dict, exist_ok=True)
    with open(device_dataset_dict + "dataset.pickled", "wb") as file:
        pickle.dump(subset, file)
    print(f"Serialized the Subset for Device #{i}")
print("Split the dataset and generated the pickle files.")

import os
import pickle

import torch
import torchvision

# --- Configuration ---
N_DEVICES = 10  # 🤖 Number of execution servers (WAFL Agents).
FILTER_PATH = "data/filter_r90_s01.pt"  # 📂 Path to the data distribution filter file.
RAW_DATA_PATH = "data/"  # 📥 Path to store the original downloaded dataset (e.g., MNIST).
OUTPUT_DATASET_PATH = "../wafl/dataset/"  # 📤 Root directory for the output WAFL datasets.


def materialize_dataset(dataset: torch.utils.data.Dataset) -> list:
    """
    Loads all samples from a dataset into a list in memory.
    This makes the dataset object fully serializable with pickle.
    """
    print("  - Materializing dataset into memory...")
    return [(sample.clone().detach(), torch.tensor(label).clone().detach()) for sample, label in dataset]


class PickledDataset(torch.utils.data.Dataset):
    """
    A simple wrapper for a materialized (in-memory) dataset.
    This ensures that when an agent loads the data, it only sees its
    own subset, not the entire original dataset.
    """

    def __init__(self, data: list):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        return self.data[idx]


# --- Main Script ---
if __name__ == "__main__":
    print("🚀 --- Starting Dataset Creation ---")

    # 1. Prepare and process the device-specific training datasets
    # -------------------------------------------------------------
    print("\nSplitting training data for each device...")

    # Load the original training dataset
    trainset = torchvision.datasets.MNIST(
        root=RAW_DATA_PATH,
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    # Load the filter which defines which data samples go to which device
    indices = torch.load(FILTER_PATH)
    print(f"Loaded a filter for {len(indices)} devices.")

    for i in range(N_DEVICES):
        print(f"\nProcessing training set for Device #{i}...")

        # Create a subset for the current device based on the filter
        subset = torch.utils.data.Subset(trainset, indices[i])

        # Materialize and wrap the dataset
        materialized_subset = materialize_dataset(subset)
        pickled_subset = PickledDataset(materialized_subset)

        # Define the output directory and create it
        # Structure: wafl/dataset/{device_id}/train/
        output_dir = os.path.join(OUTPUT_DATASET_PATH, str(i), "train")
        os.makedirs(output_dir, exist_ok=True)
        print(f"  📂 Ensured directory exists: {output_dir}")

        # Save the pickled dataset
        output_file = os.path.join(output_dir, "train.pkl")
        with open(output_file, "wb") as f:
            pickle.dump(pickled_subset, f)
        print(f"  📦 Saved serialized training data to {output_file}")

    # 2. Prepare and process the common test dataset
    # -------------------------------------------------------------
    print("\n\nProcessing the common test dataset...")

    # Load the original test dataset
    testset = torchvision.datasets.MNIST(
        root=RAW_DATA_PATH,
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    # Materialize and wrap the entire test set
    materialized_testset = materialize_dataset(testset)
    pickled_testset = PickledDataset(materialized_testset)

    # Define the output directory and create it
    # Structure: wafl/dataset/common/test/
    output_dir = os.path.join(OUTPUT_DATASET_PATH, "common", "test")
    os.makedirs(output_dir, exist_ok=True)
    print(f"  📂 Ensured directory exists: {output_dir}")

    # Save the pickled dataset
    output_file = os.path.join(output_dir, "test.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(pickled_testset, f)
    print(f"  📦 Saved serialized test data to {output_file}")

    print("\n\n✅ --- Dataset creation complete! ---")

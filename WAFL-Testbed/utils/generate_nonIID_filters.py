# Description
"""
- `ratio(r)` is the probability. `r` proportion of data with label i will be assigned to node i.
- `randomseed(s)` is the seed value used for data allocation.
"""

import argparse
import os
import random

import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Non-IID filters for federated learning")
    parser.add_argument("--ratio", type=int, default=90, help="Rate that the n-th node has n-labeled picture (default: 90)")
    parser.add_argument("--randomseed", type=int, default=1, help="Random seed for data allocation (default: 1)")
    parser.add_argument("--nodes", type=int, default=10, help="Number of nodes (default: 10)")
    parser.add_argument("--n-output", type=int, default=10, help="Number of output labels (default: 10)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for data loading (default: 16)")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory (default: ./data)")

    args = parser.parse_args()

    ## 1. Parameters from command line arguments
    randomseed = args.randomseed
    ratio = args.ratio
    n_node = args.nodes
    n_output = args.n_output
    batch_size = args.batch_size

    ## 2. Other parameters and settings
    random.seed(randomseed)

    # Specify the data and output file paths
    data_dir = os.path.normpath(args.data_dir)
    filter_dir = os.path.join(data_dir, "nonIID_filter")
    if not os.path.exists(filter_dir):
        os.makedirs(filter_dir)

    filename = os.path.join(filter_dir, f"filter_r{ratio:02d}_s{randomseed:02d}.pt")
    meanfile = os.path.join(filter_dir, f"mean_r{ratio:02d}_s{randomseed:02d}.pt")
    stdfile = os.path.join(filter_dir, f"std_r{ratio:02d}_s{randomseed:02d}.pt")
    print(f"🚀 Generating Non-IID filter... {filename}")

    # Load dataset
    print("📂 Loading dataset...")
    train_dir = os.path.join(data_dir, "train")  # Path to the dataset
    tmp_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )
    trainset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=4, pin_memory=True)

    # Initialize data structures
    indices = [[] for _ in range(0, n_node)]  # indices[i] represents the data of the i-th node
    means = [torch.zeros(3) for _ in range(n_node)]
    stds = [torch.zeros(3) for _ in range(n_node)]

    index = 0

    ## 3. Creating the Non-IID filter
    print("🔄 Creating Non-IID filter (Class-Balanced Grouping)...")

    # Pre-calculate candidate nodes for each label (Class-Balanced Grouping)
    # label_to_nodes[label] = [node_id_1, node_id_2, ...]
    label_to_nodes = {y: [] for y in range(n_output)}
    for k in range(n_node):
        label_to_nodes[k % n_output].append(k)

    for data in tqdm(trainloader, desc="Processing batches", unit="batch"):
        images, labels = data

        batch_size = len(labels)
        labels = labels.tolist()

        for i in range(batch_size):
            label = labels[i] % n_output  # Ensure label is within n_output range

            # 1. Identify candidate nodes for this label (Primary Class)
            candidate_nodes = label_to_nodes[label]

            # 2. Randomly select one target node from the candidates
            target_node = random.choice(candidate_nodes)

            if random.randint(0, 99) < ratio:
                # Assign to the target node (Primary Data)
                indices[target_node].append(index + i)
                means[target_node] += images[i].mean(dim=(1, 2))
                stds[target_node] += images[i].std(dim=(1, 2))
            else:
                # Assign to a random noise node (Noise Data)
                # Select from all nodes excluding the target_node
                # Note: random.randint(a, b) includes b
                # To pick x from [0, n_node-1] excluding target_node:
                # Pick n from [0, n_node-2], if n >= target_node, then n += 1
                n = random.randint(0, n_node - 2)
                if n >= target_node:
                    n += 1

                indices[n].append(index + i)
                means[n] += images[i].mean(dim=(1, 2))
                stds[n] += images[i].std(dim=(1, 2))

        index += batch_size

    ## 4. Calculating the mean and standard deviation of each node
    print("📊 Calculating statistics for each node...")
    for i in tqdm(range(len(indices)), desc="Computing mean/std", unit="node"):
        means[i] /= len(indices[i])
        stds[i] /= len(indices[i])
        means[i] = means[i].to("cpu")
        stds[i] = stds[i].to("cpu")

    # Save results
    print("💾 Saving results...")
    torch.save(indices, filename)
    torch.save(means, meanfile)
    torch.save(stds, stdfile)
    print(f"✅ Non-IID filter saved to {filename}")
    print(f"✅ Mean values saved to {meanfile}")
    print(f"✅ Std values saved to {stdfile}")

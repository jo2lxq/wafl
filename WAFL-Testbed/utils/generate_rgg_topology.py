#!/usr/bin/env python3
"""
Random Geometric Graph (RGG) Contact Pattern Generator for WAFL Testbed

研究計画書の要件:
- ノード位置を固定
- 通信範囲内のノードと常に接続 (静的トポロジー)
- Dense (平均次数 ≥ 10) と Sparse (平均次数 ≤ 4) の両方をサポート

使用例:
  # Dense topology (平均次数 ≥ 10)
  python generate_rgg_topology.py --nodes 28 --density dense --epochs 5000

  # Sparse topology (平均次数 ≤ 4)
  python generate_rgg_topology.py --nodes 28 --density sparse --epochs 5000
"""

import argparse
import json
import math
import os
import random
from typing import Dict, List, Tuple


def calculate_radio_range_for_density(n_nodes: int, area_size: int, target_avg_degree: float) -> float:
    """
    Calculate required radio range to achieve target average degree in RGG.

    Args:
        n_nodes: Number of nodes
        area_size: Size of square area
        target_avg_degree: Target average degree (connectivity)

    Returns:
        Required radio range
    """
    # Theoretical formula for RGG: avg_degree ≈ π * r^2 * (n / area^2)
    # Solving for r: r = sqrt(avg_degree * area^2 / (π * n))
    area = area_size**2
    radio_range = math.sqrt(target_avg_degree * area / (math.pi * n_nodes))
    return radio_range


def generate_node_positions(n_nodes: int, area_size: int, random_seed: int) -> List[Tuple[float, float]]:
    """
    Generate random node positions uniformly distributed in square area.

    Args:
        n_nodes: Number of nodes
        area_size: Size of square area
        random_seed: Random seed for reproducibility

    Returns:
        List of (x, y) positions
    """
    random.seed(random_seed)
    positions = []

    for _ in range(n_nodes):
        x = random.uniform(0, area_size)
        y = random.uniform(0, area_size)
        positions.append((x, y))

    return positions


def calculate_contacts(positions: List[Tuple[float, float]], radio_range: float) -> Dict[int, List[int]]:
    """
    Calculate contact pattern based on Euclidean distance and radio range.

    Args:
        positions: List of node positions
        radio_range: Communication range

    Returns:
        Contact pattern as {node_id: [list of neighbor IDs]}
    """
    n_nodes = len(positions)
    contacts = {i: [] for i in range(n_nodes)}

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                xi, yi = positions[i]
                xj, yj = positions[j]
                distance = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

                if distance <= radio_range:
                    contacts[i].append(j)

    return contacts


def calculate_avg_degree(contacts: Dict[int, List[int]]) -> float:
    """Calculate average degree (connectivity) of the graph."""
    total_degree = sum(len(neighbors) for neighbors in contacts.values())
    return total_degree / len(contacts)


def generate_rgg_contact_pattern(n_nodes: int, n_epochs: int, area_size: int, density: str, random_seed: int, output_dir: str) -> str:
    """
    Generate RGG contact pattern with static topology.

    Args:
        n_nodes: Number of nodes
        n_epochs: Number of epochs (contact pattern is repeated)
        area_size: Size of square area
        density: 'dense' (avg_degree ≥ 10) or 'sparse' (avg_degree ≤ 4)
        random_seed: Random seed
        output_dir: Output directory

    Returns:
        Output file path
    """
    print(f"🔧 Generating RGG contact pattern for {n_nodes} nodes ({density} topology)...")

    # Determine target average degree based on density type
    if density == "dense":
        target_avg_degree = 10.0
        density_label = "d10"
    elif density == "sparse":
        target_avg_degree = 4.0
        density_label = "s04"
    else:
        raise ValueError(f"Invalid density: {density}. Must be 'dense' or 'sparse'")

    # Calculate required radio range
    radio_range = calculate_radio_range_for_density(n_nodes, area_size, target_avg_degree)
    print(f"  Target avg degree: {target_avg_degree}")
    print(f"  Calculated radio range: {radio_range:.2f}")

    # Generate node positions
    positions = generate_node_positions(n_nodes, area_size, random_seed)
    print(f"  Generated {len(positions)} node positions")

    # Calculate contacts
    contacts = calculate_contacts(positions, radio_range)
    actual_avg_degree = calculate_avg_degree(contacts)
    print(f"  Actual avg degree: {actual_avg_degree:.2f}")

    # Validate achieved density
    if density == "dense" and actual_avg_degree < 10.0:
        print(f"⚠️  Warning: Achieved avg_degree ({actual_avg_degree:.2f}) < 10.0")
        # Adjust radio range and recalculate
        radio_range *= 1.1
        contacts = calculate_contacts(positions, radio_range)
        actual_avg_degree = calculate_avg_degree(contacts)
        print(f"  Adjusted radio range to {radio_range:.2f}, new avg_degree: {actual_avg_degree:.2f}")
    elif density == "sparse" and actual_avg_degree > 4.0:
        print(f"⚠️  Warning: Achieved avg_degree ({actual_avg_degree:.2f}) > 4.0")
        # Adjust radio range and recalculate
        radio_range *= 0.9
        contacts = calculate_contacts(positions, radio_range)
        actual_avg_degree = calculate_avg_degree(contacts)
        print(f"  Adjusted radio range to {radio_range:.2f}, new avg_degree: {actual_avg_degree:.2f}")

    # Create contact pattern list (same for all epochs in RGG)
    contact_list = [contacts for _ in range(n_epochs)]

    # Save to JSON
    filename = f"rgg_n{n_nodes:02d}_a{area_size:04d}_{density_label}_s{random_seed:02d}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(contact_list, f, indent=2)

    print(f"✅ RGG contact pattern saved to {filepath}")
    print(f"   Nodes: {n_nodes}")
    print(f"   Epochs: {n_epochs}")
    print(f"   Avg Degree: {actual_avg_degree:.2f}")
    print(f"   Radio Range: {radio_range:.2f}")

    # Save metadata
    metadata = {
        "topology_type": "RGG",
        "n_nodes": n_nodes,
        "n_epochs": n_epochs,
        "area_size": area_size,
        "density": density,
        "target_avg_degree": target_avg_degree,
        "actual_avg_degree": actual_avg_degree,
        "radio_range": radio_range,
        "random_seed": random_seed,
        "positions": positions,
    }

    metadata_path = filepath.replace(".json", "_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"📋 Metadata saved to {metadata_path}")

    return filepath


def main():
    parser = argparse.ArgumentParser(description="Generate Random Geometric Graph (RGG) contact pattern for WAFL Testbed")
    parser.add_argument("--nodes", type=int, nargs="+", default=[28], help="Number of nodes (default: 28)")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of epochs (default: 5000)")
    parser.add_argument("--areasize", type=int, default=1000, help="Area size (default: 1000)")
    parser.add_argument(
        "--density",
        type=str,
        choices=["dense", "sparse"],
        default="dense",
        help="Topology density: 'dense' (avg_degree ≥ 10) or 'sparse' (avg_degree ≤ 4)",
    )
    parser.add_argument("--randomseed", type=int, default=1, help="Random seed (default: 1)")
    parser.add_argument("--output-dir", type=str, default="./data/contact_pattern", help="Output directory (default: ./data/contact_pattern)")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate RGG for each node count
    for n_nodes in args.nodes:
        print(f"\n{'=' * 60}")
        print(f"Generating RGG for {n_nodes} nodes")
        print(f"{'=' * 60}\n")

        generate_rgg_contact_pattern(
            n_nodes=n_nodes,
            n_epochs=args.epochs,
            area_size=args.areasize,
            density=args.density,
            random_seed=args.randomseed,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()

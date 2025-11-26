#!/usr/bin/env python3
"""
Visualize SUMO mobility simulation results.

This script creates visualizations for:
- Mobility traces (node positions over time)
- Contact patterns (network topology changes)
- Network conditions (quality distribution)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

try:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("❌ Error: matplotlib is required for visualization")
    print("Install with: pip install matplotlib")
    sys.exit(1)


def load_mobility_trace(trace_file: str) -> Dict:
    """Load mobility trace from CSV file."""
    import csv

    trace = {}
    with open(trace_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row["epoch"])
            node_id = row["node_id"]
            x = float(row["x"])
            y = float(row["y"])

            if epoch not in trace:
                trace[epoch] = {}
            trace[epoch][node_id] = (x, y)

    return trace


def load_contact_pattern(contact_file: str) -> List:
    """Load contact pattern from JSON file."""
    with open(contact_file) as f:
        return json.load(f)


def load_network_conditions(conditions_file: str) -> List:
    """Load network conditions from JSON file."""
    with open(conditions_file) as f:
        return json.load(f)


def visualize_mobility_trace(trace: Dict, network_conditions: List = None, output_file: str = None):
    """
    Create an animated visualization of node mobility with network conditions.

    Args:
        trace: Mobility trace data
        network_conditions: Optional network conditions to visualize connections
        output_file: Optional path to save animation (MP4 or GIF)
    """
    print("📊 Creating mobility trace visualization...")

    epochs = sorted(trace.keys())
    all_nodes = set()
    for epoch_data in trace.values():
        all_nodes.update(epoch_data.keys())

    node_ids = sorted(all_nodes, key=lambda x: int(x))

    # Find bounds
    all_x = [pos[0] for epoch_data in trace.values() for pos in epoch_data.values()]
    all_y = [pos[1] for epoch_data in trace.values() for pos in epoch_data.values()]

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    # Add padding
    padding = 50
    min_x -= padding
    max_x += padding
    min_y -= padding
    max_y += padding

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X Position (m)", fontsize=12)
    ax.set_ylabel("Y Position (m)", fontsize=12)

    # Color map for nodes
    colors = plt.cm.tab10(np.linspace(0, 1, len(node_ids)))
    node_colors = {node_id: colors[i] for i, node_id in enumerate(node_ids)}

    # Color map for network quality ranks
    rank_colors = {
        "excellent": "#2ecc71",  # Green
        "good": "#3498db",  # Blue
        "fair": "#f39c12",  # Orange
        "poor": "#e74c3c",  # Red
    }

    rank_widths = {
        "excellent": 2.5,
        "good": 2.0,
        "fair": 1.5,
        "poor": 1.0,
    }

    # Initialize scatter plot for nodes
    scatters = {}
    for node_id in node_ids:
        scatters[node_id] = ax.scatter([], [], s=300, c=[node_colors[node_id]], label=f"Node {node_id}", alpha=0.8, edgecolors="black", linewidths=2, zorder=10)

    # Trails
    trails = {node_id: ax.plot([], [], "-", color=node_colors[node_id], alpha=0.2, linewidth=1.5, zorder=1)[0] for node_id in node_ids}

    # Network connections (will be updated each frame)
    connection_lines = []

    title = ax.text(0.5, 1.02, "", transform=ax.transAxes, ha="center", fontsize=16, weight="bold")

    # Create legend for network quality
    if network_conditions:
        legend_elements = [plt.Line2D([0], [0], color=rank_colors[rank], lw=rank_widths[rank], label=rank.capitalize()) for rank in ["excellent", "good", "fair", "poor"]]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=10, title="Network Quality")
    else:
        ax.legend(loc="upper right", fontsize=8)

    # Store trail data
    trail_data = {node_id: {"x": [], "y": []} for node_id in node_ids}

    def init():
        for scatter in scatters.values():
            scatter.set_offsets(np.empty((0, 2)))
        for trail in trails.values():
            trail.set_data([], [])
        return list(scatters.values()) + list(trails.values()) + [title] + connection_lines

    def update(frame):
        nonlocal connection_lines

        epoch = epochs[frame]
        epoch_data = trace[epoch]

        # Clear previous connection lines
        for line in connection_lines:
            line.remove()
        connection_lines = []

        # Draw network connections if available
        if network_conditions and frame < len(network_conditions):
            epoch_conditions = network_conditions[frame]
            node_positions = {}

            # Get current positions - node_ids in trace are strings
            for node_id_str in node_ids:
                if node_id_str in epoch_data:
                    # Store with both string and int keys for flexibility
                    node_positions[node_id_str] = epoch_data[node_id_str]
                    node_positions[int(node_id_str)] = epoch_data[node_id_str]

            # Draw connections
            drawn_pairs = set()
            for node_key, peers_list in epoch_conditions.items():
                # Handle both string and int keys
                if isinstance(node_key, str):
                    node_key = int(node_key)

                if node_key not in node_positions:
                    continue

                x1, y1 = node_positions[node_key]

                for peer_info in peers_list:
                    peer_id = peer_info.get("peer")
                    if peer_id not in node_positions:
                        continue

                    # Avoid drawing the same connection twice
                    pair = tuple(sorted([node_key, peer_id]))
                    if pair in drawn_pairs:
                        continue
                    drawn_pairs.add(pair)

                    x2, y2 = node_positions[peer_id]
                    rank = peer_info.get("rank", "poor")

                    line = ax.plot([x1, x2], [y1, y2], color=rank_colors.get(rank, "#95a5a6"), linewidth=rank_widths.get(rank, 1.0), alpha=0.6, zorder=5)[0]
                    connection_lines.append(line)

        # Update node positions
        for node_id in node_ids:
            if node_id in epoch_data:
                x, y = epoch_data[node_id]
                scatters[node_id].set_offsets([[x, y]])

                # Update trail
                trail_data[node_id]["x"].append(x)
                trail_data[node_id]["y"].append(y)

                # Keep only last N points for trail
                max_trail = 50
                if len(trail_data[node_id]["x"]) > max_trail:
                    trail_data[node_id]["x"] = trail_data[node_id]["x"][-max_trail:]
                    trail_data[node_id]["y"] = trail_data[node_id]["y"][-max_trail:]

                trails[node_id].set_data(trail_data[node_id]["x"], trail_data[node_id]["y"])
            else:
                scatters[node_id].set_offsets(np.empty((0, 2)))

        title.set_text(f"Mobility Trace with Network Conditions - Epoch {epoch} / {epochs[-1]}")

        return list(scatters.values()) + list(trails.values()) + connection_lines + [title]

    # Create animation
    anim = animation.FuncAnimation(fig, update, init_func=init, frames=len(epochs), interval=100, blit=True)

    if output_file:
        print(f"💾 Saving animation to {output_file}...")
        if output_file.endswith(".gif"):
            anim.save(output_file, writer="pillow", fps=10)
        elif output_file.endswith(".mp4"):
            anim.save(output_file, writer="ffmpeg", fps=10)
        else:
            print("⚠️  Unknown output format, saving as GIF")
            anim.save(output_file + ".gif", writer="pillow", fps=10)
        print(f"✅ Animation saved to {output_file}")
    else:
        plt.tight_layout()
        plt.show()

    plt.close()


def visualize_contact_pattern(contact_pattern: List, trace: Dict, output_file: str = None):
    """
    Visualize network connectivity over time.

    Args:
        contact_pattern: Contact pattern data
        trace: Mobility trace for node positions
        output_file: Optional path to save figure
    """
    print("📊 Creating contact pattern visualization...")

    # Count contacts per node per epoch
    epochs = range(len(contact_pattern))
    all_nodes = set()
    for epoch_contacts in contact_pattern:
        all_nodes.update(epoch_contacts.keys())

    node_ids = sorted(all_nodes)

    # Calculate contact counts
    contact_counts = {node_id: [] for node_id in node_ids}

    for epoch_contacts in contact_pattern:
        for node_id in node_ids:
            if node_id in epoch_contacts:
                contact_counts[node_id].append(len(epoch_contacts[node_id]))
            else:
                contact_counts[node_id].append(0)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Contact counts over time
    for node_id in node_ids:
        ax1.plot(epochs, contact_counts[node_id], label=f"Node {node_id}", marker="o", markersize=3, alpha=0.7)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Number of Contacts")
    ax1.set_title("Contact Pattern Evolution Over Time")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Average contacts and network density
    avg_contacts = [sum(counts) / len(node_ids) for counts in zip(*contact_counts.values())]
    max_possible = len(node_ids) * (len(node_ids) - 1)  # Maximum possible connections
    density = [sum(counts) / max_possible if max_possible > 0 else 0 for counts in zip(*contact_counts.values())]

    ax2_density = ax2.twinx()

    line1 = ax2.plot(epochs, avg_contacts, "b-", label="Avg Contacts per Node", linewidth=2)
    line2 = ax2_density.plot(epochs, density, "r--", label="Network Density", linewidth=2)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Average Contacts", color="b")
    ax2_density.set_ylabel("Network Density", color="r")
    ax2.set_title("Network Connectivity Metrics")
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis="y", labelcolor="b")
    ax2_density.tick_params(axis="y", labelcolor="r")

    # Combine legends
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax2.legend(lines, labels, loc="best")

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"✅ Contact pattern visualization saved to {output_file}")
    else:
        plt.show()

    plt.close()


def visualize_network_conditions(network_conditions: List, output_file: str = None):
    """
    Visualize network quality distribution over time.

    Args:
        network_conditions: Network conditions data
        output_file: Optional path to save figure
    """
    print("📊 Creating network conditions visualization...")

    # Count rank distribution
    rank_counts = {"excellent": [], "good": [], "fair": [], "poor": []}
    epochs = range(len(network_conditions))

    for epoch_cond in network_conditions:
        epoch_ranks = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}

        for node_id, peers_list in epoch_cond.items():
            for peer_info in peers_list:
                rank = peer_info.get("rank", "unknown")
                if rank in epoch_ranks:
                    epoch_ranks[rank] += 1

        for rank in rank_counts.keys():
            rank_counts[rank].append(epoch_ranks[rank])

    # Create stacked area chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Stacked area chart
    colors = {"excellent": "#2ecc71", "good": "#3498db", "fair": "#f39c12", "poor": "#e74c3c"}

    ax1.stackplot(epochs, rank_counts["excellent"], rank_counts["good"], rank_counts["fair"], rank_counts["poor"], labels=["Excellent", "Good", "Fair", "Poor"], colors=[colors["excellent"], colors["good"], colors["fair"], colors["poor"]], alpha=0.8)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Number of Connections")
    ax1.set_title("Network Quality Distribution Over Time")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Percentage distribution
    total_connections = [sum(counts) for counts in zip(*rank_counts.values())]

    for rank, color in colors.items():
        percentages = [(rank_counts[rank][i] / total_connections[i] * 100) if total_connections[i] > 0 else 0 for i in range(len(epochs))]
        ax2.plot(epochs, percentages, label=rank.capitalize(), color=color, linewidth=2)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("Network Quality Distribution (Percentage)")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"✅ Network conditions visualization saved to {output_file}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize SUMO mobility simulation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize mobility trace (animated)
  python visualize_sumo_results.py --trace data/sumo/mobility_trace.csv --mode mobility
  
  # Visualize contact pattern
  python visualize_sumo_results.py \\
    --trace data/sumo/mobility_trace.csv \\
    --contact data/sumo/contact_pattern_mobility.json \\
    --mode contact
  
  # Visualize network conditions
  python visualize_sumo_results.py \\
    --conditions data/sumo/network_conditions_mobility.json \\
    --mode conditions
  
  # Save visualizations
  python visualize_sumo_results.py \\
    --trace data/sumo/mobility_trace.csv \\
    --mode mobility \\
    --output mobility_animation.gif
        """,
    )

    parser.add_argument(
        "--trace",
        type=str,
        help="Path to mobility trace CSV file",
    )
    parser.add_argument(
        "--contact",
        type=str,
        help="Path to contact pattern JSON file",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        help="Path to network conditions JSON file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["mobility", "contact", "conditions", "all"],
        default="all",
        help="Visualization mode (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (optional, defaults to interactive display)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/sumo/visuals",
        help="Output directory for multiple visualizations (default: data/sumo/visuals)",
    )

    args = parser.parse_args()

    # Determine what to visualize
    modes_to_run = []
    if args.mode == "all":
        if args.trace:
            modes_to_run.extend(["mobility", "contact"])
        if args.conditions:
            modes_to_run.append("conditions")
    else:
        modes_to_run = [args.mode]

    if not modes_to_run:
        print("❌ Error: No visualization mode specified or required files missing")
        sys.exit(1)

    # Load data
    trace = None
    contact_pattern = None
    network_conditions = None

    if "mobility" in modes_to_run or "contact" in modes_to_run:
        if not args.trace or not Path(args.trace).exists():
            print(f"❌ Error: Trace file required for {modes_to_run} mode")
            sys.exit(1)
        trace = load_mobility_trace(args.trace)
        print(f"✅ Loaded mobility trace: {len(trace)} epochs")

    if "contact" in modes_to_run:
        if not args.contact or not Path(args.contact).exists():
            print("❌ Error: Contact pattern file required for contact mode")
            sys.exit(1)
        contact_pattern = load_contact_pattern(args.contact)
        print(f"✅ Loaded contact pattern: {len(contact_pattern)} epochs")

    # Load network conditions if available (for mobility visualization)
    if "mobility" in modes_to_run or "conditions" in modes_to_run:
        if args.conditions and Path(args.conditions).exists():
            network_conditions = load_network_conditions(args.conditions)
            print(f"✅ Loaded network conditions: {len(network_conditions)} epochs")
        elif "conditions" in modes_to_run:
            print("❌ Error: Network conditions file required for conditions mode")
            sys.exit(1)

    # Create visualizations
    if len(modes_to_run) > 1 and not args.output:
        # Multiple visualizations, save to directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Saving visualizations to {output_dir}/")

        if "mobility" in modes_to_run:
            visualize_mobility_trace(trace, network_conditions, str(output_dir / "mobility_trace.gif"))
        if "contact" in modes_to_run:
            visualize_contact_pattern(contact_pattern, trace, str(output_dir / "contact_pattern.png"))
        if "conditions" in modes_to_run:
            visualize_network_conditions(network_conditions, str(output_dir / "network_conditions.png"))
    else:
        # Single visualization
        output_file = args.output

        if "mobility" in modes_to_run:
            visualize_mobility_trace(trace, network_conditions, output_file)
        elif "contact" in modes_to_run:
            visualize_contact_pattern(contact_pattern, trace, output_file)
        elif "conditions" in modes_to_run:
            visualize_network_conditions(network_conditions, output_file)

    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    main()

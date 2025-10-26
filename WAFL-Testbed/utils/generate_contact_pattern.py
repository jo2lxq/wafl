import argparse
import json
import math
import os
import random
import shutil

import matplotlib
from tqdm import tqdm

matplotlib.use("Agg")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image


def snapshot(x, y, t, areasize, radio_range, images_dir, title=""):
    """Generate a snapshot of node positions and connections."""
    plt.clf()
    plt.scatter(x, y, s=20, c="Black", marker="o")
    n_node = len(x)
    for i in range(n_node):
        c = patches.Circle((x[i], y[i]), radius=radio_range, fc="b", fill=False)
        plt.gca().add_patch(c)
        plt.annotate(str(i), (x[i] + 5, y[i] + 5))
    for i in range(n_node):
        for j in range(i):
            if ((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2) <= radio_range**2:
                px = [x[i], x[j]]
                py = [y[i], y[j]]
                plt.plot(px, py, color="Black")
    plt.xlim([0, areasize])
    plt.ylim([0, areasize])
    plt.gca().set_aspect("equal")
    plt.xlabel("X")
    plt.ylabel("Y")
    if title:
        plt.title(title, fontsize=12)
    plt.savefig(os.path.join(images_dir, f"node_location_{t:04d}.png"))


def generate_contact_pattern(
    n_time,
    n_node,
    min_travel_speed,
    max_travel_speed,
    radio_range,
    areasize,
    pose_time,
    randomseed,
    output_dir,
    generate_animation=False,
):
    """Generate contact pattern using Random Waypoint mobility model."""
    print(f"🚀 Starting contact pattern generation for {n_node} nodes...")
    area = (areasize, areasize)
    random.seed(randomseed)

    # Initialize node positions and parameters
    node_location = [None] * n_node
    node_travel_speed = [None] * n_node
    node_pose_remaining_time = [pose_time] * n_node
    node_next_location = [None] * n_node

    for i in range(n_node):
        max_x, max_y = area
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        node_location[i] = (x, y)

    contact_list = []

    # Create directories for output
    filename = f"rwp_n{n_node:02d}_a{areasize:04d}_r{radio_range:03d}_p{pose_time:02d}_s{randomseed:02d}"
    images_dir = os.path.join(output_dir, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    animation_frames = []

    # Main simulation loop
    for t in tqdm(range(n_time), desc="Simulating time steps", unit="step"):
        if generate_animation:
            px, py = zip(*node_location)
            title = f"{filename} | Epoch: {t}"
            snapshot(px, py, t, areasize, radio_range, images_dir, title)
            animation_frames.append(os.path.join(images_dir, f"node_location_{t:04d}.png"))

        # Update node positions
        for i in range(n_node):
            if node_pose_remaining_time[i] == 0:
                x, y = node_location[i]
                tx, ty = node_next_location[i]
                ax, ay = tx - x, ty - y
                distance = math.sqrt(ax**2 + ay**2)
                vx, vy = node_travel_speed[i] * ax / distance, node_travel_speed[i] * ay / distance
                x, y = x + vx, y + vy

                if (x - tx) ** 2 + (y - ty) ** 2 < node_travel_speed[i] ** 2:
                    node_location[i] = (tx, ty)
                    node_pose_remaining_time[i] = pose_time
                else:
                    node_location[i] = (x, y)
            else:
                node_pose_remaining_time[i] -= 1
                if node_pose_remaining_time[i] == 0:
                    max_x, max_y = area
                    node_next_location[i] = (random.randint(0, max_x), random.randint(0, max_y))
                    node_travel_speed[i] = random.uniform(min_travel_speed, max_travel_speed)

        # Calculate contacts
        node_in_contact = {i: [] for i in range(n_node)}
        for i in range(n_node):
            for j in range(n_node):
                if i != j:
                    xi, yi = node_location[i]
                    xj, yj = node_location[j]
                    if (xi - xj) ** 2 + (yi - yj) ** 2 < radio_range**2:
                        node_in_contact[i].append(j)
        contact_list.append(node_in_contact)

    # Save contact pattern to JSON
    filepath = os.path.join(output_dir, f"{filename}.json")
    with open(filepath, "w") as f:
        json.dump(contact_list, f, indent=4)
    print(f"✅ Contact pattern saved to {filepath}")

    # Generate GIF animation
    if generate_animation and animation_frames:
        gif_path = os.path.join(output_dir, f"{filename}.gif")
        images = [Image.open(frame) for frame in animation_frames[:256]]
        if images:
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                duration=64,
                loop=0,
            )
            print(f"🎥 GIF animation saved to {gif_path}")

        # Clean up individual frame images
        for frame_path in animation_frames:
            if os.path.exists(frame_path):
                os.remove(frame_path)
        os.rmdir(images_dir)
        print("🧹 Cleaned up individual frame images.")


def main():
    parser = argparse.ArgumentParser(description="Generate contact pattern using Random Waypoint mobility model")
    parser.add_argument("--times", type=int, default=5000, help="Number of time steps (default: 5000)")
    parser.add_argument("--nodes", type=int, nargs="+", default=[10], help="Number of nodes (default: 10)")
    parser.add_argument("--min-speed", type=float, default=3, help="Minimum travel speed (default: 3)")
    parser.add_argument("--max-speed", type=float, default=7, help="Maximum travel speed (default: 7)")
    parser.add_argument("--radio-range", type=int, default=100, help="Radio range (default: 100)")
    parser.add_argument("--pose-time", type=int, default=10, help="Pose time (default: 10)")
    parser.add_argument("--areasize", type=int, default=500, help="Area size (default: 500)")
    parser.add_argument("--randomseed", type=int, default=1, help="Random seed (default: 1)")
    parser.add_argument("--output-dir", type=str, default="./data/contact_pattern", help="Output directory")
    parser.add_argument("--animation", action="store_true", help="Generate animation snapshots and trajectories")
    parser.add_argument("--clean", action="store_true", help="Clean output directory before generating")

    args = parser.parse_args()

    # Clean output directory if requested
    if args.clean and os.path.exists(args.output_dir):
        print(f"🧹 Cleaning output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Generate contact patterns for all node configurations
    for n_node in args.nodes:
        print("\n🔧 Generating contact pattern with parameters:")
        print(f"  times: {args.times}")
        print(f"  nodes: {n_node}")
        print(f"  min_speed: {args.min_speed}")
        print(f"  max_speed: {args.max_speed}")
        print(f"  radio_range: {args.radio_range}")
        print(f"  areasize: {args.areasize}")
        print(f"  pose_time: {args.pose_time}")
        print(f"  randomseed: {args.randomseed}")
        print(f"  animation: {args.animation}")
        print(f"  output_dir: {args.output_dir}")

        generate_contact_pattern(
            args.times,
            n_node,
            args.min_speed,
            args.max_speed,
            args.radio_range,
            args.areasize,
            args.pose_time,
            args.randomseed,
            args.output_dir,
            args.animation,
        )


if __name__ == "__main__":
    main()

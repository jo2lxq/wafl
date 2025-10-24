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


def snapshot(x, y, t, areasize, radio_range, animation_dir):
    """Generate snapshot visualization of node positions and connections."""
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
    plt.savefig(os.path.join(animation_dir, f"node_location_{t:04d}.png"))


def trajectory(x, y, start, end, areasize, trajectory_dir):
    """Generate trajectory visualization."""
    plt.clf()
    cmap = plt.get_cmap()
    n_node = len(x)
    for i in range(n_node):
        for t in range(len(x[i]) - 1):
            plt.plot(x[i][t : t + 2], y[i][t : t + 2], color=cmap(i))
        plt.annotate(str(i), (x[i][0] + 5, y[i][0] + 5), fontsize=14, color=cmap(i))
        plt.scatter([x[i][0]], [y[i][0]], s=50, color=cmap(i), marker="o")
        plt.scatter([x[i][len(x[i]) - 1]], [y[i][len(x[i]) - 1]], s=100, color=cmap(i), marker="*")
    plt.xlim([0, areasize])
    plt.ylim([0, areasize])
    plt.gca().set_aspect("equal")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(os.path.join(trajectory_dir, f"node_trajectory_{start:04d}_{end:04d}.png"))


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
    area = (areasize, areasize)
    random.seed(randomseed)

    node_location = [None] * n_node
    node_travel_speed = [None] * n_node
    node_pose_remaining_time = [pose_time] * n_node
    node_next_location = [None] * n_node

    # Initialize node locations
    for i in range(n_node):
        max_x, max_y = area
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        node_location[i] = (x, y)

    contact_list = []

    # Create filename for contact pattern
    filename = f"rwp_n{n_node:02d}_a{areasize:04d}_r{radio_range:03d}_p{pose_time:02d}_s{randomseed:02d}"

    # Create subdirectory for this contact pattern
    animation_dir = os.path.join(
        output_dir,
        "animation",
        filename,
    )
    trajectory_dir = os.path.join(
        output_dir,
        "trajectory",
        filename,
    )

    if not os.path.exists(animation_dir):
        os.makedirs(animation_dir)
    if not os.path.exists(trajectory_dir):
        os.makedirs(trajectory_dir)

    x_trajectory = [[] for i in range(n_node)]
    y_trajectory = [[] for i in range(n_node)]

    animation_frames = []

    for t in tqdm(range(n_time), desc="Generating contact pattern", unit="step"):
        # Generate snapshots if animation is enabled
        if generate_animation:
            px = []
            py = []
            for i in range(n_node):
                x, y = node_location[i]
                px.append(x)
                py.append(y)
                x_trajectory[i].append(x)
                y_trajectory[i].append(y)

            snapshot_path = os.path.join(animation_dir, f"node_location_{t:04d}.png")
            snapshot(px, py, t, areasize, radio_range, animation_dir)
            animation_frames.append(snapshot_path)

            # Generate trajectory every 100 steps
            if t % 100 == 99:
                # trajectory(x_trajectory, y_trajectory, int(t / 100) * 100, t, areasize, trajectory_dir)
                x_trajectory = [[] for i in range(n_node)]
                y_trajectory = [[] for i in range(n_node)]

        # Update node positions
        for i in range(n_node):
            if node_pose_remaining_time[i] == 0:
                x, y = node_location[i]
                tx, ty = node_next_location[i]
                ax = tx - x
                ay = ty - y
                vx = node_travel_speed[i] * ax / math.sqrt(ax**2 + ay**2)
                vy = node_travel_speed[i] * ay / math.sqrt(ax**2 + ay**2)

                x += vx
                y += vy

                if (x - tx) ** 2 + (y - ty) ** 2 < node_travel_speed[i] ** 2:
                    node_location[i] = tx, ty
                    node_travel_speed[i] = None
                    node_next_location[i] = None
                    node_pose_remaining_time[i] = pose_time
                else:
                    node_location[i] = x, y
            else:
                node_pose_remaining_time[i] -= 1
                if node_pose_remaining_time[i] == 0:
                    max_x, max_y = area
                    x = random.randint(0, max_x)
                    y = random.randint(0, max_y)
                    node_next_location[i] = (x, y)
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

    print(f"Contact pattern saved to {filepath}")

    # Generate GIF animation if animation is enabled
    if generate_animation and animation_frames:
        gif_path = os.path.join(animation_dir, "animation.gif")
        images = []
        # Sample frames to reduce GIF size (e.g., every 10th frame)
        sampled_frames = animation_frames[::10] if len(animation_frames) > 100 else animation_frames
        for frame_path in sampled_frames:
            images.append(Image.open(frame_path))

        if images:
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                duration=100,  # milliseconds per frame
                loop=0,
            )
            print(f"GIF animation saved to {gif_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate contact pattern using Random Waypoint mobility model")
    parser.add_argument("--times", type=int, default=5000, help="Number of time steps (default: 5000)")
    parser.add_argument("--nodes", type=int, default=10, help="Number of nodes (default: 10)")
    parser.add_argument("--min-speed", type=float, default=3, help="Minimum travel speed (default: 3)")
    parser.add_argument("--max-speed", type=float, default=7, help="Maximum travel speed (default: 7)")
    parser.add_argument("--radio-range", type=int, default=100, help="Radio range (default: 100)")
    parser.add_argument(
        "--pose-time",
        type=int,
        nargs="+",
        default=[10, 40, 100],
        help="Pose time values (default: 10 40 100)",
    )
    parser.add_argument("--areasize", type=int, nargs="+", default=[500], help="Area size values (default: 500)")
    parser.add_argument("--randomseed", type=int, nargs="+", default=[1], help="Random seed values (default: 1)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/contact_pattern",
        help="Output directory (default: ./data/contact_pattern)",
    )
    parser.add_argument(
        "--animation",
        action="store_true",
        help="Generate animation snapshots and trajectories",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean output directory before generating",
    )

    args = parser.parse_args()

    # Clean output directory if requested
    if args.clean and os.path.exists(args.output_dir):
        print(f"Cleaning output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Generate contact patterns for all parameter combinations
    parameters = []
    for areasize in args.areasize:
        for pose_time in args.pose_time:
            for randomseed in args.randomseed:
                parameters.append((areasize, pose_time, randomseed))

    for areasize, pose_time, randomseed in parameters:
        print(f"\nGenerating contact pattern: areasize={areasize}, pose_time={pose_time}, randomseed={randomseed}")
        generate_contact_pattern(
            args.times,
            args.nodes,
            args.min_speed,
            args.max_speed,
            args.radio_range,
            areasize,
            pose_time,
            randomseed,
            args.output_dir,
            args.animation,
        )


if __name__ == "__main__":
    main()

import argparse
import os
import re
import subprocess
import sys
from typing import Dict, Optional, Tuple

"""
WAFL Emulation Testbed: Result Collection Program

This script collects experiment artifacts from each WAFL execution server
and copies them to the control server.

# Prerequisites:
1.  This script is in the same directory as `execution_config`.
2.  Passwordless SSH access is configured from the control server to all
    execution servers (e.g., using public key authentication).
3.  The `scp` command is available and in the system's PATH.

# Usage:
This script should be placed in the project's `ctrl` directory.

# To collect a specific experiment:
$ python ctrl/collect.py [experiment_id]

# To collect the latest experiment automatically:
$ python ctrl/collect.py
"""


def get_project_paths() -> Tuple[str, str, str]:
    """
    Determines project-related paths based on the script's location.
    Assumes the script is located at `PROJECT_NAME/ctrl/collect.py`.

    Returns:
        A tuple containing (project_name, local_project_root, config_file_path).
    """
    try:
        script_path = os.path.abspath(__file__)
        ctrl_dir = os.path.dirname(script_path)
        project_root_path = os.path.dirname(ctrl_dir)
        project_name = os.path.basename(project_root_path)
        config_path = os.path.join(ctrl_dir, "execution_config")
        return project_name, project_root_path, config_path
    except Exception as e:
        print(f"Error: Could not determine project paths: {e}", file=sys.stderr)
        print("Please run this script from the 'PROJECT_NAME/ctrl/' directory.", file=sys.stderr)
        sys.exit(1)


def get_config_from_file(config_path: str) -> Dict[str, str]:
    """
    Parses the `execution_config` file into a dictionary.

    Args:
        config_path: The path to the configuration file.

    Returns:
        A dictionary of configuration key-value pairs.
    """
    config = {}
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # Regex to match lines like: export KEY='VALUE' or export KEY=VALUE
    pattern = re.compile(r"^\s*export\s+([A-Za-z0-9_]+)=(.*)")

    with open(config_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                key, value = match.groups()
                # Strip leading/trailing quotes
                if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
                    value = value[1:-1]
                config[key] = value
    return config


def find_latest_experiment(results_path: str) -> Optional[str]:
    """
    Finds the most recently modified directory (experiment) in the results folder.

    Args:
        results_path: The path to the 'results' directory.

    Returns:
        The name of the latest experiment directory, or None if not found.
    """
    if not os.path.isdir(results_path):
        return None

    all_dirs = [d for d in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, d))]

    if not all_dirs:
        return None

    latest_dir = max(all_dirs, key=lambda d: os.path.getmtime(os.path.join(results_path, d)))
    return latest_dir


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Collects experiment results from WAFL execution servers.", formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "experiment_id",
        nargs="?",  # Make argument optional
        default=None,
        help="ID of the experiment to collect (e.g., 'my-exp-20250916T150000').\n"
        "If omitted, the latest experiment will be used.",
    )
    return parser.parse_args()


def main():
    """
    Main execution function.
    """
    # 1. Initialize paths and arguments
    args = parse_arguments()
    project_name, local_project_path, config_path = get_project_paths()

    # 2. Load configuration from file
    config = get_config_from_file(config_path)
    try:
        device_names = config["WAFL_DEVICE_NAMES"].split(",")
        device_ips = config["WAFL_DEVICE_IPS"].split(",")
        ssh_user = config["USER"]
        remote_base_path = config["DEPLOYMENT_LOCATION"]
    except KeyError as e:
        print(f"❌ Error: Required key '{e}' not found in configuration file.", file=sys.stderr)
        sys.exit(1)

    if len(device_names) != len(device_ips):
        print("❌ Error: WAFL_DEVICE_NAMES and WAFL_DEVICE_IPS have a different number of items.", file=sys.stderr)
        sys.exit(1)

    # 3. Determine the target experiment ID
    experiment_id = args.experiment_id
    if experiment_id is None:
        print("🔍 Experiment ID not specified. Searching for the latest experiment...")
        results_path = os.path.join(local_project_path, "results")
        experiment_id = find_latest_experiment(results_path)
        if experiment_id is None:
            print(f"❌ Error: No experiment directories found in {results_path}", file=sys.stderr)
            sys.exit(1)
        print(f"✅ Found latest experiment: '{experiment_id}'")

    # 4. Start the collection process
    remote_project_path = os.path.join(remote_base_path, project_name)

    print("\n🚀 --- Starting Result Collection ---")
    print(f"Project Name:     {project_name}")
    print(f"Experiment ID:    {experiment_id}")
    print(f"SSH User:         {ssh_user}")
    print(f"Target Devices:   {len(device_ips)}")
    print("------------------------------------")

    # 5. Loop through each device and collect its results
    for device_name, device_ip in zip(device_names, device_ips):
        print(f"\n🖥️  [Device {device_name} ({device_ip})] Processing...")

        # Define source and destination paths
        remote_source_path = os.path.join(remote_project_path, "results", experiment_id)
        remote_full_path = f"{ssh_user}@{device_ip}:{remote_source_path}"
        local_dest_path = os.path.join(local_project_path, "results", experiment_id, device_name)

        # Create local directory if it doesn't exist
        try:
            os.makedirs(local_dest_path, exist_ok=True)
            print(f"  📂 Verified local directory: {local_dest_path}")
        except OSError as e:
            print(f"  ❌ Error: Failed to create directory. {e}", file=sys.stderr)
            continue

        # Build and run the scp command
        command = ["scp", "-rp", f"{remote_full_path}/*", local_dest_path]

        print(f"  💨 Executing: {' '.join(command)}")
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
            if result.stderr:  # scp often prints status to stderr
                print(f"  - SCP output:\n{result.stderr.strip()}")
            print(f"  ✅ Success: Collected results from device {device_name}.")
        except FileNotFoundError:
            print("  ❌ Error: 'scp' command not found. Please check your system's PATH.", file=sys.stderr)
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Error: Failed to collect from device {device_name} (return code: {e.returncode}).", file=sys.stderr)
            print(f"  - STDERR:\n{e.stderr.strip()}", file=sys.stderr)
        except Exception as e:
            print(f"  ❌ Error: An unexpected error occurred: {e}", file=sys.stderr)

    print("\n🎉 --- Collection process finished. ---")


if __name__ == "__main__":
    main()

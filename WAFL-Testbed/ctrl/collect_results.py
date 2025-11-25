import json
import subprocess
import sys
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CTRL_DIR = PROJECT_ROOT / "ctrl"
CONFIG_FILE = CTRL_DIR / "execution_config.json"
RESULTS_DIR = PROJECT_ROOT / "results"

# SSH Settings
SSH_OPTS = [
    "-o",
    "ConnectTimeout=10",
    "-o",
    "ServerAliveInterval=60",
    "-o",
    "ServerAliveCountMax=3",
    "-o",
    "StrictHostKeyChecking=no",
]
CTL_DIR = Path.home() / ".ssh" / "ctl"
MUX_OPTS = ["-o", "ControlMaster=auto", "-o", f"ControlPath={CTL_DIR}/%C", "-o", "ControlPersist=600"]


def load_config():
    if not CONFIG_FILE.exists():
        print(f"❌ Error: Configuration file not found: {CONFIG_FILE}")
        sys.exit(1)

    with open(CONFIG_FILE) as f:
        return json.load(f)


def run_command(cmd, shell=False):
    try:
        subprocess.run(cmd, check=True, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e.cmd}")
        print(f"Error output: {e.stderr.decode()}")
        return False


def collect_from_node(node, config, deployment_location, user, experiment_id):
    device_name = str(node["id"])
    ip = node["physical_ip"]
    host = f"{user}@{ip}"

    # Remote results path: <deploy_loc>/WAFL-Testbed/results/<experiment_id>
    # Note: We need to know the experiment ID.
    # Option 1: Pass it as argument.
    # Option 2: Fetch the latest directory from remote.

    # Let's assume we want to collect EVERYTHING in results/ from remote to local results/remote/<node_id>/
    remote_results_dir = f"{deployment_location}/WAFL-Testbed/results"
    local_node_dir = RESULTS_DIR / "collected" / device_name
    local_node_dir.mkdir(parents=True, exist_ok=True)

    print(f"📥 Collecting results from Node {device_name} ({ip})...")

    # Use rsync if available, else scp
    # scp -r user@ip:remote_dir/* local_dir/
    cmd = ["scp"] + SSH_OPTS + MUX_OPTS + ["-r", "-q", f"{host}:{remote_results_dir}/*", str(local_node_dir)]

    if run_command(cmd):
        print(f"✅ Collected results from Node {device_name}")
        return True
    else:
        print(f"❌ Failed to collect from Node {device_name}")
        return False


def main():
    config = load_config()

    deployment_location = config.get("deployment_location", "/home/denjo")
    user = config.get("user", "denjo")
    nodes = config.get("nodes", [])

    print(f"🚀 Starting result collection from {len(nodes)} nodes")

    # Create local collection directory
    (RESULTS_DIR / "collected").mkdir(parents=True, exist_ok=True)

    success_count = 0
    for node in nodes:
        if collect_from_node(node, config, deployment_location, user, None):
            success_count += 1

    print(f"📈 Collection Summary: {success_count}/{len(nodes)} nodes collected")
    print(f"📂 Results saved to: {RESULTS_DIR}/collected")


if __name__ == "__main__":
    main()

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CTRL_DIR = PROJECT_ROOT / "ctrl"
CONFIG_FILE = CTRL_DIR / "execution_config.json"
RESULTS_DIR = PROJECT_ROOT / "results" / ".deploy"
LOG_FILE = RESULTS_DIR / "ctrl.log"

# Registry URL (management server runs registry on port 5000)
# Use REGISTRY_HOST from .env (loaded by mise)
REGISTRY_HOST = os.environ.get("REGISTRY_HOST", "localhost")
REGISTRY_URL = f"{REGISTRY_HOST}:5000"

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
MUX_OPTS = [
    "-o",
    "ControlMaster=auto",
    "-o",
    f"ControlPath={CTL_DIR}/%C",
    "-o",
    "ControlPersist=600",
]


def setup_logging():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CTL_DIR.mkdir(parents=True, exist_ok=True)
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    # Clear device logs
    for f in RESULTS_DIR.glob("wafl*.log"):
        f.unlink()


def log(message, device=None):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"
    print(formatted_msg, flush=True)

    with open(LOG_FILE, "a") as f:
        f.write(formatted_msg + "\n")

    if device:
        device_log = RESULTS_DIR / f"wafl{device}.log"
        with open(device_log, "a") as f:
            f.write(formatted_msg + "\n")


def load_config():
    if not CONFIG_FILE.exists():
        log(f"❌ Error: Configuration file not found: {CONFIG_FILE}")
        sys.exit(1)

    with open(CONFIG_FILE) as f:
        return json.load(f)


def run_command(cmd, shell=False):
    """Run a command and return success status."""
    try:
        subprocess.run(cmd, check=True, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        log(f"❌ Command failed: {e.cmd}")
        if e.stderr:
            log(f"Error output: {e.stderr.decode()}")
        return False


def deploy_to_node(node, deployment_location, user, registry_url):
    """Deploy Docker image to a single node via registry pull."""
    device_name = node["name"]
    ip = node["physical_ip"]
    host = f"{user}@{ip}"
    target_dir = f"{deployment_location}/WAFL-Testbed"

    log(f"🔗 Connecting to {device_name} ({ip})", device=device_name)

    # Create remote directories
    setup_cmd = f"mkdir -p {target_dir}/wafl/dataset"
    if not run_command(["ssh"] + SSH_OPTS + MUX_OPTS + [host, setup_cmd]):
        log(f"❌ Failed to setup directories on {device_name}", device=device_name)
        return False

    # Step 1: Sync wafl source code (excluding dataset)
    log(f"📤 Syncing source files to {device_name}...", device=device_name)
    rsync_wafl_cmd = [
        "rsync",
        "-a",
        "--delete",
        "--inplace",
        "--partial",
        "--compress",
        "--compress-level=6",
        "-e",
        "ssh " + " ".join(SSH_OPTS + MUX_OPTS),
        str(PROJECT_ROOT / "wafl" / "src"),
        str(PROJECT_ROOT / "wafl" / "__init__.py"),
        f"{host}:{target_dir}/wafl/",
    ]
    if not run_command(rsync_wafl_cmd):
        log(f"❌ Failed to sync wafl source to {device_name}", device=device_name)
        return False

    # Step 2: Sync data, utils, ctrl to target directory
    other_items = ["data", "utils", "ctrl"]
    other_sources = [str(PROJECT_ROOT / item) for item in other_items if (PROJECT_ROOT / item).exists()]
    rsync_other_cmd = [
        "rsync",
        "-a",
        "--delete",
        "--inplace",
        "--partial",
        "--compress",
        "--compress-level=6",
        "-e",
        "ssh " + " ".join(SSH_OPTS + MUX_OPTS),
        *other_sources,
        f"{host}:{target_dir}/",
    ]
    if not run_command(rsync_other_cmd):
        log(f"❌ Failed to sync data/utils/ctrl to {device_name}", device=device_name)
        return False

    # Step 3: Sync node-specific dataset only (node's number + common)
    log(f"📊 Syncing dataset for node {device_name}...", device=device_name)
    dataset_dir = PROJECT_ROOT / "wafl" / "dataset"
    node_dataset = dataset_dir / str(device_name)
    common_dataset = dataset_dir / "common"

    dataset_sources = []
    if node_dataset.exists():
        dataset_sources.append(str(node_dataset) + "/")
    if common_dataset.exists():
        dataset_sources.append(str(common_dataset) + "/")

    if dataset_sources:
        rsync_dataset_cmd = [
            "rsync",
            "-a",
            "--delete",
            "--inplace",
            "--partial",
            "--compress",
            "--compress-level=6",
            "-e",
            "ssh " + " ".join(SSH_OPTS + MUX_OPTS),
            *dataset_sources,
            f"{host}:{target_dir}/wafl/dataset/",
        ]
        if not run_command(rsync_dataset_cmd):
            log(f"❌ Failed to sync dataset to {device_name}", device=device_name)
            return False

    # Step 4: Pull Docker image from registry (layer-based differential transfer)
    log(f"📦🐳 Pulling Docker image on {device_name}...", device=device_name)
    pull_cmd = f"docker pull {registry_url}/wafl-node:latest && docker tag {registry_url}/wafl-node:latest wafl-node:latest"
    if not run_command(["ssh"] + SSH_OPTS + MUX_OPTS + [host, pull_cmd]):
        log(f"❌ Failed to pull Docker image on {device_name}", device=device_name)
        return False

    # Step 5: Prune old Docker images
    prune_cmd = "docker image prune -f"
    run_command(["ssh"] + SSH_OPTS + MUX_OPTS + [host, prune_cmd])

    log(f"✅ Successfully deployed to {device_name}", device=device_name)
    return True


def main():
    setup_logging()
    config = load_config()

    deployment_location = config.get("deployment_location", "/home/denjo")
    user = config.get("user", "denjo")

    nodes = config.get("nodes", [])
    if not nodes:
        log("❌ No nodes defined in configuration")
        sys.exit(1)

    log(f"🚀 Starting deployment to {len(nodes)} nodes (registry pull mode)")
    log(f"📦 Registry: {REGISTRY_URL}")

    # Deploy to all nodes in parallel
    threads = []
    results = []
    lock = threading.Lock()

    def thread_target(node):
        success = deploy_to_node(node, deployment_location, user, REGISTRY_URL)
        with lock:
            results.append(success)

    for node in nodes:
        t = threading.Thread(target=thread_target, args=(node,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    success_count = sum(results)
    log(f"📈 Deployment Summary: {success_count}/{len(nodes)} successful")

    if success_count != len(nodes):
        sys.exit(1)

    log("🎉 Deployment completed successfully!")


if __name__ == "__main__":
    main()

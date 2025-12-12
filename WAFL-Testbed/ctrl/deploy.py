import hashlib
import json
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


def get_build_context_hash():
    """Generate hash of files that affect Docker build."""
    h = hashlib.sha256()
    # Hash files that affect Docker build
    for path in ["Dockerfile", "pyproject.toml", "uv.lock"]:
        full_path = PROJECT_ROOT / path
        if full_path.exists():
            h.update(full_path.read_bytes())

    # Hash wafl directory (all Python files and config files)
    wafl_dir = PROJECT_ROOT / "wafl"
    if wafl_dir.exists():
        for file_path in sorted(wafl_dir.rglob("*")):
            if file_path.is_file() and not file_path.name.startswith("."):
                try:
                    h.update(file_path.read_bytes())
                except (OSError, IOError):
                    pass

    # Hash data directory (all files that will be copied to container)
    data_dir = PROJECT_ROOT / "data"
    if data_dir.exists():
        for file_path in sorted(data_dir.rglob("*")):
            if file_path.is_file() and not file_path.name.startswith("."):
                try:
                    h.update(file_path.read_bytes())
                except (OSError, IOError):
                    pass

    return h.hexdigest()[:16]


def deploy_to_node(node, deployment_location, user):
    """Deploy and build Docker image on a single node."""
    device_name = node["name"]
    ip = node["physical_ip"]
    host = f"{user}@{ip}"
    target_dir = f"{deployment_location}/WAFL-Testbed"

    log(f"🔗 Connecting to {device_name} ({ip})", device=device_name)

    # Create remote directories
    setup_cmd = f"mkdir -p {target_dir}"
    if not run_command(["ssh"] + SSH_OPTS + MUX_OPTS + [host, setup_cmd]):
        log(f"❌ Failed to setup directories on {device_name}", device=device_name)
        return False

    # Sync project files (Build Context) - OPTIMIZED: batch rsync with better compression
    log(f"📤 Syncing project files to {device_name}...", device=device_name)
    sync_items = [
        "Dockerfile",
        ".dockerignore",
        "pyproject.toml",
        "uv.lock",
        "wafl",
        "data",
        "utils",
    ]

    # Collect existing source paths
    sources = [str(PROJECT_ROOT / item) for item in sync_items if (PROJECT_ROOT / item).exists()]

    # Single optimized rsync command with all files
    rsync_cmd = [
        "rsync",
        "-a",  # Archive mode (preserves permissions, timestamps, etc.)
        "--compress",  # Enable compression
        "--compress-level=6",  # Good balance between speed and compression
        "-e",
        "ssh " + " ".join(SSH_OPTS + MUX_OPTS),
        *sources,
        f"{host}:{target_dir}/",
    ]
    if not run_command(rsync_cmd):
        log(f"❌ Failed to sync files to {device_name}", device=device_name)
        return False

    # Check if Docker build is needed by comparing hashes
    local_hash = get_build_context_hash()
    hash_file = f"{target_dir}/.build_hash"

    check_hash_cmd = f"cat {hash_file} 2>/dev/null || echo ''"
    try:
        result = subprocess.run(
            ["ssh"] + SSH_OPTS + MUX_OPTS + [host, check_hash_cmd],
            capture_output=True,
            text=True,
            check=True,
        )
        remote_hash = result.stdout.strip()
    except subprocess.CalledProcessError:
        remote_hash = ""

    # Check if Docker image exists
    check_image_cmd = "docker images -q wafl-node:latest 2>/dev/null | head -1"
    try:
        result = subprocess.run(
            ["ssh"] + SSH_OPTS + MUX_OPTS + [host, check_image_cmd],
            capture_output=True,
            text=True,
            check=True,
        )
        image_exists = bool(result.stdout.strip())
    except subprocess.CalledProcessError:
        image_exists = False

    # Skip build if hash matches AND image exists
    if local_hash == remote_hash and image_exists:
        log(
            f"⏭️  Docker image unchanged, skipping build on {device_name}",
            device=device_name,
        )
        log(f"✅ Successfully deployed to {device_name}", device=device_name)
        return True

    # Build Docker image on remote node with persistent cache
    log(f"🏗️  Building Docker image on {device_name}...", device=device_name)

    # Create persistent cache directory on host for BuildKit cache
    cache_dir = f"/home/{user}/.cache/docker-buildx"
    mkdir_cache_cmd = f"mkdir -p {cache_dir}"
    if not run_command(["ssh"] + SSH_OPTS + MUX_OPTS + [host, mkdir_cache_cmd]):
        log(
            f"⚠️  Warning: Failed to create cache directory on {device_name}",
            device=device_name,
        )

    # Setup buildx builder with docker-container driver for advanced caching
    # Optimized: only create if not exists, then use
    setup_buildx_cmd = """
if ! docker buildx inspect wafl-builder > /dev/null 2>&1; then
    docker buildx create --name wafl-builder --driver docker-container
fi
docker buildx use wafl-builder
"""
    run_command(["ssh"] + SSH_OPTS + MUX_OPTS + [host, setup_buildx_cmd])

    # Build with Buildx and persistent local cache
    # --cache-from: Load cache from host directory (speeds up subsequent builds)
    # --cache-to: Save cache to host directory (persists for future builds)
    # mode=max: Cache all layers (not just final image layers)
    build_cmd = f"cd {target_dir} && docker buildx build --builder wafl-builder --cache-from type=local,src={cache_dir} --cache-to type=local,dest={cache_dir},mode=max --load -t wafl-node:latest ."
    if not run_command(["ssh"] + SSH_OPTS + MUX_OPTS + [host, build_cmd]):
        log(f"❌ Failed to build Docker image on {device_name}", device=device_name)
        return False

    # Save build hash for future comparison
    save_hash_cmd = f"echo '{local_hash}' > {hash_file}"
    run_command(["ssh"] + SSH_OPTS + MUX_OPTS + [host, save_hash_cmd])

    # Prune Docker images
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

    log(f"🚀 Starting deployment to {len(nodes)} nodes (parallel build on each node)")

    # Deploy and build on all nodes in parallel
    threads = []
    results = []
    lock = threading.Lock()

    def thread_target(node):
        success = deploy_to_node(node, deployment_location, user)
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

"""
Apply dynamic network conditions using tc (Traffic Control) with HTB + Filter.

This script applies per-peer network limitations using hierarchical token bucket (HTB)
queuing discipline with filtering based on destination IP addresses.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def get_veth_interface(container_name: str) -> str:
    """
    Get the host veth interface for a Docker container.

    Args:
        container_name: Docker container name or ID

    Returns:
        veth interface name (e.g., "veth1a2b3c4")
    """
    try:
        # Get container PID
        cmd = ["docker", "inspect", "-f", "{{.State.Pid}}", container_name]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        pid = result.stdout.strip()

        if not pid or pid == "0":
            raise RuntimeError(f"Container {container_name} not running")

        # Get iflink index from container's eth0
        cmd = ["sudo", "nsenter", "-t", pid, "-n", "cat", "/sys/class/net/eth0/iflink"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        iflink = result.stdout.strip()

        # Find corresponding veth interface on host
        cmd = ["ip", "link"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        for line in result.stdout.split("\n"):
            if line.startswith(f"{iflink}:"):
                # Extract interface name
                parts = line.split(":")
                if len(parts) >= 2:
                    veth_name = parts[1].strip().split("@")[0]
                    return veth_name

        raise RuntimeError(f"Could not find veth interface for container {container_name}")

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to get veth interface: {e.stderr}")


def clear_tc_rules(interface: str):
    """Clear existing tc rules on interface."""
    try:
        subprocess.run(
            ["sudo", "tc", "qdisc", "del", "dev", interface, "root"],
            capture_output=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        # No existing rules, ignore
        pass


def apply_htb_filter_tc(
    interface: str,
    peers: Dict[str, dict],
    rank_definitions: List[dict],
):
    """
    Apply HTB + Filter tc configuration for per-peer limitations.

    Args:
        interface: Network interface name
        peers: Dict mapping peer IP to their rank and parameters
        rank_definitions: List of rank definitions from path loss model
    """
    print(f"  📡 Applying tc rules to {interface}...")

    # Step 1: Create root HTB qdisc
    cmd = ["sudo", "tc", "qdisc", "add", "dev", interface, "root", "handle", "1:", "htb"]
    subprocess.run(cmd, check=True, capture_output=True)
    print("    ✓ Created root HTB qdisc")

    # Step 2: Create classes for each rank
    rank_to_classid = {}
    for i, rank in enumerate(rank_definitions, start=1):
        classid = f"1:{i}"
        rank_to_classid[rank["name"]] = (classid, i * 10)

        cmd = [
            "sudo",
            "tc",
            "class",
            "add",
            "dev",
            interface,
            "parent",
            "1:",
            "classid",
            classid,
            "htb",
            "rate",
            rank["rate"],
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"    ✓ Created class {classid} for rank '{rank['name']}' (rate={rank['rate']})")

    # Step 3: Add netem qdisc to each class
    for rank in rank_definitions:
        classid, handle = rank_to_classid[rank["name"]]

        cmd = [
            "sudo",
            "tc",
            "qdisc",
            "add",
            "dev",
            interface,
            "parent",
            classid,
            "handle",
            f"{handle}:",
            "netem",
            "delay",
            rank["delay"],
            "loss",
            rank["loss"],
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"    ✓ Applied netem to {classid} (delay={rank['delay']}, loss={rank['loss']})")

    # Step 4: Add filters to route traffic to appropriate class based on destination IP
    filter_count = 0
    for peer_ip, peer_config in peers.items():
        rank_name = peer_config["rank"]

        if rank_name not in rank_to_classid:
            print(f"    ⚠️  Warning: Unknown rank '{rank_name}' for peer {peer_ip}, skipping")
            continue

        classid, _ = rank_to_classid[rank_name]

        cmd = [
            "sudo",
            "tc",
            "filter",
            "add",
            "dev",
            interface,
            "protocol",
            "ip",
            "parent",
            "1:0",
            "prio",
            "1",
            "u32",
            "match",
            "ip",
            "dst",
            peer_ip,
            "flowid",
            classid,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        filter_count += 1

    print(f"    ✓ Applied {filter_count} filters for peer-specific routing")


def apply_dynamic_tc(
    container_name: str,
    epoch: int,
    node_id: str,
    conditions_file: str,
    rank_definitions: List[dict],
    execution_config_file: str = None,
):
    """
    Apply dynamic tc configuration for a specific epoch and node.

    Args:
        container_name: Docker container name
        epoch: Current epoch number
        node_id: Node ID (e.g., "0", "1")
        conditions_file: Path to network conditions JSON file (array format)
        rank_definitions: List of rank definitions from path loss model
        execution_config_file: Path to execution_config.json for IP mapping
    """
    print("🔧 Applying dynamic network conditions:")
    print(f"   Container: {container_name}")
    print(f"   Epoch: {epoch}")
    print(f"   Node: {node_id}")

    # Load network conditions (array format)
    with open(conditions_file) as f:
        conditions = json.load(f)

    # Check if epoch is within bounds
    if not isinstance(conditions, list):
        print("❌ Error: network_conditions must be an array", file=sys.stderr)
        sys.exit(1)

    if epoch >= len(conditions):
        print(f"⚠️  Warning: No conditions for epoch {epoch} (only {len(conditions)} epochs available), skipping")
        return

    # Get conditions for this epoch
    epoch_conditions = conditions[epoch]

    # Convert node_id to integer for lookup
    node_key = int(node_id)

    if node_key not in epoch_conditions:
        print(f"⚠️  Warning: No conditions for node {node_id} in epoch {epoch}, skipping")
        return

    # Get peer list for this node (array format)
    peers_list = epoch_conditions[node_key]

    if not isinstance(peers_list, list):
        print(f"❌ Error: Node conditions must be an array, got {type(peers_list)}", file=sys.stderr)
        sys.exit(1)

    if not peers_list:
        print(f"ℹ️  No peers in range for node {node_id} at epoch {epoch}")
        # Still apply empty tc rules to clear previous settings
        interface = get_veth_interface(container_name)
        clear_tc_rules(interface)
        print("  ✓ Cleared tc rules (no peers in range)")
        return

    print(f"  Found {len(peers_list)} peers in range")

    # Load execution_config for node name to IP mapping
    node_to_ip = {}
    if execution_config_file and Path(execution_config_file).exists():
        with open(execution_config_file) as f:
            exec_config = json.load(f)

        # Build mapping: node name -> Docker IP
        for i, node in enumerate(exec_config.get("nodes", [])):
            node_name = str(node["name"])
            docker_ip = f"172.18.0.{i + 2}"
            node_to_ip[int(node_name)] = docker_ip

        print(f"  Loaded IP mapping for {len(node_to_ip)} nodes")
    else:
        # Fallback: assume sequential IPs
        print("  ⚠️  No execution_config provided, using sequential IP assignment")
        for peer_info in peers_list:
            peer_id = peer_info.get("peer")
            if peer_id is not None:
                # Assume sequential: node 15 -> 172.18.0.2, node 16 -> 172.18.0.3, etc.
                # This is a fallback and may not be accurate
                node_to_ip[peer_id] = f"172.18.0.{peer_id + 2}"

    # Convert peer list to IP-based dict for tc application
    peers_for_tc = {}
    for peer_info in peers_list:
        peer_id = peer_info.get("peer")
        if peer_id is None:
            continue

        peer_ip = node_to_ip.get(peer_id)
        if not peer_ip:
            print(f"  ⚠️  Warning: No IP mapping for peer {peer_id}, skipping")
            continue

        peers_for_tc[peer_ip] = {
            "rank": peer_info["rank"],
            "rate": peer_info.get("rate", "100mbit"),
            "delay": peer_info.get("delay", "10ms"),
            "loss": peer_info.get("loss", "0%"),
        }

    if not peers_for_tc:
        print("ℹ️  No valid peer IPs found for tc configuration")
        interface = get_veth_interface(container_name)
        clear_tc_rules(interface)
        print("  ✓ Cleared tc rules (no valid peers)")
        return

    # Get veth interface
    interface = get_veth_interface(container_name)
    print(f"  Identified interface: {interface}")

    # Clear existing tc rules
    clear_tc_rules(interface)

    # Apply new HTB + Filter configuration
    apply_htb_filter_tc(interface, peers_for_tc, rank_definitions)

    print("✅ Dynamic network conditions applied successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Apply dynamic network conditions using tc with HTB + Filter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python apply_dynamic_tc.py \\
    --container wafl-node-0 \\
    --epoch 0 \\
    --node-id 0 \\
    --conditions data/network_conditions_mobility.json \\
    --pathloss config/sumo/path_loss_model.json

This applies per-peer network limitations based on the distance-based
path loss model using HTB (Hierarchical Token Bucket) with IP-based filtering.
        """,
    )
    parser.add_argument(
        "--container",
        type=str,
        required=True,
        help="Docker container name or ID",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        required=True,
        help="Current epoch number",
    )
    parser.add_argument(
        "--node-id",
        type=str,
        required=True,
        help="Node ID (e.g., '0', '1')",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        required=True,
        help="Path to network conditions JSON file",
    )
    parser.add_argument(
        "--pathloss",
        type=str,
        required=True,
        help="Path to path loss model JSON file",
    )
    parser.add_argument(
        "--execution-config",
        type=str,
        default=None,
        help="Path to execution_config.json for IP mapping (optional)",
    )

    args = parser.parse_args()

    # Validate input files
    if not Path(args.conditions).exists():
        print(f"❌ Error: Conditions file not found: {args.conditions}", file=sys.stderr)
        sys.exit(1)

    if not Path(args.pathloss).exists():
        print(f"❌ Error: Path loss model file not found: {args.pathloss}", file=sys.stderr)
        sys.exit(1)

    # Load path loss model for rank definitions
    with open(args.pathloss) as f:
        pathloss_model = json.load(f)

    rank_definitions = pathloss_model.get("ranks", [])
    if not rank_definitions:
        print("❌ Error: No rank definitions in path loss model", file=sys.stderr)
        sys.exit(1)

    try:
        apply_dynamic_tc(
            args.container,
            args.epoch,
            args.node_id,
            args.conditions,
            rank_definitions,
            args.execution_config,
        )
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

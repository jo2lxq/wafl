import json
import subprocess
from collections import defaultdict


def load_config():
    with open("ctrl/execution_config.json", "r") as f:
        return json.load(f)


def run_ssh_command(ip, command, user):
    ssh_cmd = f"ssh -o StrictHostKeyChecking=no {user}@{ip} '{command}'"
    print(f"[{ip}] Executing: {command}")
    subprocess.run(ssh_cmd, shell=True, check=True)


def main():
    config = load_config()
    user = config.get("user", "denjo")
    nodes = config.get("nodes", [])

    host_nodes = defaultdict(list)
    for node in nodes:
        host_nodes[node["physical_ip"]].append(node)

    for ip, nodes_on_host in host_nodes.items():
        print(f"Cleaning up nodes on {ip}...")
        container_names = [f"wafl-node-{n['id']}" for n in nodes_on_host]
        cleanup_cmd = f"docker rm -f {' '.join(container_names)} || true"
        run_ssh_command(ip, cleanup_cmd, user)

        # Also reset TC rules?
        # reset_network_rules.sh takes container name.
        # But if container is gone, veth is gone.
        # So explicit reset might fail or be unnecessary.
        # But good practice to try.
        # However, we need container PID to find veth. If container is gone, we can't find veth.
        # So we should reset BEFORE removing container?
        # Or just rely on container removal cleaning up veth.
        # Docker usually cleans up veth.
        pass


if __name__ == "__main__":
    main()

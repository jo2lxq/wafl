import json
import os
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

    deploy_loc = config.get("deployment_location", "/home/denjo")
    user = config.get("user", "denjo")
    nodes = config.get("nodes", [])

    # Project name is assumed to be WAFL-Testbed based on deploy.py
    proj_name = "WAFL-Testbed"
    target_path = os.path.join(deploy_loc, proj_name)

    host_nodes = defaultdict(list)
    for node in nodes:
        host_nodes[node["physical_ip"]].append(node)

    for ip, nodes_on_host in host_nodes.items():
        print(f"Starting nodes on {ip}...")

        # Cleanup existing containers
        container_names = [f"wafl-node-{n['id']}" for n in nodes_on_host]
        cleanup_cmd = f"docker rm -f {' '.join(container_names)} || true"
        run_ssh_command(ip, cleanup_cmd, user)

        for node in nodes_on_host:
            node_id = node["id"]
            container_name = f"wafl-node-{node_id}"
            host_ctrl = node["host_port_ctrl"]
            cont_ctrl = node["container_port_ctrl"]
            host_p2p = node["host_port_p2p"]
            cont_p2p = 10002

            # Mount parameters.json for all nodes (simplifies logic, though only ControlServer might need it)
            # Assuming parameters.json is in ctrl/ directory which is deployed to remote
            mounts = (
                f"-v {target_path}/wafl/dataset:/app/dataset "
                f"-v {target_path}/wafl/config/config_{node_id}.json:/app/config/config.json "
                f"-v {target_path}/results:/app/results "
                f"-v {target_path}/wafl/src:/app/wafl/src "
                f"-v {target_path}/ctrl/parameters.json:/app/ctrl/parameters.json "
            )

            ports = f"-p {host_ctrl}:{cont_ctrl} -p {host_p2p}:{cont_p2p} "

            env_vars = "-e LOG_LEVEL=INFO "
            image = "wafl-node:v1.0"

            # Resource limits
            cpu_limit = node.get("cpu_limit")
            resource_flags = ""
            if cpu_limit:
                resource_flags += f"--cpus={cpu_limit} "

            run_cmd = f"docker run -d --name {container_name} {ports} {mounts} {env_vars} {resource_flags} {image}"
            run_ssh_command(ip, run_cmd, user)

            # Apply TC
            # Read global network conditions from parameters.json
            try:
                with open("ctrl/parameters.json", "r") as f:
                    params = json.load(f)
                    net_cond = params.get("network_condition", {})
                    delay = net_cond.get("delay", "50ms")
                    loss = net_cond.get("loss", "0%")
                    rate = net_cond.get("rate", "100mbit")
            except Exception as e:
                print(f"⚠️ Failed to read parameters.json for network rules: {e}. Using defaults.")
                delay = "50ms"
                loss = "0%"
                rate = "100mbit"

            apply_tc_cmd = f"sudo {target_path}/ctrl/apply_network_rules.sh {container_name} {delay} {loss} {rate}"
            run_ssh_command(ip, apply_tc_cmd, user)


if __name__ == "__main__":
    main()

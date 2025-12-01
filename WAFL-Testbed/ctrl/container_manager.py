import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import paramiko


class ContainerManager:
    """
    Manages Docker containers and network conditions for WAFL-Testbed.
    Shared logic between main.py and verify.py.
    """

    def __init__(
        self,
        user: str,
        deployment_location: str,
        project_name: str = "WAFL-Testbed",
        verbose: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        self.user = user
        self.deployment_location = deployment_location
        self.project_name = project_name
        self.verbose = verbose
        self.logger = logger
        self.private_key_path = os.path.expanduser("~/.ssh/id_ed25519")

        if not os.path.exists(self.private_key_path):
            raise FileNotFoundError(f"SSH private key not found: {self.private_key_path}")

        self.key = paramiko.Ed25519Key.from_private_key_file(self.private_key_path)

    def _log_info(self, msg: str):
        if self.logger:
            self.logger.info(msg)
        elif self.verbose:
            print(f"  \033[96mℹ\033[0m {msg}")

    def _log_error(self, msg: str):
        if self.logger:
            self.logger.error(msg)
        else:
            print(f"  \033[91m✗\033[0m {msg}")

    def _log_debug(self, msg: str):
        if self.logger:
            self.logger.debug(msg)
        elif self.verbose:
            print(f"  \033[90m[DEBUG]\033[0m {msg}")

    def connect_ssh(self, ip: str) -> Optional[paramiko.SSHClient]:
        """Establish SSH connection"""
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(ip, port=22, username=self.user, pkey=self.key, timeout=10)
            return ssh
        except Exception as e:
            self._log_error(f"SSH connection to {ip} failed: {e}")
            return None

    def exec_command(
        self,
        ssh: paramiko.SSHClient,
        command: str,
        timeout: Optional[float] = None,
        get_pty: bool = False,
    ) -> Tuple[Any, Any, Any]:
        """Execute SSH command with logging"""
        if self.verbose or (self.logger and self.logger.level <= logging.DEBUG):
            # Use a simplified log format for commands to avoid clutter
            cmd_msg = f"[CMD] {command}"
            if self.logger:
                self.logger.debug(cmd_msg)
            elif self.verbose:
                print(f"  \033[96mℹ\033[0m {cmd_msg}")

        return ssh.exec_command(command, timeout=timeout, get_pty=get_pty)

    def stop_container(self, ip: str, container_name: str) -> bool:
        """Stop and remove a container"""
        ssh = self.connect_ssh(ip)
        if not ssh:
            return False

        try:
            self._log_debug(f"Stopping container {container_name} on {ip}...")
            cleanup_cmd = f"docker rm -f {container_name} 2>/dev/null || true"
            stdin, stdout, stderr = self.exec_command(ssh, cleanup_cmd)
            stdout.channel.recv_exit_status()
            ssh.close()
            return True
        except Exception as e:
            self._log_error(f"Failed to stop container {container_name}: {e}")
            if ssh:
                ssh.close()
            return False

    def start_wafl_container(
        self,
        node_info: Dict[str, Any],
        experiment_params: Dict[str, Any],
        env_vars: str = "-e LOG_LEVEL=INFO",
    ) -> bool:
        """
        Start the WAFL node container with standard configuration.

        Args:
            node_info: Node configuration dict (must contain 'name', 'physical_ip', etc.)
            experiment_params: Experiment parameters (for network conditions)
            env_vars: Environment variables string
        """
        node_name = str(node_info.get("name", "unknown"))
        ip = node_info.get("physical_ip", "")
        container_name = f"wafl-node-{node_name}"
        cpu_limit = node_info.get("cpu_limit")

        # Ports
        host_ctrl = node_info.get("host_port_ctrl", 10001)  # Default from verify.py logic
        # main.py uses self.ctrl_port which comes from config, usually 10001 + index?
        # Let's standardize. main.py passes ctrl_port in __init__.
        # verify.py assumes 10001 if not in node dict.
        # We'll use values from node_info, caller must ensure they are correct.

        cont_ctrl = node_info.get("container_port_ctrl", 10001)
        host_p2p = node_info.get("host_port_p2p", 10002)
        cont_p2p = 10002

        ports = f"-p {host_ctrl}:{cont_ctrl} -p {host_p2p}:{cont_p2p}"

        # Mounts
        target_path = f"{self.deployment_location}/{self.project_name}"
        mounts = f"-v {target_path}/dataset:/app/dataset -v {target_path}/config/config.json:/app/config.json -v {target_path}/config/contact_pattern.json:/app/contact_pattern.json -v {target_path}/results:/app/results -v {target_path}/wafl/src:/app/wafl/src -v {target_path}/ctrl/parameters.json:/app/ctrl/parameters.json"

        # Resource limits
        resource_flags = f"--cpus={cpu_limit}" if cpu_limit else ""
        image = "wafl-node:latest"

        ssh = self.connect_ssh(ip)
        if not ssh:
            return False

        try:
            # 1. Cleanup
            self.stop_container(ip, container_name)

            # 2. Verify image
            verify_img_cmd = "docker images wafl-node:latest -q"
            stdin, stdout, stderr = self.exec_command(ssh, verify_img_cmd)
            image_id = stdout.read().decode().strip()

            if not image_id:
                self._log_error(f"wafl-node:latest image not found on {ip}")
                ssh.close()
                return False

            # 3. Run container
            run_cmd = f"docker run -d --name {container_name} {ports} {mounts} {env_vars} {resource_flags} {image}"
            stdin, stdout, stderr = self.exec_command(ssh, run_cmd)
            exit_status = stdout.channel.recv_exit_status()

            if exit_status != 0:
                error_msg = stderr.read().decode().strip()
                self._log_error(f"Docker run failed for {container_name}: {error_msg}")
                ssh.close()
                return False

            container_id = stdout.read().decode().strip()
            self._log_debug(f"Container {container_name} started (ID: {container_id[:12]})")

            # 4. Apply network conditions
            self.apply_network_conditions(ssh, container_name, experiment_params)

            # Wait for container to be somewhat ready (verify.py does this)
            time.sleep(3)

            ssh.close()
            return True

        except Exception as e:
            self._log_error(f"Failed to start container {container_name}: {e}")
            if ssh:
                ssh.close()
            return False

    def apply_network_conditions(self, ssh: paramiko.SSHClient, container_name: str, params: Dict[str, Any]) -> bool:
        """Apply network conditions using the shell script"""
        net_cond = params.get("network_condition", {})

        # Default to enabled=True if not specified (backward compatibility, matches main.py logic)
        # But verify.py defaults to False?
        # main.py: enabled = net_cond.get("enabled", True)
        # verify.py: if not net_cond.get("enabled", False): return
        # The user request says "ensure consistency with verify.py".
        # However, main.py logic is "initialize with network conditions".
        # Let's check verify.py again.
        # verify.py line 943: if not net_cond.get("enabled", False): return ...
        # So verify.py assumes disabled by default if key is missing?
        # parameters.json usually has "enabled": true.
        # Let's use the logic: if "enabled" key exists, use it. If not, what is safe?
        # verify.py is strict. main.py is permissive.
        # I will follow the explicit config.

        enabled = net_cond.get("enabled", False)

        if not enabled:
            self._log_debug(f"Network conditions disabled for {container_name}")
            return True

        delay = net_cond.get("delay", "50ms")
        loss = net_cond.get("loss", "0%")
        rate = net_cond.get("rate", "100mbit")

        target_path = f"{self.deployment_location}/{self.project_name}"
        # Try running without sudo first, as the script contains sudo for tc commands
        # and the user might have permissions for docker but not full sudo
        tc_cmd = f"{target_path}/ctrl/apply_network_rules.sh {container_name} {delay} {loss} {rate}"

        # Use get_pty=True because sudo inside the script might require a TTY
        stdin, stdout, stderr = self.exec_command(ssh, tc_cmd, get_pty=True)
        exit_status = stdout.channel.recv_exit_status()

        output = stdout.read().decode().strip()
        error = stderr.read().decode().strip()

        if exit_status != 0:
            self._log_error(f"Failed to apply network rules to {container_name}: {error}")
            return False

        if self.verbose:
            self._log_debug(f"Network rules output: {output}")
            if error:
                self._log_debug(f"Network rules stderr: {error}")

        self._log_debug(f"Applied network rules to {container_name}: delay={delay}, loss={loss}, rate={rate}")
        return True

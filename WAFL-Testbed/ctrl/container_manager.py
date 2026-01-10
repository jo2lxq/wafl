import logging
import os
import time
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import paramiko

if TYPE_CHECKING:
    from ctrl.ssh_connection_manager import SSHConnectionManager


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
        ssh_manager: Optional["SSHConnectionManager"] = None,
    ):
        self.user = user
        self.deployment_location = deployment_location
        self.project_name = project_name
        self.verbose = verbose
        self.logger = logger
        self.ssh_manager = ssh_manager
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
        """
        Establish SSH connection.

        If an SSHConnectionManager was provided, uses it for connection pooling.
        Otherwise, creates a new connection each time.
        """
        try:
            if self.ssh_manager:
                return self.ssh_manager.get_connection(ip)

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
            # Only close if not using connection manager
            if not self.ssh_manager:
                ssh.close()
            return True
        except Exception as e:
            self._log_error(f"Failed to stop container {container_name}: {e}")
            if ssh and not self.ssh_manager:
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

        Note: CPU limits are NOT applied at container startup.
        Use apply_cpu_limit() at the start of WAFL phase to apply cpu_limit.

        Args:
            node_info: Node configuration dict (must contain 'name', 'physical_ip', etc.)
            experiment_params: Experiment parameters (for network conditions)
            env_vars: Environment variables string
        """
        node_name = str(node_info.get("name", "unknown"))
        ip = node_info.get("physical_ip", "")
        container_name = f"wafl-node-{node_name}"
        # NOTE: cpu_limit is NOT applied here - it will be applied at WAFL phase start

        # Ports
        host_ctrl = node_info.get("host_port_ctrl", 10001)  # Default from verify.py logic
        # main.py uses self.ctrl_port which comes from config, usually 10001 + index?
        # Let's standardize. main.py passes ctrl_port in __init__.
        # verify.py assumes 10001 if not in node dict.
        # We'll use values from node_info, caller must ensure they are correct.

        cont_ctrl = node_info.get("container_port_ctrl", 10001)
        host_p2p = node_info.get("host_port_p2p", 10002)
        cont_p2p = 10002

        # TCP ports for control and P2P, plus UDP for P2P (for FEC-based model sharing)
        ports = f"-p {host_ctrl}:{cont_ctrl} -p {host_p2p}:{cont_p2p} -p {host_p2p}:{cont_p2p}/udp"

        # Mounts
        target_path = f"{self.deployment_location}/{self.project_name}"
        mounts = f"-v {target_path}/wafl/dataset:/app/dataset -v {target_path}/config/config.json:/app/config.json -v {target_path}/config/contact_pattern.json:/app/contact_pattern.json -v {target_path}/results:/app/results -v {target_path}/wafl/src:/app/wafl/src -v {target_path}/ctrl/parameters.json:/app/ctrl/parameters.json"  # noqa: E501

        # No resource limits at startup - CPU limits applied at WAFL phase start
        resource_flags = ""
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
                if not self.ssh_manager:
                    ssh.close()
                return False

            # 3. Run container
            run_cmd = f"docker run -d --name {container_name} --cap-add=NET_ADMIN {ports} {mounts} {env_vars} {resource_flags} {image}"
            stdin, stdout, stderr = self.exec_command(ssh, run_cmd)
            exit_status = stdout.channel.recv_exit_status()

            if exit_status != 0:
                error_msg = stderr.read().decode().strip()
                self._log_error(f"Docker run failed for {container_name}: {error_msg}")
                if not self.ssh_manager:
                    ssh.close()
                return False

            container_id = stdout.read().decode().strip()
            self._log_debug(f"Container {container_name} started (ID: {container_id[:12]})")

            # 4. Apply network conditions
            self.apply_network_conditions(ssh, container_name, experiment_params)

            # Wait for container to be somewhat ready (verify.py does this)
            time.sleep(3)

            if not self.ssh_manager:
                ssh.close()
            return True

        except Exception as e:
            self._log_error(f"Failed to start container {container_name}: {e}")
            if ssh and not self.ssh_manager:
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

        # Network conditions are now applied dynamically by ctrl/main.py using
        # direct tc commands inside the container via SSH + docker exec.
        # Static network rules at container startup are no longer used.
        # This method is kept for compatibility but does nothing.
        self._log_debug(f"Network conditions will be applied dynamically for {container_name} (delay={net_cond.get('delay')}, loss={net_cond.get('loss')}, rate={net_cond.get('rate')})")
        return True

    def apply_cpu_limit(self, ip: str, container_name: str, cpu_limit: float) -> bool:
        """
        Apply CPU limit to a running container using docker update.

        This should be called at the start of WAFL phase to apply cpu_limit
        that was specified in execution_config.json for each node.

        Args:
            ip: IP address of the host running the container
            container_name: Name of the Docker container
            cpu_limit: CPU limit value (e.g., 0.5 for half a core, 1.0 for one core)

        Returns:
            bool: True if successful, False otherwise
        """
        if not cpu_limit:
            self._log_debug(f"No CPU limit to apply for {container_name}")
            return True

        ssh = self.connect_ssh(ip)
        if not ssh:
            return False

        try:
            # Use docker update to apply CPU limit to running container
            update_cmd = f"docker update --cpus={cpu_limit} {container_name}"
            stdin, stdout, stderr = self.exec_command(ssh, update_cmd)
            exit_status = stdout.channel.recv_exit_status()

            if exit_status != 0:
                error_msg = stderr.read().decode().strip()
                self._log_error(f"Failed to apply CPU limit to {container_name}: {error_msg}")
                if not self.ssh_manager:
                    ssh.close()
                return False

            self._log_info(f"Applied CPU limit {cpu_limit} to {container_name}")
            if not self.ssh_manager:
                ssh.close()
            return True

        except Exception as e:
            self._log_error(f"Failed to apply CPU limit to {container_name}: {e}")
            if ssh and not self.ssh_manager:
                ssh.close()
            return False

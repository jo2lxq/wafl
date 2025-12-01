import datetime
import json
import logging
import os
import sys

# Add project root to sys.path to allow importing from ctrl package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import socket
import time
from typing import Any, Dict, List, Tuple

import paramiko

from ctrl.container_manager import ContainerManager


class WaflAgent:
    """
    Represents each execution server (WAFL device) and manages communication.

    Attributes:
        agent_index (int): Device index (e.g., 0, 1)
        name (str): Device name (e.g., "100", "101")
        ip (str): IP address
        ctrl_port (int): Control TCP port number
        status (str): Current status ("UNKNOWN", "READY", "RUNNING", "DONE", "ERROR", "TERMINATED")
        pid (str): Process ID of the remote wafl/main.py script.
    """

    def __init__(
        self,
        agent_index: int,
        device_name: str,
        ip_address: str,
        ctrl_port: int,
        config: Dict[str, Any],
        experiment_parameters: Dict[str, Any],
        experiment_id: str,
        start_timestamp: float,
        timeout: int = 10,
        container_ctrl_port: int = None,
        host_p2p_port: int = None,
        container_p2p_port: int = None,
        node_config: dict = None,
    ):
        self.agent_index = agent_index
        self.name = device_name
        self.ip = ip_address
        self.ctrl_port = ctrl_port
        self.container_ctrl_port = container_ctrl_port or ctrl_port
        self.host_p2p_port = host_p2p_port or config.get("WAFL_DEVICE_P2P_PORT", 10002)
        self.container_p2p_port = container_p2p_port or self.host_p2p_port
        self.status = "UNKNOWN"
        self.logger = logging.getLogger(f"WaflAgent-{device_name}")
        self.pid = None
        self.timeout = timeout
        self.config = config
        self.experiment_id = experiment_id
        self.start_timestamp = start_timestamp
        self.node_config = node_config or {}  # Store full node configuration for container management

        # Initialize ContainerManager
        self.container_manager = ContainerManager(
            user=self.config.get("USER", "denjo"),
            deployment_location=self.config.get("DEPLOYMENT_LOCATION", "/home/denjo"),
            project_name=self.config.get("PROJECT_NAME", "WAFL-Testbed"),
            logger=self.logger,
        )

        # Deploy configurations during initialization
        self._deploy_configurations(experiment_parameters)

        # Start Docker container
        self.logger.debug(f"🐳 Container [1/4] - Initiating startup for agent {self.name}")
        self.start_remote_process(self.experiment_id, self.node_config)

    def _create_unified_config(self, experiment_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create unified configuration for this agent.

        Args:
            experiment_parameters: Experiment parameters from ControlServer

        Returns:
            Dict containing unified configuration for this agent
        """

        unified_config = {
            "agent_info": {
                "index": self.agent_index,
                "device_name": self.name,
                "ip_address": self.ip,
            },
            "experiment_info": {
                "project_name": self.config["PROJECT_NAME"],
                "experiment_name": self.config["EXPERIMENT_NAME"],
                "experiment_id": self.experiment_id,
                "start_timestamp": self.start_timestamp,
            },
            "infrastructure": {
                "device_names": self.config["WAFL_DEVICE_NAMES"],
                "device_ips": self.config["WAFL_DEVICE_IPS"],
                "ctrl_port": self.container_ctrl_port,
                "p2p_port": self.container_p2p_port,
            },
            "experiment_parameters": {
                "epochs": experiment_parameters.get("epochs"),
                "wafl_phase": experiment_parameters.get("wafl_phase", {}),
            },
            "runtime": {
                "log_level": os.environ.get("LOG_LEVEL", "INFO"),
            },
        }

        # Add mobility-aware configuration if provided
        if "mobility_aware" in experiment_parameters:
            unified_config["mobility_aware"] = experiment_parameters["mobility_aware"]

        self.logger.debug(f"🔧 Created unified configuration for agent {self.name}")
        return unified_config

    def _deploy_configurations(self, experiment_parameters: Dict[str, Any]) -> bool:
        """
        Deploy all configuration files (contact pattern and agent config) to this agent via SSH.

        Args:
            experiment_parameters: Experiment parameters containing configuration data

        Returns:
            bool: True if all deployments successful, False otherwise
        """
        self.logger.debug(f"📦 Deployment [1/3] - Starting configuration deployment to agent {self.name}")

        try:
            ssh_port = 22
            username = self.config["USER"]
            private_key_path = os.path.expanduser("~/.ssh/id_ed25519")

            if not os.path.exists(private_key_path):
                raise FileNotFoundError(f"🔑 SSH private key not found at {private_key_path}")

            key = paramiko.Ed25519Key.from_private_key_file(private_key_path)
            target_path = os.path.join(self.config["DEPLOYMENT_LOCATION"], self.config["PROJECT_NAME"])
            config_dir = os.path.join(target_path, "config")

            self.logger.debug(f"🔗 Connecting to {username}@{self.ip} for configuration deployment")

            with paramiko.SSHClient() as ssh:
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(self.ip, port=ssh_port, username=username, pkey=key, timeout=10)

                # Ensure config directory exists
                command_mkdir = f"mkdir -p {config_dir}"
                stdin, stdout, stderr = ssh.exec_command(command_mkdir)
                exit_status = stdout.channel.recv_exit_status()

                if exit_status != 0:
                    error_msg = stderr.read().decode().strip()
                    raise RuntimeError(f"Failed to create config directory: {error_msg}")

                # Prepare configuration files for deployment
                files_to_deploy = []

                # 1. Agent configuration (always required)
                unified_config = self._create_unified_config(experiment_parameters)
                config_json = json.dumps(unified_config, indent=2, ensure_ascii=False)
                files_to_deploy.append(
                    {
                        "content": config_json,
                        "filename": "config.json",
                        "description": "agent configuration",
                    }
                )

                # 2. Contact pattern
                contact_pattern = experiment_parameters.get("contact_pattern")
                if contact_pattern is None:
                    raise ValueError("contact_pattern cannot be None")
                contact_pattern_path = os.path.join("data", "contact_pattern", contact_pattern)

                if not os.path.exists(contact_pattern_path):
                    raise FileNotFoundError(f"Contact pattern file not found: {contact_pattern_path}")

                with open(contact_pattern_path, "r", encoding="utf-8") as f:
                    contact_pattern_data = json.load(f)

                contact_pattern_json = json.dumps(contact_pattern_data, indent=2, ensure_ascii=False)
                files_to_deploy.append(
                    {
                        "content": contact_pattern_json,
                        "filename": "contact_pattern.json",
                        "description": f"contact pattern '{contact_pattern}'",
                    }
                )

                # Deploy all files via SFTP
                with ssh.open_sftp() as sftp:
                    import io

                    # Create config directory
                    try:
                        sftp.stat(config_dir)
                    except FileNotFoundError:
                        sftp.mkdir(config_dir)

                    # Create dataset directories
                    dataset_dir = os.path.join(target_path, "dataset")
                    train_dir = os.path.join(dataset_dir, "train")
                    test_dir = os.path.join(dataset_dir, "test")

                    for directory in [dataset_dir, train_dir, test_dir]:
                        try:
                            sftp.stat(directory)
                            self.logger.debug(f"📁 Directory exists: {directory}")
                        except FileNotFoundError:
                            sftp.mkdir(directory)
                            self.logger.debug(f"📁 Created directory: {directory}")

                    deployed_files = []

                    # Deploy config files
                    for file_info in files_to_deploy:
                        try:
                            file_path = os.path.join(config_dir, file_info["filename"])
                            file_obj = io.BytesIO(file_info["content"].encode("utf-8"))

                            self.logger.debug(f"📤 Deploying {file_info['filename']} to {file_path} ({len(file_info['content'])} bytes)")
                            sftp.putfo(file_obj, file_path)
                            sftp.chmod(file_path, 0o644)

                            deployed_files.append(file_info["filename"])
                            self.logger.debug(f"Deployment [1/3] - {file_info['description'].capitalize()} deployed")

                        except Exception as e:
                            self.logger.error(f"💥 Failed to deploy {file_info['description']}: {type(e).__name__}: {str(e)}")
                            import traceback

                            self.logger.error(f"Traceback: {traceback.format_exc()}")
                            raise RuntimeError(f"Failed to deploy agent configuration: {e}")

                    # Deploy train dataset (node-specific)
                    local_train_path = os.path.join("wafl", "dataset", str(self.agent_index), "train", "train.pkl")
                    remote_train_path = os.path.join(train_dir, "train.pkl")

                    if not os.path.exists(local_train_path):
                        raise FileNotFoundError(f"Train dataset not found: {local_train_path}")

                    train_size = os.path.getsize(local_train_path)
                    train_size_mb = train_size / (1024 * 1024)
                    self.logger.debug(f"📊 Deployment [2/3] - Uploading train dataset ({train_size_mb:.1f} MB)")
                    sftp.put(local_train_path, remote_train_path)
                    sftp.chmod(remote_train_path, 0o644)
                    deployed_files.append("train.pkl")
                    self.logger.debug("Deployment [2/3] - Train dataset deployed successfully")

                    # Deploy test dataset (common)
                    local_test_path = os.path.join("wafl", "dataset", "common", "test", "test.pkl")
                    remote_test_path = os.path.join(test_dir, "test.pkl")

                    if not os.path.exists(local_test_path):
                        raise FileNotFoundError(f"Test dataset not found: {local_test_path}")

                    test_size = os.path.getsize(local_test_path)
                    test_size_mb = test_size / (1024 * 1024)
                    self.logger.debug(f"📊 Deployment [2/3] - Uploading test dataset ({test_size_mb:.1f} MB)")
                    sftp.put(local_test_path, remote_test_path)
                    sftp.chmod(remote_test_path, 0o644)
                    deployed_files.append("test.pkl")
                    self.logger.debug("Deployment [2/3] - Test dataset deployed successfully")

                # Verify config files only (datasets already confirmed via SFTP)
                verification_success = True
                config_files = ["config.json", "contact_pattern.json"]
                for filename in config_files:
                    if filename in deployed_files:
                        file_path = os.path.join(config_dir, filename)
                        stdin, stdout, stderr = ssh.exec_command(f"test -f {file_path} && echo '{filename}: OK'")
                        verification = stdout.read().decode().strip()
                        exit_status = stdout.channel.recv_exit_status()

                        if exit_status != 0 or "OK" not in verification:
                            self.logger.error(f"❌ File verification failed for {filename}")
                            verification_success = False
                        else:
                            self.logger.debug(f"📊 {verification}")

                if not verification_success:
                    raise RuntimeError("Configuration file verification failed")

                total_files = len(deployed_files)
                self.logger.info(f"✅ Deployment [3/3] - Complete ({total_files} files deployed to agent {self.name})")
                return True

        except FileNotFoundError as e:
            self.logger.error(f"📁 Configuration file error for agent {self.name}: {e}")
            return False
        except json.JSONDecodeError as e:
            self.logger.error(f"📄 JSON parse error for agent {self.name}: {e}")
            return False
        except paramiko.AuthenticationException as e:
            self.logger.error(f"🔒 SSH authentication failed for agent {self.name}: {e}")
            return False
        except paramiko.SSHException as e:
            self.logger.error(f"🌐 SSH connection error to agent {self.name}: {e}")
            return False
        except Exception as e:
            self.logger.error(
                f"💥 Configuration deployment failed for agent {self.name}: {e}",
                exc_info=True,
            )
            return False

    def _deploy_agent_config(self, experiment_parameters: Dict[str, Any]) -> bool:
        """
        Deploy unified configuration JSON file to this agent via SSH.

        Args:
            experiment_parameters: Experiment parameters from ControlServer

        Returns:
            bool: True if deployment successful, False otherwise
        """
        self.logger.info(f"📋 Deploying configuration to agent {self.name}")

        try:
            unified_config = self._create_unified_config(experiment_parameters)
            config_json = json.dumps(unified_config, indent=2, ensure_ascii=False)

            ssh_port = 22
            username = self.config["USER"]
            private_key_path = os.path.expanduser("~/.ssh/id_ed25519")

            if not os.path.exists(private_key_path):
                raise FileNotFoundError(f"🔑 SSH private key not found at {private_key_path}")

            key = paramiko.Ed25519Key.from_private_key_file(private_key_path)
            target_path = os.path.join(self.config["DEPLOYMENT_LOCATION"], self.config["PROJECT_NAME"])
            config_dir = os.path.join(target_path, "config")
            config_file_path = os.path.join(config_dir, f"config_{self.agent_index}.json")

            self.logger.debug(f"🔗 Deploying config to {username}@{self.ip}:{config_file_path}")

            with paramiko.SSHClient() as ssh:
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(self.ip, port=ssh_port, username=username, pkey=key, timeout=10)

                # Create config directory
                command_mkdir = f"mkdir -p {config_dir}"
                stdin, stdout, stderr = ssh.exec_command(command_mkdir)
                exit_status = stdout.channel.recv_exit_status()

                if exit_status != 0:
                    error_msg = stderr.read().decode().strip()
                    raise RuntimeError(f"Failed to create config directory: {error_msg}")

                # Use SFTP to transfer the file instead of heredoc
                with ssh.open_sftp() as sftp:
                    # Create a temporary file-like object
                    import io

                    file_obj = io.BytesIO(config_json.encode("utf-8"))

                    # Upload file via SFTP
                    sftp.putfo(file_obj, config_file_path)

                    # Set proper permissions
                    sftp.chmod(config_file_path, 0o644)

                # Verify file was created
                stdin, stdout, stderr = ssh.exec_command(f"test -f {config_file_path} && echo 'OK'")
                verification = stdout.read().decode().strip()

                if verification != "OK":
                    raise RuntimeError("Config file verification failed")

                self.logger.info(f"✅ Config deployed successfully to agent {self.name} at {config_file_path}")
                return True

        except FileNotFoundError as e:
            self.logger.error(f"🔑 SSH key error for agent {self.name}: {e}")
            return False
        except paramiko.AuthenticationException as e:
            self.logger.error(f"🔒 SSH authentication failed for agent {self.name}: {e}")
            return False
        except paramiko.SSHException as e:
            self.logger.error(f"🌐 SSH connection error to agent {self.name}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"💥 Config deployment failed for agent {self.name}: {e}", exc_info=True)
            return False

    def _start_docker_container(self, node_config: dict) -> bool:
        """
        Start Docker container on execution server with proper configuration.
        Delegates to ContainerManager for consistency with verify.py.
        """
        self.logger.debug(f"🐳 Container [1/4] - Starting Docker container on {self.ip}")

        try:
            # Prepare node info
            # main.py uses self.ctrl_port etc.
            # ContainerManager expects keys in node_info.
            node_info = node_config.copy()
            node_info["physical_ip"] = self.ip
            node_info["name"] = self.name
            node_info["host_port_ctrl"] = self.ctrl_port
            node_info["container_port_ctrl"] = self.container_ctrl_port
            node_info["host_port_p2p"] = self.host_p2p_port

            # Load experiment parameters for network conditions
            # main.py logic loaded parameters.json manually in the original code.
            # We should do the same or pass it in.
            # WaflAgent.__init__ receives experiment_parameters, but doesn't store it fully?
            # It stores self.config, but experiment_parameters is passed to _deploy_configurations.
            # Let's check if we can access it.
            # Wait, WaflAgent is initialized with experiment_parameters.
            # But it doesn't seem to save it as an attribute in __init__.
            # Let's reload parameters.json as the original code did, to be safe.

            params_path = os.path.join("ctrl", "parameters.json")
            experiment_params = {}
            try:
                with open(params_path) as f:
                    experiment_params = json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to read parameters.json: {e}. Using defaults.")

            # Start container using manager
            success = self.container_manager.start_wafl_container(
                node_info=node_info,
                experiment_params=experiment_params,
                env_vars="-e LOG_LEVEL=INFO",
            )

            if not success:
                return False

            # Step 4: Wait for container to be ready
            return self._wait_for_container_ready()

        except Exception as e:
            self.logger.error(
                f"💥 Failed to start Docker container for agent {self.name}: {e}",
                exc_info=True,
            )
            return False

    def _wait_for_container_ready(self) -> bool:
        """
        Wait for Docker container to be ready by checking TCP port availability.

        Returns:
            bool: True if container is ready, False if timeout
        """
        self.logger.debug(f"Container [4/4] - Waiting for readiness at {self.ip}:{self.container_ctrl_port}")

        max_retries = 30
        retry_interval = 2

        for i in range(max_retries):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(2)
                    result = s.connect_ex((self.ip, self.container_ctrl_port))
                    if result == 0:
                        self.status = "READY"
                        self.logger.info(f"✅ Container [4/4] - Ready and listening on port {self.container_ctrl_port}")
                        return True
            except Exception:
                pass

            self.logger.debug(f"⏳ Waiting for container ({i + 1}/{max_retries})...")
            time.sleep(retry_interval)

        self.logger.error("❌ Timeout waiting for container to be ready")
        self.status = "ERROR"
        return False

    def start_remote_process(self, experiment_id: str, node_config: dict) -> bool:
        """
        Start remote process (Docker container) on execution server.

        This is the main entry point that orchestrates container lifecycle.

        Args:
            experiment_id: Experiment identifier
            node_config: Node configuration from execution_config.json

        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.debug(f"Starting remote process - Experiment: {experiment_id}, Node: {self.ip}")

        # Start Docker container with all configuration
        return self._start_docker_container(node_config)

    def _send_command(self, command: str) -> Tuple[bool, str]:
        """
        Send command to control TCP port and receive response.

        Returns:
            Tuple[bool, str]: (success flag, response string)
        """
        self.logger.debug(f"📤 Sending TCP command to {self.ip}:{self.ctrl_port}: {command.strip()}")

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(self.timeout)
                s.connect((self.ip, self.ctrl_port))
                s.sendall(command.encode("utf-8"))

                response_parts = []
                while True:
                    data = s.recv(4096)
                    if not data:
                        break
                    response_parts.append(data)

                full_response = b"".join(response_parts).decode("utf-8").strip()
                self.logger.debug(f"📥 Received TCP response from {self.ip}:{self.ctrl_port}: {full_response}")
                return True, full_response

        except socket.timeout:
            error_msg = f"⏰ TCP connection to {self.ip}:{self.ctrl_port} timed out after {self.timeout}s"
            self.logger.error(error_msg)
            return False, "ERROR:TIMEOUT"
        except ConnectionRefusedError:
            error_msg = f"🚫 Connection refused to {self.ip}:{self.ctrl_port}"
            self.logger.error(error_msg)
            return False, "ERROR:CONNECTION_REFUSED"
        except socket.error as e:
            error_msg = f"🌐 TCP socket error on {self.ip}:{self.ctrl_port}: {e}"
            self.logger.error(error_msg)
            return False, f"ERROR:{e}"
        except Exception as e:
            error_msg = f"💥 Unexpected error in TCP communication: {e}"
            self.logger.error(error_msg, exc_info=True)
            return False, f"ERROR:UNEXPECTED:{e}"

    def get_status(self) -> Tuple[str, List[str]]:
        """
        Send STAT command and get status and stdout.

        Returns:
            Tuple[str, List[str]]: (status string, stdout list)
        """
        success, response = self._send_command("STAT\r\n")

        if not success:
            self.status = "ERROR"
            self.logger.warning(f"📊 Failed to get status from agent {self.name}: {response}")
            return "ERROR_COMM", [response]

        try:
            lines = response.split("\n")
            if not lines or not lines[0]:
                self.status = "ERROR"
                self.logger.error(f"📊 Empty response received from agent {self.name}")
                return "ERROR_PARSE", ["Empty response from agent"]

            first_line = lines[0].strip()
            parts = first_line.split(":", 1)
            status_code = parts[0]
            logs = lines[1:]

            # Validate log count if specified
            if len(parts) == 2 and parts[1].isdigit():
                expected_log_count = int(parts[1])
                if len(logs) != expected_log_count:
                    self.logger.warning(f"📊 Log count mismatch for agent {self.name}. Expected {expected_log_count}, got {len(logs)}")

            # Validate status code
            valid_statuses = ["EXEC", "DONE", "ERROR", "READY"]
            if status_code not in valid_statuses and not status_code.startswith(tuple(valid_statuses)):
                self.logger.warning(f"📊 Unrecognized status format from agent {self.name}: {first_line}")
                self.status = "ERROR"
                return "ERROR_FORMAT", [f"Unrecognized status: {first_line}"]

            self.status = status_code
            self.logger.debug(f"📈 Status update for agent {self.name}: {status_code}")
            return status_code, logs

        except Exception as e:
            self.logger.error(
                f"💥 Error parsing status response from agent {self.name}: {e}",
                exc_info=True,
            )
            self.status = "ERROR"
            return "ERROR_PARSE", [f"Parse error: {e}"]

    def begin_epoch(self, phase: str, epoch: int) -> bool:
        """
        Send BEGIN command to start training epoch.

        Returns:
            bool: True if successful, False otherwise
        """
        command = f"BEGIN-{phase}-{epoch:05d}"

        self.logger.debug(f"Epoch {epoch} ({phase}) - Sending start command to agent {self.name}")

        success, response = self._send_command(f"{command}\r\n")
        if not success:
            self.logger.error(f"❌ Failed to send BEGIN command to agent {self.name}: {response}")
            return False

        if response != "OK":
            self.logger.error(f"❌ Agent {self.name} rejected BEGIN command. Response: {response}")
            return False

        self.status = "RUNNING"
        self.logger.debug(f"Epoch {epoch} - Agent {self.name} acknowledged start command")
        return True

    def begin_evaluation(self, eval_name: str = "eval") -> bool:
        """
        Start evaluation routine with BEGIN-eval command.

        Returns:
            bool: True if successful, False otherwise
        """
        command = f"BEGIN-{eval_name}"
        self.logger.info(f"Evaluation '{eval_name}' - Sending command to agent {self.name}")

        success, response = self._send_command(f"{command}\r\n")
        if not success:
            self.logger.error(f"❌ Failed to send evaluation command to agent {self.name}: {response}")
            return False

        if response != "OK":
            self.logger.error(f"❌ Agent {self.name} rejected evaluation command. Response: {response}")
            return False

        self.logger.info(f"Evaluation '{eval_name}' - Agent {self.name} started")
        return True

    def send_kill_command(self) -> bool:
        """
        Send KILL command to terminate process normally.
        """
        self.logger.warning(f"🛑 Sending graceful shutdown command to agent {self.name}")

        success, response = self._send_command("KILL\r\n")
        if success and response == "OK":
            self.logger.info(f"Shutdown - Agent {self.name} acknowledged gracefully")
            self.status = "TERMINATED"
            return True
        else:
            self.logger.error(f"❌ Graceful shutdown failed for agent {self.name}. Response: {response}")
            return False

    def force_kill_process(self, args: Dict[str, Any]) -> bool:
        """
        Force kill process via SSH.

        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.error(f"💀 Force killing process on agent {self.name}")

        if self.pid is None:
            self.logger.warning(f"⚠️ No PID available for agent {self.name}, cannot force kill")
            return False

        try:
            ssh_port = 22
            username = args["USER"]
            private_key_path = os.path.expanduser("~/.ssh/id_ed25519")
            key = paramiko.Ed25519Key.from_private_key_file(private_key_path)
            command_kill = f"kill -9 {self.pid}"

            with paramiko.SSHClient() as ssh:
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(self.ip, port=ssh_port, username=username, pkey=key, timeout=10)
                stdin, stdout, stderr = ssh.exec_command(command_kill)
                exit_status = stdout.channel.recv_exit_status()

                if exit_status == 0:
                    self.status = "TERMINATED"
                    self.logger.info(f"Force kill - Agent {self.name} terminated successfully")
                    return True
                else:
                    error_msg = stderr.read().decode().strip()
                    self.logger.error(f"❌ Force kill failed for agent {self.name}: {error_msg}")
                    return False

        except Exception as e:
            self.logger.error(f"💥 Error during force kill for agent {self.name}: {e}", exc_info=True)
            return False


class ControlServer:
    """
    Manages and controls entire WAFL experiment.
    Main implementation of ctrl/main.py.
    """

    def __init__(self):
        self.logger = logging.getLogger("ControlServer")
        self.config = self._load_config()
        self.experiment_id = self._generate_experiment_id(self.config.get("EXPERIMENT_NAME", "exp"))
        self.results_dir = self._create_results_directory()
        self.agents: List[WaflAgent] = []
        self.mobility_aware_config = None
        self.network_conditions = None
        self.path_loss_model = None
        self._setup_logging()
        self._load_mobility_aware_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load config file (execution_config.json)."""
        config_path = os.path.join("ctrl", "execution_config.json")
        self.logger.debug(f"Loading configuration from {config_path}")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"🚫 Config file not found: {config_path}")

        try:
            with open(config_path, "r") as f:
                exec_config = json.load(f)

            # Extract device names and IPs from nodes array
            nodes = exec_config.get("nodes", [])
            device_names = [str(node["name"]) for node in nodes]
            device_ips = [node["physical_ip"] for node in nodes]

            # Read log level (default to 'info')
            log_level = exec_config.get("log_level", "info").upper()

            # Build config dictionary from execution_config.json
            # Transform from execution_config format to internal config format
            config = {
                "USER": exec_config.get("user", "denjo"),
                "DEPLOYMENT_LOCATION": exec_config.get("deployment_location", "/home/denjo/workspace/ktakahashi"),
                "PROJECT_NAME": "WAFL-Testbed",
                "WAFL_DEVICE_P2P_PORT": 10002,
                "WAFL_DEVICE_NAMES": device_names,
                "WAFL_DEVICE_IPS": device_ips,
                "EXPERIMENT_NAME": exec_config.get("experiment_name", "wafl-experiment"),
                "LOG_LEVEL": log_level,
            }

            self.logger.debug(f"Configuration loaded successfully ({len(device_names)} devices)")
            return config

        except Exception as e:
            self.logger.error(f"💥 Failed to load configuration: {e}", exc_info=True)
            raise

    def _generate_experiment_id(self, name: str) -> str:
        """Generate experiment ID in 'experiment-name-timestamp' format."""
        now = datetime.datetime.now()
        timestamp_str = now.strftime("%Y%m%dT%H%M%S")
        self.start_timestamp = now.timestamp()  # Store epoch timestamp
        experiment_id = f"{name}-{timestamp_str}"
        self.logger.debug(f"Generated experiment ID: {experiment_id}")
        return experiment_id

    def _create_results_directory(self):
        """Create directory to save experiment results."""
        results_path = os.path.join("results", self.experiment_id, "ctrl")
        try:
            os.makedirs(results_path, exist_ok=True)
            self.logger.debug(f"Created results directory: {results_path}")
            return results_path
        except Exception as e:
            self.logger.error(f"💥 Failed to create results directory {results_path}: {e}")
            raise

    def _create_agents(self, experiment_parameters: Dict[str, Any]) -> list:
        """Create agents from execution config, using JSON topology if available."""
        # Check for execution_config.json
        json_config_path = os.path.join("ctrl", "execution_config.json")
        if os.path.exists(json_config_path):
            self.logger.debug(f"Loading topology from {json_config_path}")
            try:
                with open(json_config_path, "r") as f:
                    topology = json.load(f)

                nodes = topology.get("nodes", [])
                if not nodes:
                    raise ValueError("No nodes found in execution_config.json")

                # Add experiment_id and results_dir to experiment_parameters
                experiment_parameters["experiment_id"] = self.experiment_id
                experiment_parameters["results_dir"] = self.results_dir

                # Add mobility-aware configuration if it exists
                if self.network_conditions or self.path_loss_model:
                    node_ip_mapping = {str(node["name"]): node["physical_ip"] for node in nodes}
                    experiment_parameters["mobility_aware"] = {
                        "enabled": True,  # Keep enabled flag for consistency
                        "network_conditions": self.network_conditions,
                        "node_ip_mapping": node_ip_mapping,
                    }
                    self.logger.debug("Mobility-aware configuration added to experiment parameters")

                # Parallel agent creation
                self.logger.info(f"Creating {len(nodes)} agents in parallel...")

                from concurrent.futures import ThreadPoolExecutor, as_completed

                def create_single_agent(node):
                    """Create a single agent with error handling."""
                    try:
                        device_name = str(node["name"])
                        ip = node["physical_ip"]
                        ctrl_port = node.get("container_port_ctrl", 10001)

                        agent = WaflAgent(
                            agent_index=node["name"],
                            device_name=device_name,
                            ip_address=ip,
                            ctrl_port=ctrl_port,
                            config=self.config,
                            experiment_parameters=experiment_parameters,
                            experiment_id=self.experiment_id,
                            start_timestamp=self.start_timestamp,
                            container_ctrl_port=ctrl_port,
                            host_p2p_port=node.get("host_port_p2p", 10002),
                            container_p2p_port=node.get("container_port_p2p", 10002),  # Ensure this is passed
                            node_config=node,
                        )
                        return (True, agent, device_name, None)
                    except Exception as e:
                        return (False, None, str(node.get("name")), e)

                agents = []
                failed_agents = []

                # Use ThreadPoolExecutor for parallel creation
                with ThreadPoolExecutor(max_workers=len(nodes)) as executor:
                    # Submit all agent creation tasks
                    futures = {executor.submit(create_single_agent, node): node for node in nodes}

                    # Collect results as they complete
                    for future in as_completed(futures):
                        success, agent, name, error = future.result()
                        if success:
                            agents.append(agent)
                            self.logger.debug(f"Agent {name} created successfully")
                        else:
                            failed_agents.append(name)
                            self.logger.error(f"Failed to create agent {name}: {error}")

                if failed_agents:
                    raise RuntimeError(f"❌ Failed to create agents: {', '.join(failed_agents)}")

                # Sort agents by index to maintain order
                agents.sort(key=lambda a: a.agent_index)
                self.logger.debug(f"Created {len(agents)} agents from topology configuration)")
                return agents
            except Exception as e:
                self.logger.error(f"💥 Failed to load topology from JSON: {e}")
                raise
        else:
            raise FileNotFoundError(f"Execution config not found: {json_config_path}")

    def _setup_logging(self):
        """Setup experiment logging to file and console."""
        try:
            log_file = os.path.join(self.results_dir, "output.log")
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            # Clear any existing handlers to avoid duplicate logs
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)

            # Get log level from config (default to INFO)
            log_level_str = self.config.get("LOG_LEVEL", "INFO")
            log_level = getattr(logging, log_level_str, logging.INFO)

            logging.basicConfig(
                level=log_level,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                handlers=[
                    logging.FileHandler(log_file, encoding="utf-8"),
                    logging.StreamHandler(),
                ],
                force=True,
            )

            # Suppress paramiko verbose logs (SSH connection details)
            logging.getLogger("paramiko").setLevel(logging.WARNING)

            self.logger.debug(f"Logging configured - Output: {log_file}, Level: {log_level_str}")
            self.logger.info(f"Experiment ID: {self.experiment_id}")

        except Exception as e:
            print(f"💥 Failed to setup logging: {e}")
            raise

    def _load_mobility_aware_config(self):
        """Load mobility-aware configuration if enabled."""
        params_path = "ctrl/parameters.json"
        try:
            with open(params_path) as f:
                params = json.load(f)

            mobility_config = params.get("mobility_aware", {})
            if not mobility_config.get("enabled", False):
                self.logger.debug("Mobility mode: Disabled (using static network conditions)")
                return

            self.logger.info("Mobility mode: Enabled")
            self.mobility_aware_config = mobility_config

            # Load network conditions
            conditions_file = os.path.join(
                "data",
                mobility_config.get("network_conditions_file", "network_conditions_mobility.json"),
            )
            if os.path.exists(conditions_file):
                with open(conditions_file) as f:
                    self.network_conditions = json.load(f)
                self.logger.info(f"✅ Loaded network conditions from {conditions_file}")
            else:
                self.logger.warning(f"⚠️ Network conditions file not found: {conditions_file}")

            # Load path loss model
            model_file = os.path.join(
                "data",
                mobility_config.get("path_loss_model_file", "sumo/path_loss_model.json"),
            )
            if os.path.exists(model_file):
                with open(model_file) as f:
                    self.path_loss_model = json.load(f)
                self.logger.info(f"✅ Loaded path loss model from {model_file}")
            else:
                self.logger.warning(f"⚠️ Path loss model file not found: {model_file}")

        except Exception as e:
            self.logger.error(f"💥 Failed to load mobility-aware config: {e}")
            self.mobility_aware_config = None

    def run_experiment(
        self,
        epochs: Dict[str, int],
        wafl_phase: Dict[str, Any],
        contact_pattern: str,
        ssp_config: Dict[str, Any] = None,
    ):
        """
        Execute entire experiment sequence (startup, training loop, shutdown).

        Args:
            epochs: Dictionary with 'self' and 'wafl' epoch counts
            wafl_phase: WAFL phase parameters
            contact_pattern: Contact pattern file name
            ssp_config: SSP configuration with 'staleness' and 'ssp_threshold' keys

        Note:
            Agent shutdown is always performed in the finally block, even if an exception occurs during the experiment.
        """
        self.logger.info(f"🚀 Experiment started - ID: {self.experiment_id} (SELF: {epochs['self']}, WAFL: {epochs['wafl']} epochs)")
        experiment_success = False

        # Set default SSP config if not provided
        if ssp_config is None:
            ssp_config = {"staleness": 0, "ssp_threshold": 1.0}

        try:
            # 0. Create agents with unified configuration deployment
            self.logger.info("📋 Phase [0/4] - Agent creation and configuration deployment")
            experiment_parameters = {
                "epochs": epochs,
                "wafl_phase": wafl_phase,
                "contact_pattern": contact_pattern,
            }

            self.agents = self._create_agents(experiment_parameters)
            self.logger.info("✅ Phase [0/4] - Complete (all agents ready)")

            # Verify container configurations after startup
            self.logger.info("🔍 Verifying container configurations...")

            try:
                import verify

                # Load configurations
                config_validator = verify.ConfigValidator()
                if config_validator.load_configs():
                    # Validate config files with logger
                    params_valid = config_validator.validate_parameters(logger=self.logger)
                    exec_valid = config_validator.validate_execution_config(logger=self.logger)

                    if params_valid and exec_valid:
                        self.logger.info("✅ Configuration files validated")

                        # Verify container applications with logger
                        container_verifier = verify.ContainerVerifier(
                            config_validator.params,
                            config_validator.exec_config,
                            verbose=False,
                        )

                        containers_valid = container_verifier.verify_all(logger=self.logger)

                        if containers_valid and len(container_verifier.errors) == 0:
                            self.logger.info("✅ All container verifications passed")
                            if container_verifier.warnings:
                                for warning in container_verifier.warnings:
                                    self.logger.warning(f"⚠ {warning}")
                        else:
                            self.logger.error("💥 Container verification FAILED")
                            for error in container_verifier.errors:
                                self.logger.error(f"✗ {error}")
                            raise RuntimeError("Container verification failed. Please check container configurations.")
                    else:
                        for error in config_validator.errors:
                            self.logger.error(f"✗ {error}")
                        raise RuntimeError("Configuration validation failed")
                else:
                    for error in config_validator.errors:
                        self.logger.error(f"✗ {error}")
                    raise RuntimeError("Failed to load configuration files")

            except ImportError as e:
                self.logger.warning(f"⚠️ Could not import verification module: {e}")
                self.logger.warning("Skipping container verification...")
            except Exception as e:
                self.logger.error(f"💥 Verification error: {e}")
                raise

            # 1. Run SELF phase
            self.logger.info(f"🏃 Phase [1/4] - SELF training ({epochs['self']} epochs)")
            # SELF phase is independent, so staleness is effectively infinite or irrelevant.
            # We use a large staleness to allow free running.
            self._run_phase("SELF", epochs["self"], staleness=999999)
            self.logger.info("✅ Phase [1/4] - Complete (SELF training finished)")

            # 2. Run WAFL phase
            self.logger.info(f"🤝 Phase [2/4] - WAFL training ({epochs['wafl']} epochs)")

            # Log WAFL strategy details
            staleness = ssp_config.get("staleness", 0)
            ssp_threshold = ssp_config.get("ssp_threshold", 1.0)
            aggregation = wafl_phase.get("aggregation_strategy", "FedAvg")

            self.logger.info("📊 WAFL Strategy Configuration:")
            self.logger.info(f"   - Aggregation: {aggregation}")
            self.logger.info(f"   - Synchronization: SSP (Staleness={staleness}, Threshold={ssp_threshold:.1%})")
            self.logger.info(f"   - Batch Size: {wafl_phase.get('batch_size', 32)}")
            self.logger.info(f"   - Learning Rate: {wafl_phase.get('learning_rate', 0.001)}")
            self.logger.info(f"   - Coefficiency: {wafl_phase.get('coefficiency', 1.0)}")

            self._run_phase("WAFL", epochs["wafl"], staleness=staleness, ssp_threshold=ssp_threshold)

            self.logger.info("✅ Phase [2/4] - Complete (WAFL training finished)")
            experiment_success = True

        except KeyboardInterrupt:
            self.logger.warning("⚠️ Experiment interrupted by user")
        except Exception as e:
            self.logger.error(f"💥 Experiment failed: {e}", exc_info=True)
        finally:
            # 3. Shutdown all agents
            self.logger.info("🛑 Phase [3/4] - Shutting down all agents")
            self._shutdown_all_agents()

            status = "SUCCESS" if experiment_success else "FAILED"
            self.logger.info(f"✅ Experiment complete - ID: {self.experiment_id}, Status: {status}")

    def _run_phase(
        self,
        phase_name: str,
        total_epochs: int,
        staleness: int,
        ssp_threshold: float = 1.0,
    ):
        """
        Run a single training phase (SELF or WAFL) with SSP-based synchronization.
        ssp_threshold: Fraction of agents (0.0-1.0) required to complete an epoch before forcing others to skip.
        """
        if phase_name == "WAFL":
            self.logger.debug(f"Starting WAFL phase with staleness={staleness}, threshold={ssp_threshold:.1%}")
        else:
            self.logger.debug(f"Starting {phase_name} phase with {total_epochs} epochs")

        # Initialize agent status
        agent_status = {agent.name: "IDLE" for agent in self.agents}
        epochs_completed = {agent.name: 0 for agent in self.agents}

        start_time = time.time()
        last_progress_log = 0
        last_epoch_logged = -1

        while True:
            # Check if all agents have completed all epochs
            min_epoch = min(epochs_completed.values())

            # Log epoch progress for WAFL with strategy context
            if phase_name == "WAFL" and min_epoch > last_epoch_logged:
                max_epoch = max(epochs_completed.values())
                epoch_spread = max_epoch - min_epoch
                if epoch_spread > staleness:
                    self.logger.info(f"⚠️  WAFL Epoch {min_epoch}: Spread={epoch_spread} (exceeds staleness={staleness})")
                else:
                    self.logger.debug(f"WAFL Epoch {min_epoch}: Spread={epoch_spread}, agents in sync")
                last_epoch_logged = min_epoch

            if min_epoch >= total_epochs:
                break

            # SSP Reset Check
            if ssp_threshold < 1.0:
                # Check if enough agents have completed max_epoch (or any epoch > min_epoch)
                # Actually, we usually check if enough agents completed epoch E to force everyone to E+1?
                # Or if enough completed E, force everyone to finish E?
                # The plan says: "When completed nodes reach N * p ... FORCE_NEXT_EPOCH ... Uncompleted nodes discard ... and proceed to next epoch"
                # So if N*p agents finish epoch E, we force everyone else to finish E (skip) and be ready for E+1.

                # Count completions for each epoch
                epoch_counts = {}
                for e in epochs_completed.values():
                    epoch_counts[e] = epoch_counts.get(e, 0) + 1

                # Check if any epoch E has enough completions
                # We are interested in the highest epoch that has reached threshold
                # But we only care about epochs > min_epoch?
                # If min_epoch is 5, and 90% are at 6, we force the 10% at 5 to skip to 6.

                target_epoch = -1
                for e in sorted(epoch_counts.keys(), reverse=True):
                    if e > min_epoch:
                        count = 0
                        # Count agents who have completed e OR HIGHER
                        for ae in epochs_completed.values():
                            if ae >= e:
                                count += 1

                        if count >= len(self.agents) * ssp_threshold:
                            target_epoch = e
                            break

                if target_epoch != -1:
                    # SSP Threshold enforcement
                    if ssp_threshold < 1.0:
                        max_epoch = max(epochs_completed.values())
                        slow_agents = [name for name, e in epochs_completed.items() if e < max_epoch - staleness]
                        if slow_agents and len(slow_agents) / len(self.agents) > (1 - ssp_threshold):
                            self.logger.info(f"⚡ SSP Threshold: {len(slow_agents)}/{len(self.agents)} agents slow, skipping to epoch {target_epoch}")
                            self.logger.debug(f"   Slow agents: {slow_agents}")
                    # Force everyone < target_epoch to skip to target_epoch
                    self.logger.info(f"⚡ SSP Threshold reached for epoch {target_epoch}. Forcing slow agents to skip.")
                    for agent in self.agents:
                        if epochs_completed[agent.name] < target_epoch:
                            self.logger.warning(f"⏩ Forcing agent {agent.name} (epoch {epochs_completed[agent.name]}) to skip to {target_epoch}")
                            # Send FORCE_NEXT command
                            # We might need to send it multiple times if they are far behind?
                            # Or FORCE_NEXT just stops current.
                            # We need to update our tracking.

                            # If agent is RUNNING, stop it.
                            if agent_status[agent.name] == "RUNNING":
                                success, response = agent._send_command("FORCE_NEXT\r\n")
                                # We assume it stops and reports something?
                                # Actually, we just assume it stops.
                                agent_status[agent.name] = "IDLE"

                            # Update local state to pretend it finished
                            epochs_completed[agent.name] = target_epoch

            # Apply dynamic network conditions if mobility-aware mode is enabled
            if self.mobility_aware_config and phase_name == "WAFL":
                self._apply_dynamic_network_conditions(min_epoch)

            # Schedule agents
            for agent in self.agents:
                current_epoch = epochs_completed[agent.name]

                if current_epoch < total_epochs:
                    # SSP Constraint
                    if (current_epoch + 1) - min_epoch <= staleness + 1:
                        if agent_status[agent.name] == "IDLE":
                            next_epoch = current_epoch + 1
                            success = agent.begin_epoch(phase_name, next_epoch)
                            if success:
                                agent_status[agent.name] = "RUNNING"
                            else:
                                self.logger.warning(f"Failed to start epoch {next_epoch} on agent {agent.name}, retrying...")

            # Poll status
            for agent in self.agents:
                if agent_status[agent.name] == "RUNNING":
                    status, logs = agent.get_status()

                    if logs:
                        for log_line in logs:
                            if log_line.strip():
                                self.logger.debug(f"[{agent.name}] {log_line}")

                    if "ERROR" in status:
                        self.logger.error(f"❌ Agent {agent.name} encountered error: {status}")
                        agent_status[agent.name] = "ERROR"
                    elif status.startswith("DONE"):
                        try:
                            parts = status.split("-")
                            if len(parts) >= 3:
                                done_epoch = int(parts[2])
                                if done_epoch > epochs_completed[agent.name]:
                                    epochs_completed[agent.name] = done_epoch
                                    agent_status[agent.name] = "IDLE"
                                    self.logger.info(f"✅ Agent {agent.name} completed {phase_name} epoch {done_epoch}")
                        except Exception as e:
                            self.logger.error(f"Error parsing status {status}: {e}")

            # Progress logging
            elapsed_time = time.time() - start_time
            if elapsed_time - last_progress_log >= 30:
                self.logger.info(f"📊 {phase_name} Progress: Min Epoch {min_epoch}/{total_epochs}")
                last_progress_log = elapsed_time

            time.sleep(1)

    def _shutdown_all_agents(self):
        """Shutdown all agents gracefully by stopping Docker containers in parallel."""
        if not self.agents:
            self.logger.warning("No agents to shutdown")
            return

        self.logger.warning(f"🛑 Shutting down {len(self.agents)} agents")

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def shutdown_single_agent(agent):
            """Shutdown a single agent."""
            if agent is None:
                return (False, None, "Agent is None")

            try:
                # Send KILL command to container
                self.logger.debug(f"Sending KILL to agent {agent.name}")
                agent.send_kill_command()

                # Stop Docker container via SSH
                container_name = f"wafl-node-{agent.name}"
                ssh_port = 22
                username = self.config.get("USER", "denjo")
                key_path = os.path.expanduser("~/.ssh/id_ed25519")

                with paramiko.SSHClient() as ssh:
                    key = paramiko.Ed25519Key.from_private_key_file(key_path)
                    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    ssh.connect(agent.ip, port=ssh_port, username=username, pkey=key, timeout=10)

                    # Stop and remove container
                    stop_cmd = f"docker stop {container_name} && docker rm {container_name}"
                    stdin, stdout, stderr = ssh.exec_command(stop_cmd)
                    exit_status = stdout.channel.recv_exit_status()

                    if exit_status == 0:
                        return (True, agent.name, None)
                    else:
                        error_msg = stderr.read().decode().strip()
                        return (False, agent.name, error_msg)

            except Exception as e:
                return (False, agent.name if agent else "unknown", str(e))

        # Shutdown all agents in parallel
        shutdown_count = 0
        failed = []

        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            futures = {executor.submit(shutdown_single_agent, agent): agent for agent in self.agents}

            for future in as_completed(futures):
                success, name, error = future.result()
                if success:
                    shutdown_count += 1
                    self.logger.debug(f"Agent {name} shutdown complete")
                else:
                    failed.append(name)
                    if error:
                        self.logger.error(f"Failed to shutdown agent {name}: {error}")

        if failed:
            self.logger.warning(f"Failed to shutdown {len(failed)} agents: {failed}")

        self.logger.info(f"🏁 Agent shutdown complete ({shutdown_count}/{len(self.agents)} containers stopped)")

    def _apply_dynamic_network_conditions(self, epoch: int):
        """Apply dynamic network conditions for the given epoch using tc."""
        if not self.network_conditions or not self.path_loss_model:
            return

        # Check if epoch is within bounds (array-based format)
        if epoch >= len(self.network_conditions):
            self.logger.debug(f"No network conditions for epoch {epoch} (only {len(self.network_conditions)} epochs available)")
            return

        self.logger.info(f"📡 Applying dynamic network conditions for epoch {epoch}")

        # Get path loss model file path
        model_file = os.path.join(
            "data",
            self.mobility_aware_config.get("path_loss_model_file", "sumo/path_loss_model.json"),
        )
        conditions_file = os.path.join(
            "data",
            self.mobility_aware_config.get("network_conditions_file", "network_conditions_mobility.json"),
        )
        exec_config_file = "ctrl/execution_config.json"

        # Apply tc rules for each agent
        for agent in self.agents:
            container_name = f"wafl-node-{agent.agent_index}"
            node_id = str(agent.agent_index)

            # Run apply_dynamic_tc.py via SSH with execution-config for IP mapping
            cmd = f"cd {self.config.get('deployment_location', '/home/denjo')} && python3 utils/apply_dynamic_tc.py --container {container_name} --epoch {epoch} --node-id {node_id} --conditions {conditions_file} --pathloss {model_file} --execution-config {exec_config_file}"

            try:
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(agent.ip, username=self.config.get("user", "denjo"))

                stdin, stdout, stderr = ssh.exec_command(cmd)
                exit_status = stdout.channel.recv_exit_status()

                if exit_status == 0:
                    self.logger.debug(f"✅ Applied tc rules for agent {agent.name} (epoch {epoch})")
                else:
                    error_msg = stderr.read().decode().strip()
                    self.logger.warning(f"⚠️ Failed to apply tc rules for agent {agent.name}: {error_msg}")

                ssh.close()
            except Exception as e:
                self.logger.error(f"💥 Error applying tc for agent {agent.name}: {e}")


if __name__ == "__main__":
    try:
        # Parameters file path
        PARAMETERS_PATH = "ctrl/parameters.json"

        # Load experiment parameters from JSON file
        try:
            with open(PARAMETERS_PATH, "r", encoding="utf-8") as f:
                experiment_parameters = json.load(f)
            print(f"📄 Loaded experiment parameters from: {PARAMETERS_PATH}")
        except json.JSONDecodeError as e:
            print(f"💥 Error parsing JSON file {PARAMETERS_PATH}: {e}")
            exit(1)
        except Exception as e:
            print(f"💥 Error reading parameters file {PARAMETERS_PATH}: {e}")
            exit(1)

        # Validate required parameters
        required_params = ["epochs", "contact_pattern", "wafl_phase"]
        missing_params = [param for param in required_params if param not in experiment_parameters]

        if missing_params:
            print(f"💥 Missing required parameters in {PARAMETERS_PATH}: {', '.join(missing_params)}")
            exit(1)

        epochs_self = experiment_parameters["epochs"]["self"]
        epochs_wafl = experiment_parameters["epochs"]["wafl"]

        print(f"🚀 Starting experiment with {epochs_self} SELF epochs and {epochs_wafl} WAFL epochs")
        print(f"📋 Contact pattern: {experiment_parameters['contact_pattern']}")

        # Log wafl_phase parameters
        wafl_phase = experiment_parameters["wafl_phase"]
        print("\n📊 WAFL Phase Parameters:")
        print(f"   - Aggregation Strategy: {wafl_phase.get('aggregation_strategy', 'FedAvg')}")
        print(f"   - Batch Size: {wafl_phase.get('batch_size', 32)}")
        print(f"   - Learning Rate: {wafl_phase.get('learning_rate', 0.001)}")
        print(f"   - Coefficiency: {wafl_phase.get('coefficiency', 1.0)}")

        # Log network_condition parameters
        net_cond = experiment_parameters.get("network_condition", {})
        net_enabled = net_cond.get("enabled", False)
        print("\n🌐 Network Condition Parameters:")
        print(f"   - Enabled: {'Yes' if net_enabled else 'No'}")
        if net_enabled:
            print(f"   - Delay: {net_cond.get('delay', '50ms')}")
            print(f"   - Loss: {net_cond.get('loss', '0%')}")
            print(f"   - Rate: {net_cond.get('rate', '100mbit')}")

        # Log mobility_aware parameters
        mobility = experiment_parameters.get("mobility_aware", {})
        mobility_enabled = mobility.get("enabled", False)
        print("\n📍 Mobility-Aware Parameters:")
        print(f"   - Enabled: {'Yes' if mobility_enabled else 'No'}")
        if mobility_enabled:
            print(f"   - Contact Pattern File: {mobility.get('contact_pattern_file', 'N/A')}")
            print(f"   - Network Conditions File: {mobility.get('network_conditions_file', 'N/A')}")
            print(f"   - Path Loss Model File: {mobility.get('path_loss_model_file', 'N/A')}")

        # Log method parameters
        method = experiment_parameters.get("method", {})
        print("\n🔧 Method Parameters:")

        # SSP
        ssp_settings = method.get("ssp", {})
        ssp_enabled = ssp_settings.get("enabled", False)
        print(f"   - SSP: {'Enabled' if ssp_enabled else 'Disabled'}")
        if ssp_enabled:
            print(f"     • Staleness: {ssp_settings.get('staleness', 0)}")
            print(f"     • SSP Threshold: {ssp_settings.get('ssp_threshold', 1.0)}")

        # UDP
        udp_settings = method.get("udp", {})
        udp_enabled = udp_settings.get("enabled", False)
        print(f"   - UDP: {'Enabled' if udp_enabled else 'Disabled'}")
        if udp_enabled:
            print(f"     • FEC M: {udp_settings.get('fec_m', 9)}")

        # Compression
        comp_settings = method.get("compression", {})
        comp_enabled = comp_settings.get("enabled", False)
        print(f"   - Compression: {'Enabled' if comp_enabled else 'Disabled'}")
        if comp_enabled:
            print(f"     • Initial Method: {comp_settings.get('initial_method', 'zlib')}")

        # Create ControlServer instance
        controller = ControlServer()

        # Run experiment
        # Extract SSP settings
        staleness = ssp_settings.get("staleness", 0) if ssp_enabled else 0
        ssp_threshold = ssp_settings.get("ssp_threshold", 1.0) if ssp_enabled else 1.0

        controller.run_experiment(
            epochs={"self": epochs_self, "wafl": epochs_wafl},
            wafl_phase=experiment_parameters["wafl_phase"],
            contact_pattern=experiment_parameters["contact_pattern"],
            ssp_config={"staleness": staleness, "ssp_threshold": ssp_threshold},
        )

    except KeyboardInterrupt:
        print("\n⚠️ Experiment interrupted by user")
        exit(130)
    except FileNotFoundError as e:
        print(f"💥 File not found: {e}")
        exit(1)
    except Exception as e:
        print(f"💥 Fatal error in main: {e}")
        exit(1)

import datetime
import json
import logging
import os
import sys

# Add project root to sys.path to allow importing from ctrl package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import paramiko

from ctrl.container_manager import ContainerManager
from ctrl.ssh_connection_manager import SSHConnectionManager


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
        ssh_manager=None,
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
        self.node_config = node_config or {}
        self.ssh_manager = ssh_manager

        # Initialize ContainerManager (with shared SSH manager if available)
        self.container_manager = ContainerManager(
            user=self.config.get("USER", "denjo"),
            deployment_location=self.config.get("DEPLOYMENT_LOCATION", "/home/denjo"),
            project_name=self.config.get("PROJECT_NAME", "WAFL-Testbed"),
            logger=self.logger,
            ssh_manager=self.ssh_manager,
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
                "log_level": os.environ.get("LOG_LEVEL", "DEBUG"),
            },
            "timeouts": self.config["TIMEOUTS"],
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
            target_path = os.path.join(self.config["DEPLOYMENT_LOCATION"], self.config["PROJECT_NAME"])
            config_dir = os.path.join(target_path, "config")

            # Use SSH connection manager if available, otherwise create new connection
            if self.ssh_manager:
                ssh = self.ssh_manager.get_connection(self.ip)
                use_context_manager = False
            else:
                ssh_port = 22
                username = self.config["USER"]
                private_key_path = os.path.expanduser("~/.ssh/id_ed25519")

                if not os.path.exists(private_key_path):
                    raise FileNotFoundError(f"🔑 SSH private key not found at {private_key_path}")

                key = paramiko.Ed25519Key.from_private_key_file(private_key_path)
                self.logger.debug(f"🔗 Connecting to {username}@{self.ip} for configuration deployment")

                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(self.ip, port=ssh_port, username=username, pkey=key, timeout=10)
                use_context_manager = True

            try:
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
                contact_pattern_path = os.path.join("data", contact_pattern)

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

                # 3. Parameters file (for method settings like UDP and compression)
                parameters_path = os.path.join("ctrl", "parameters.json")
                if os.path.exists(parameters_path):
                    with open(parameters_path, "r", encoding="utf-8") as f:
                        parameters_data = json.load(f)
                    parameters_json = json.dumps(parameters_data, indent=2, ensure_ascii=False)
                    files_to_deploy.append(
                        {
                            "content": parameters_json,
                            "filename": "parameters.json",
                            "description": "parameters file",
                            "target_dir": "ctrl",
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

                    # Create ctrl directory for parameters.json
                    ctrl_dir = os.path.join(target_path, "ctrl")
                    try:
                        sftp.stat(ctrl_dir)
                    except FileNotFoundError:
                        sftp.mkdir(ctrl_dir)

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
                            # Use custom target directory if specified, otherwise use config_dir
                            target_directory = file_info.get("target_dir")
                            if target_directory:
                                file_path = os.path.join(target_path, target_directory, file_info["filename"])
                            else:
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

                    # NOTE: Dataset transfer (train.pkl, test.pkl) removed.
                    # It is now handled efficiently by deploy.py using rsync.

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

            finally:
                # Only close SSH connection if we created it (not using manager)
                if use_context_manager:
                    ssh.close()

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

            target_path = os.path.join(self.config["DEPLOYMENT_LOCATION"], self.config["PROJECT_NAME"])
            config_dir = os.path.join(target_path, "config")
            config_file_path = os.path.join(config_dir, f"config_{self.agent_index}.json")

            # Use SSH connection manager if available, otherwise create new connection
            if self.ssh_manager:
                ssh = self.ssh_manager.get_connection(self.ip)
                use_context_manager = False
            else:
                ssh_port = 22
                username = self.config["USER"]
                private_key_path = os.path.expanduser("~/.ssh/id_ed25519")

                if not os.path.exists(private_key_path):
                    raise FileNotFoundError(f"🔑 SSH private key not found at {private_key_path}")

                key = paramiko.Ed25519Key.from_private_key_file(private_key_path)
                self.logger.debug(f"🔗 Deploying config to {username}@{self.ip}:{config_file_path}")

                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(self.ip, port=ssh_port, username=username, pkey=key, timeout=10)
                use_context_manager = True

            try:
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

            finally:
                # Only close SSH connection if we created it (not using manager)
                if use_context_manager:
                    ssh.close()

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

            # Prepare environment variables
            env_vars_str = "-e LOG_LEVEL=DEBUG"
            wandb_key = os.environ.get("WANDB_API_KEY")

            # Fallback: Try reading .env file directly if key is missing
            if not wandb_key and os.path.exists(".env"):
                try:
                    with open(".env", "r") as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith("WANDB_API_KEY="):
                                wandb_key = line.split("=", 1)[1].strip()
                                # Remove quotes if present
                                if (wandb_key.startswith('"') and wandb_key.endswith('"')) or (wandb_key.startswith("'") and wandb_key.endswith("'")):
                                    wandb_key = wandb_key[1:-1]
                                break
                except Exception as e:
                    self.logger.warning(f"Failed to parse .env file: {e}")

            if wandb_key:
                env_vars_str += f" -e WANDB_API_KEY={wandb_key}"
                self.logger.debug("✅ WANDB_API_KEY found and passed to container")
            else:
                self.logger.warning("⚠️ WANDB_API_KEY not found in environment or .env. Logging to WandB will fail.")

            # Start container using manager
            success = self.container_manager.start_wafl_container(
                node_info=node_info,
                experiment_params=experiment_params,
                env_vars=env_vars_str,
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
            command_kill = f"kill -9 {self.pid}"

            # Use SSH connection manager if available, otherwise create new connection
            if self.ssh_manager:
                ssh = self.ssh_manager.get_connection(self.ip)
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
            else:
                ssh_port = 22
                username = args["USER"]
                private_key_path = os.path.expanduser("~/.ssh/id_ed25519")
                key = paramiko.Ed25519Key.from_private_key_file(private_key_path)

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
        self.start_timestamp = time.time()
        self._setup_logging()
        self._load_mobility_aware_config()

        # Initialize SSH connection manager for connection reuse
        self.ssh_manager = SSHConnectionManager(
            user=self.config.get("USER", "denjo"),
            private_key_path="~/.ssh/id_ed25519",
            timeout=30,
            logger=self.logger,
        )

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

            # Load timeouts configuration (REQUIRED)
            timeouts_config = exec_config.get("timeouts")
            if timeouts_config is None:
                raise ValueError("'timeouts' section is missing in execution_config.json. Please configure timeouts.")

            # Validate all required timeout values
            required_timeout_keys = ["model_fetch", "udp_initial_packet", "udp_inter_packet", "udp_model_completion"]
            missing_keys = [k for k in required_timeout_keys if k not in timeouts_config]
            if missing_keys:
                raise ValueError(f"Missing required timeout keys in execution_config.json: {missing_keys}")

            # Build config dictionary from execution_config.json
            # Transform from execution_config format to internal config format
            config = {
                "USER": exec_config.get("user", "denjo"),
                "DEPLOYMENT_LOCATION": exec_config.get("deployment_location", "/home/denjo/workspace/ktakahashi"),
                "PROJECT_NAME": "WAFL-Testbed",
                "WAFL_DEVICE_P2P_PORT": exec_config.get("host_port_p2p", 10002),
                "CONTAINER_PORT_CTRL": exec_config.get("container_port_ctrl", 10001),
                "WAFL_DEVICE_NAMES": device_names,
                "WAFL_DEVICE_IPS": device_ips,
                "EXPERIMENT_NAME": exec_config.get("experiment_name", "wafl-experiment"),
                "LOG_LEVEL": log_level,
                "TIMEOUTS": timeouts_config,
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

                def create_single_agent(node):
                    """Create a single agent with error handling."""
                    try:
                        device_name = str(node["name"])
                        ip = node["physical_ip"]
                        ctrl_port = node.get("container_port_ctrl", self.config.get("CONTAINER_PORT_CTRL", 10001))

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
                            host_p2p_port=node.get("host_port_p2p", self.config.get("WAFL_DEVICE_P2P_PORT", 10002)),
                            container_p2p_port=node.get("container_port_p2p", self.config.get("WAFL_DEVICE_P2P_PORT", 10002)),  # Ensure this is passed
                            node_config=node,
                            ssh_manager=self.ssh_manager,
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

            # Get log level from config (default to DEBUG)
            log_level_str = self.config.get("LOG_LEVEL", "DEBUG")
            log_level = getattr(logging, log_level_str, logging.DEBUG)

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
            ssp_config: SSP configuration with 'enabled' and 'ssp_threshold' keys

        Note:
            Agent shutdown is always performed in the finally block, even if an exception occurs during the experiment.
        """
        self.logger.info(f"🚀 Experiment started - ID: {self.experiment_id} (SELF: {epochs['self']}, WAFL: {epochs['wafl']} epochs)")
        experiment_success = False

        if ssp_config is None:
            ssp_config = {"enabled": False, "ssp_threshold": 1.0}

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

            # 1. Run SELF phase (always BSP - nodes train independently)
            self.logger.info(f"🏃 Phase [1/4] - SELF training ({epochs['self']} epochs)")
            # SELF phase uses BSP mode - each node completes independently
            # NOTE: No CPU limits during SELF phase for maximum training speed
            self._run_phase(
                "SELF",
                epochs["self"],
                ssp_threshold=1.0,
                ssp_enabled=False,
            )
            self.logger.info("✅ Phase [1/4] - Complete (SELF training finished)")

            # 2. Apply CPU limits before WAFL phase
            self.logger.info("⚙️ Applying CPU limits for WAFL phase...")
            self._apply_cpu_limits_to_all_agents()

            # 2.5. Apply static network conditions if configured
            self._apply_static_network_conditions()

            # 3. Run WAFL phase
            self.logger.info(f"🤝 Phase [3/4] - WAFL training ({epochs['wafl']} epochs)")

            # Extract SSP configuration
            ssp_enabled = ssp_config.get("enabled", False)
            ssp_threshold = ssp_config.get("ssp_threshold", 1.0)
            aggregation = wafl_phase.get("aggregation_strategy", "FedAvg")

            # Log WAFL strategy details
            self.logger.info("📊 WAFL Strategy Configuration:")
            self.logger.info(f"   - Aggregation: {aggregation}")
            if ssp_enabled:
                self.logger.info(f"   - Synchronization: SSP (Threshold={ssp_threshold:.1%})")
            else:
                self.logger.info("   - Synchronization: BSP (Strict Sync - all nodes synchronized per epoch)")
            self.logger.info(f"   - Batch Size: {wafl_phase.get('batch_size', 32)}")
            self.logger.info(f"   - Learning Rate: {wafl_phase.get('learning_rate', 0.001)}")
            self.logger.info(f"   - Coefficiency: {wafl_phase.get('coefficiency', 1.0)}")

            # Record WAFL phase start timestamp for time-to-accuracy analysis
            wafl_phase_start_timestamp = time.time()
            wafl_phase_start_relative = wafl_phase_start_timestamp - self.start_timestamp

            # Load existing metadata (may contain epoch_durations from SELF phase)
            metadata_path = os.path.join(self.results_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            else:
                metadata = {}

            # Update with WAFL phase metadata (preserving epoch_durations if present)
            metadata.update(
                {
                    "wafl_phase_start_timestamp": wafl_phase_start_timestamp,
                    "wafl_phase_start_relative": wafl_phase_start_relative,
                    "experiment_start_timestamp": self.start_timestamp,
                    "self_epochs": epochs["self"],
                    "wafl_epochs": epochs["wafl"],
                }
            )

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            self.logger.info(f"📝 WAFL phase metadata saved to {metadata_path}")

            self._run_phase(
                "WAFL",
                epochs["wafl"],
                ssp_threshold=ssp_threshold,
                ssp_enabled=ssp_enabled,
                epoch_offset=epochs["self"],
            )

            self.logger.info("✅ Phase [3/4] - Complete (WAFL training finished)")
            experiment_success = True

        except KeyboardInterrupt:
            self.logger.warning("⚠️ Experiment interrupted by user")
        except Exception as e:
            self.logger.error(f"💥 Experiment failed: {e}", exc_info=True)
        finally:
            # 4. Shutdown all agents
            self.logger.info("🛑 Phase [4/4] - Shutting down all agents")
            self._shutdown_all_agents()

            status = "SUCCESS" if experiment_success else "FAILED"
            self.logger.info(f"✅ Experiment complete - ID: {self.experiment_id}, Status: {status}")

    def _apply_cpu_limits_to_all_agents(self):
        """
        Apply CPU limits to all agents at the start of WAFL phase.

        This reads cpu_limit from execution_config.json and applies it
        to each running container using docker update.
        """
        # Load execution config to get cpu_limit for each node
        exec_config_path = os.path.join("ctrl", "execution_config.json")
        try:
            with open(exec_config_path, "r") as f:
                exec_config = json.load(f)
        except Exception as e:
            self.logger.error(f"💥 Failed to load execution config: {e}")
            return

        nodes = exec_config.get("nodes", [])
        node_cpu_limits = {str(node["name"]): node.get("cpu_limit") for node in nodes}

        # Create a ContainerManager for applying CPU limits (using shared SSH connections)
        container_manager = ContainerManager(
            user=self.config["USER"],
            deployment_location=self.config["DEPLOYMENT_LOCATION"],
            project_name=self.config["PROJECT_NAME"],
            logger=self.logger,
            ssh_manager=self.ssh_manager,
        )

        applied_count = 0
        for agent in self.agents:
            cpu_limit = node_cpu_limits.get(agent.name)
            if cpu_limit:
                container_name = f"wafl-node-{agent.name}"
                success = container_manager.apply_cpu_limit(
                    ip=agent.ip,
                    container_name=container_name,
                    cpu_limit=cpu_limit,
                )
                if success:
                    applied_count += 1

        self.logger.info(f"✅ CPU limits applied to {applied_count}/{len(self.agents)} agents")

    def _apply_static_network_conditions(self):
        """
        Apply static network conditions from parameters.json to all agent containers.

        This is called before WAFL phase to set up tc rules for delay, loss, and rate limiting.
        The tc rules are applied bidirectionally between all node pairs.
        """
        try:
            with open("ctrl/parameters.json", "r") as f:
                params = json.load(f)
        except Exception as e:
            self.logger.error(f"💥 Failed to load parameters.json: {e}")
            return

        net_cond = params.get("network_condition", {})
        if not net_cond.get("enabled", False):
            self.logger.debug("Static network conditions disabled")
            return

        delay = net_cond.get("delay", "50ms")
        loss = net_cond.get("loss", "0%")
        rate = net_cond.get("rate", "100mbit")

        self.logger.info(f"📡 Applying static network conditions: delay={delay}, loss={loss}, rate={rate}")

        # Load execution config to get peer node IPs
        try:
            with open("ctrl/execution_config.json", "r") as f:
                exec_config = json.load(f)
        except Exception as e:
            self.logger.error(f"💥 Failed to load execution_config.json: {e}")
            return

        # Get list of all peer node IPs for filtering (only apply tc to peer-to-peer traffic)
        peer_ips = [node["physical_ip"] for node in exec_config.get("nodes", [])]
        self.logger.debug(f"📋 Peer IPs for tc filtering: {peer_ips}")

        def apply_tc_to_agent(agent):
            """Apply tc rules to a single agent's container, only for peer-to-peer traffic."""
            container_name = f"wafl-node-{agent.agent_index}"

            # Build tc commands with filters to apply only to traffic destined for peer nodes
            # This ensures control server communication is not affected
            # 1. Clear existing rules
            # 2. Add root htb qdisc with default class 20 (unrestricted)
            # 3. Add restricted class 1:10 for peer traffic
            # 4. Add unrestricted class 1:20 for other traffic (control server, etc.)
            # 5. Add netem qdisc to restricted class
            # 6. Add filters to route peer traffic to restricted class

            # Build filter commands for each peer IP
            filter_cmds_list = []
            for peer_ip in peer_ips:
                # Skip self (same container would have same IP)
                if peer_ip == agent.ip:
                    continue
                filter_cmds_list.append(f"tc filter add dev eth0 protocol ip parent 1:0 prio 1 u32 match ip dst {peer_ip} flowid 1:10")

            # Join filter commands with actual newlines
            filter_cmds = "\n".join(filter_cmds_list)

            tc_script = f"""
set -e
tc qdisc del dev eth0 root 2>/dev/null || true
tc qdisc add dev eth0 root handle 1: htb default 20
tc class add dev eth0 parent 1: classid 1:10 htb rate {rate}
tc class add dev eth0 parent 1: classid 1:20 htb rate 1000mbit
tc qdisc add dev eth0 parent 1:10 handle 10: netem delay {delay} loss {loss}
{filter_cmds}
echo "TC applied successfully with peer filters"
"""

            cmd = f"docker exec {container_name} sh -c '{tc_script}'"

            try:
                ssh = self.ssh_manager.get_connection(agent.ip)

                stdin, stdout, stderr = ssh.exec_command(cmd)
                exit_status = stdout.channel.recv_exit_status()

                if exit_status == 0:
                    return (True, agent.name, None)
                else:
                    error = stderr.read().decode().strip()
                    return (False, agent.name, error)
            except Exception as e:
                return (False, agent.name, str(e))

        # Apply tc rules in parallel for all agents
        applied_count = 0
        failed_agents = []

        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            futures = {executor.submit(apply_tc_to_agent, agent): agent for agent in self.agents}
            for future in as_completed(futures):
                success, name, error = future.result()
                if success:
                    applied_count += 1
                    self.logger.debug(f"✅ TC rules applied to agent {name}")
                else:
                    failed_agents.append(name)
                    if error:
                        self.logger.warning(f"⚠️ Failed to apply TC to agent {name}: {error}")

        if failed_agents:
            self.logger.warning(f"⚠️ TC rules failed for {len(failed_agents)} agents: {failed_agents}")
        else:
            self.logger.info(f"✅ Static network conditions applied to {applied_count}/{len(self.agents)} agents (peer traffic only)")

    def _run_phase(
        self,
        phase_name: str,
        total_epochs: int,
        ssp_threshold: float = 1.0,
        ssp_enabled: bool = False,
        epoch_offset: int = 0,
    ):
        """
        Run a single training phase (SELF or WAFL) with configurable synchronization.

        Synchronization Modes:
        - BSP (ssp_enabled=False): Bulk Synchronous Parallel
          All nodes must complete the current epoch before any node can proceed to the next.
        - SSP (ssp_enabled=True): Semi-Synchronous Parallel
          When 'ssp_threshold' fraction of nodes complete an epoch, slow nodes are force-skipped.
          This ensures no node is ever more than 1 epoch behind.

        Args:
            phase_name: "SELF" or "WAFL"
            total_epochs: Total number of epochs to run
            ssp_threshold: Fraction of agents (0.0-1.0) required before forcing slow agents to skip (SSP only)
            ssp_enabled: Whether SSP mode is enabled
        """
        if ssp_enabled:
            self.logger.info(f"🔄 {phase_name} Phase: SSP Mode (threshold={ssp_threshold:.1%})")
        else:
            self.logger.info(f"🔄 {phase_name} Phase: BSP Mode (strict epoch synchronization)")

        # Initialize agent status
        agent_status = {agent.name: "IDLE" for agent in self.agents}
        epochs_completed = {agent.name: 0 for agent in self.agents}

        # Track retry state for stalled agents
        last_status_success = {agent.name: time.time() for agent in self.agents}
        consecutive_errors = {agent.name: 0 for agent in self.agents}
        MAX_CONSECUTIVE_ERRORS = 5  # Log warning after this many consecutive errors
        STALL_WARNING_INTERVAL = 60.0  # Warn about stalled agents every 60 seconds

        start_time = time.time()
        last_epoch_logged = 0
        last_progress_log = 0
        last_stall_warning = 0
        last_tc_applied_epoch = 0  # Track last epoch where tc was applied

        # Track epoch completion times on the control server side
        epoch_start_times = {0: start_time}  # When each epoch started (epoch 0 = phase start)
        epoch_durations = []  # List of {epoch, phase, duration_ms}

        # Create a single shared thread pool for all parallel operations
        # This avoids the overhead of creating/destroying thread pools in each loop iteration
        max_workers = max(len(self.agents), 1)
        with ThreadPoolExecutor(max_workers=max_workers) as shared_executor:
            while True:
                current_time = time.time()

                # Check epoch progress
                min_epoch = min(epochs_completed.values())
                max_epoch = max(epochs_completed.values())

                # Log epoch progress
                if min_epoch > last_epoch_logged:
                    # Record epoch completion time and duration
                    current_epoch_end_time = time.time()
                    if min_epoch > 0:  # Don't record for epoch 0 (phase start)
                        epoch_start = epoch_start_times.get(min_epoch - 1, start_time)
                        duration_ms = (current_epoch_end_time - epoch_start) * 1000
                        # Use global epoch number (1-indexed, continuous across SELF and WAFL phases)
                        global_epoch = min_epoch + epoch_offset
                        epoch_duration_entry = {
                            "epoch": global_epoch,
                            "phase": phase_name,
                            "duration_ms": duration_ms,
                        }
                        epoch_durations.append(epoch_duration_entry)
                        self.logger.debug(f"{phase_name} Epoch {global_epoch} completed: {duration_ms:.1f}ms (server-side)")
                        # Immediately save to file (incremental save)
                        self._append_ctrl_epoch_duration(epoch_duration_entry)

                    # Record start time for next epoch
                    epoch_start_times[min_epoch] = current_epoch_end_time

                    # Log with 1-indexed epoch number for display
                    display_epoch = min_epoch + epoch_offset
                    if ssp_enabled:
                        self.logger.debug(f"{phase_name} Epoch {display_epoch}: SSP mode (threshold={ssp_threshold:.1%})")
                    else:
                        self.logger.debug(f"{phase_name} Epoch {display_epoch}: All nodes synchronized (BSP)")
                    last_epoch_logged = min_epoch
                if min_epoch >= total_epochs:
                    break

                # ========== SSP Mode: Threshold-based force skip ==========
                # When ssp_threshold fraction of nodes complete an epoch, force remaining nodes
                # to skip to that epoch. This ensures no node is more than 1 epoch behind.
                if ssp_enabled and ssp_threshold < 1.0:
                    # Find the highest epoch where threshold is met
                    # Count agents who have completed each epoch level or higher
                    target_epoch = -1
                    for check_epoch in range(max_epoch, min_epoch, -1):
                        # Count agents who have completed check_epoch OR HIGHER
                        count = sum(1 for ae in epochs_completed.values() if ae >= check_epoch)

                        if count >= len(self.agents) * ssp_threshold:
                            target_epoch = check_epoch
                            break

                    if target_epoch != -1 and target_epoch > min_epoch:
                        # Identify slow agents (those who haven't completed target_epoch)
                        slow_agents = [name for name, e in epochs_completed.items() if e < target_epoch]

                        if slow_agents:
                            self.logger.info(f"⚡ SSP Threshold ({ssp_threshold:.0%}) reached for epoch {target_epoch}. Forcing {len(slow_agents)} slow agents to skip.")
                            self.logger.debug(f"   Slow agents: {slow_agents}")

                            # Force slow agents to skip to target_epoch (PARALLEL using shared executor)
                            def force_skip_agent(agent):
                                """Send FORCE_NEXT to a slow agent."""
                                if epochs_completed[agent.name] >= target_epoch:
                                    return (agent.name, False, None)

                                self.logger.warning(f"⏩ Forcing agent {agent.name} (epoch {epochs_completed[agent.name]}) to skip to {target_epoch}")

                                result = None
                                if agent_status[agent.name] == "RUNNING":
                                    success, response = agent._send_command("FORCE_NEXT\r\n")
                                    result = (agent.name, True, success)
                                else:
                                    result = (agent.name, True, None)
                                return result

                            slow_agent_objs = [a for a in self.agents if epochs_completed[a.name] < target_epoch]

                            # Use shared executor instead of creating new one
                            futures = {shared_executor.submit(force_skip_agent, agent): agent for agent in slow_agent_objs}
                            for future in as_completed(futures):
                                agent_name, was_slow, cmd_success = future.result()
                                if was_slow:
                                    if cmd_success:
                                        self.logger.debug(f"   FORCE_NEXT sent to {agent_name}")
                                    agent_status[agent_name] = "IDLE"
                                    epochs_completed[agent_name] = target_epoch

                # Apply dynamic network conditions if mobility-aware mode is enabled (once per epoch)
                if self.mobility_aware_config and phase_name == "WAFL":
                    if min_epoch > last_tc_applied_epoch:
                        self._apply_dynamic_network_conditions(min_epoch)
                        last_tc_applied_epoch = min_epoch

                # ========== Schedule agents based on synchronization mode (PARALLEL) ==========
                def begin_epoch_for_agent(agent):
                    """Start epoch for an agent if conditions are met."""
                    current_epoch = epochs_completed[agent.name]
                    if current_epoch < total_epochs and agent_status[agent.name] == "IDLE":
                        next_epoch = current_epoch + 1
                        can_proceed = min_epoch >= current_epoch
                        if can_proceed:
                            success = agent.begin_epoch(phase_name, next_epoch + epoch_offset)
                            return (agent.name, success, next_epoch)
                    return (agent.name, None, None)  # No action needed

                # Filter agents that might need to start
                agents_to_schedule = [a for a in self.agents if epochs_completed[a.name] < total_epochs and agent_status[a.name] == "IDLE"]

                if agents_to_schedule:
                    # Use shared executor instead of creating new one
                    futures = {shared_executor.submit(begin_epoch_for_agent, agent): agent for agent in agents_to_schedule}
                    for future in as_completed(futures):
                        agent_name, success, next_epoch = future.result()
                        if success is not None:
                            if success:
                                agent_status[agent_name] = "RUNNING"
                                last_status_success[agent_name] = current_time
                                consecutive_errors[agent_name] = 0
                            else:
                                self.logger.warning(f"Failed to start epoch {next_epoch} on agent {agent_name}, retrying...")

                # ========== Poll agent status (PARALLEL) ==========
                # Poll RUNNING agents, and also retry agents that haven't responded recently
                def poll_agent_status(agent):
                    """Poll status for a single agent."""
                    if agent_status[agent.name] != "RUNNING":
                        return (agent.name, None, None, [], False)
                    status, logs = agent.get_status()
                    # Check if this was a communication error
                    is_comm_error = status is not None and status.startswith("ERROR_COMM")
                    return (agent.name, status, logs, [], is_comm_error)

                # Include agents that are RUNNING
                running_agents = [a for a in self.agents if agent_status[a.name] == "RUNNING"]

                if running_agents:
                    # Use shared executor instead of creating new one
                    futures = {shared_executor.submit(poll_agent_status, agent): agent for agent in running_agents}
                    for future in as_completed(futures):
                        agent_name, status, logs, _, is_comm_error = future.result()
                        if status is None:
                            continue

                        if logs:
                            for log_line in logs:
                                if log_line.strip():
                                    self.logger.debug(f"[{agent_name}] {log_line}")

                        # Handle communication errors - keep agent as RUNNING and retry later
                        if is_comm_error:
                            consecutive_errors[agent_name] += 1
                            time_since_success = current_time - last_status_success[agent_name]

                            if consecutive_errors[agent_name] >= MAX_CONSECUTIVE_ERRORS:
                                self.logger.warning(f"⚠️ Agent {agent_name}: {consecutive_errors[agent_name]} consecutive comm errors (last success {time_since_success:.1f}s ago), will retry...")
                            else:
                                self.logger.debug(f"Agent {agent_name}: comm error #{consecutive_errors[agent_name]}, will retry")
                            # Keep agent in RUNNING state to retry on next poll
                            continue

                        # Check for application-level errors (not comm errors)
                        if status.startswith("ERROR") and not is_comm_error:
                            self.logger.error(f"❌ Agent {agent_name} encountered error: {status}")
                            agent_status[agent_name] = "ERROR"
                            continue

                        # Successful status response - reset error counters
                        last_status_success[agent_name] = current_time
                        consecutive_errors[agent_name] = 0

                        if status.startswith("DONE"):
                            try:
                                parts = status.split("-")
                                if len(parts) >= 3:
                                    done_epoch_global = int(parts[2])
                                    done_epoch_local = done_epoch_global - epoch_offset
                                    if done_epoch_local > epochs_completed[agent_name]:
                                        epochs_completed[agent_name] = done_epoch_local
                                        agent_status[agent_name] = "IDLE"
                                        self.logger.info(f"✅ Agent {agent_name} completed {phase_name} epoch {done_epoch_global}")
                            except Exception as e:
                                self.logger.error(f"Error parsing status {status}: {e}")

                # ========== Warn about stalled agents periodically ==========
                elapsed_time = current_time - start_time
                if elapsed_time - last_stall_warning >= STALL_WARNING_INTERVAL:
                    stalled_agents = []
                    for agent in self.agents:
                        if agent_status[agent.name] == "RUNNING":
                            time_since_success = current_time - last_status_success[agent.name]
                            if time_since_success > STALL_WARNING_INTERVAL:
                                stalled_agents.append(f"{agent.name} (no response for {time_since_success:.1f}s, epoch {epochs_completed[agent.name]})")

                    if stalled_agents:
                        self.logger.warning(f"⏳ Agents not responding (will keep retrying): {', '.join(stalled_agents)}")
                    last_stall_warning = elapsed_time

                # Progress logging every 30 seconds
                if elapsed_time - last_progress_log >= 30:
                    # Count agents by status for detailed progress
                    running_count = sum(1 for s in agent_status.values() if s == "RUNNING")
                    idle_count = sum(1 for s in agent_status.values() if s == "IDLE")
                    error_count = sum(1 for s in agent_status.values() if s == "ERROR")

                    # Display with 1-indexed epoch number (continuous across phases)
                    display_epoch = min_epoch + epoch_offset
                    if ssp_enabled:
                        self.logger.info(f"📊 {phase_name} Progress: Epoch {display_epoch}/{total_epochs + epoch_offset} (SSP threshold={ssp_threshold:.1%}, running={running_count}, idle={idle_count}, error={error_count})")
                    else:
                        self.logger.info(f"📊 {phase_name} Progress: Epoch {display_epoch}/{total_epochs + epoch_offset} (BSP sync, running={running_count}, idle={idle_count}, error={error_count})")

                    # Verification-friendly snapshot (compact, with samples)
                    running_sample = [f"{a.name}@{epochs_completed[a.name] + 1}" for a in self.agents if agent_status[a.name] == "RUNNING"][:5]
                    idle_sample = [f"{a.name}@{epochs_completed[a.name]}" for a in self.agents if agent_status[a.name] == "IDLE"][:5]
                    error_sample = [a.name for a in self.agents if agent_status[a.name] == "ERROR"][:5]
                    self.logger.info(f"📍 Snapshot: min_epoch={min_epoch}, max_epoch={max_epoch}, RUNNING={running_count} (sample={running_sample}), IDLE={idle_count} (sample={idle_sample}), ERROR={error_count} (sample={error_sample})")
                    last_progress_log = elapsed_time

                time.sleep(0.5)

        # Note: epoch durations are saved incrementally via _append_ctrl_epoch_duration
        # No need to save again here

    def _append_ctrl_epoch_duration(self, epoch_duration_entry: dict):
        """
        Append a single epoch duration entry to the metadata file immediately.

        This ensures data is saved incrementally, even if the experiment is interrupted.

        Args:
            epoch_duration_entry: Dict with {epoch, phase, duration_ms}
        """
        # Save to results_dir (same directory as output.log) as metadata.json (unified format)
        metadata_path = os.path.join(self.results_dir, "metadata.json")
        try:
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                # Ensure epoch_durations key exists (may have been created without it)
                if "epoch_durations" not in metadata:
                    metadata["epoch_durations"] = []
            else:
                metadata = {"epoch_durations": []}

            # Append single entry
            metadata["epoch_durations"].append(epoch_duration_entry)

            # Save updated metadata
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"💥 Failed to append epoch duration: {e}")

    def _shutdown_all_agents(self):
        """Shutdown all agents gracefully by stopping Docker containers in parallel."""
        if not self.agents:
            self.logger.warning("No agents to shutdown")
            return

        self.logger.warning(f"🛑 Shutting down {len(self.agents)} agents")

        def shutdown_single_agent(agent):
            """Shutdown a single agent."""
            if agent is None:
                return (False, None, "Agent is None")

            try:
                # Send KILL command to container
                self.logger.debug(f"Sending KILL to agent {agent.name}")
                agent.send_kill_command()

                # Stop Docker container via SSH using connection manager
                container_name = f"wafl-node-{agent.name}"

                ssh = self.ssh_manager.get_connection(agent.ip)

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

        # Close all SSH connections after shutdown
        self.ssh_manager.close_all()

    def _apply_dynamic_network_conditions(self, epoch: int):
        """
        Apply dynamic network conditions for the given epoch using tc.

        This method reads network conditions from self.network_conditions and
        self.path_loss_model, then applies tc rules directly inside each container
        using SSH + docker exec (same pattern as static network conditions).

        Bidirectional consideration:
        - For node A, we need to apply tc rules for traffic TO all peers it communicates with
        - This includes both nodes A fetches from AND nodes that fetch from A
        """
        if not self.network_conditions or not self.path_loss_model:
            return

        # Check if epoch is within bounds (array-based format)
        if epoch >= len(self.network_conditions):
            self.logger.debug(f"No network conditions for epoch {epoch} (only {len(self.network_conditions)} epochs available)")
            return

        self.logger.info(f"📡 Applying dynamic network conditions for epoch {epoch}")

        # Get rank definitions from path loss model
        rank_definitions = self.path_loss_model.get("ranks", [])
        if not rank_definitions:
            self.logger.error("💥 No rank definitions found in path loss model")
            return

        # Build rank name to parameters mapping
        rank_params = {}
        for i, rank in enumerate(rank_definitions, start=1):
            rank_params[rank["name"]] = {
                "classid": f"1:{i}",
                "handle": i * 10,
                "rate": rank.get("rate", "100mbit"),
                "delay": rank.get("delay", "10ms"),
                "loss": rank.get("loss", "0%"),
            }

        # Get epoch conditions
        epoch_conditions = self.network_conditions[epoch]

        # Load execution config for IP mapping
        try:
            with open("ctrl/execution_config.json", "r") as f:
                exec_config = json.load(f)
        except Exception as e:
            self.logger.error(f"💥 Failed to load execution_config.json: {e}")
            return

        # Build node ID to IP mapping
        node_to_ip = {}
        for node in exec_config.get("nodes", []):
            node_id = int(node["name"])
            node_to_ip[node_id] = node.get("physical_ip")

        def apply_tc_for_agent(agent):
            """Apply tc rules for a single agent with bidirectional peer consideration."""
            container_name = f"wafl-node-{agent.agent_index}"
            node_id = str(agent.agent_index)
            node_id_int = int(node_id)

            # Collect all peers this node communicates with (bidirectional)
            peers_for_tc = {}

            # Direction 1: Peers this node fetches FROM (from network_conditions)
            if node_id in epoch_conditions:
                for peer_info in epoch_conditions[node_id]:
                    peer_id = peer_info.get("peer")
                    if peer_id is None:
                        continue
                    peer_ip = node_to_ip.get(peer_id)
                    if not peer_ip:
                        continue
                    rank_name = peer_info.get("rank", "Excellent")
                    if rank_name in rank_params:
                        peers_for_tc[peer_ip] = rank_name

            # Direction 2: Nodes that fetch FROM this node (dispatch direction)
            # For each other node, check if this node is in their peer list
            for other_node_key, other_peers in epoch_conditions.items():
                if other_node_key == node_id:
                    continue
                if not isinstance(other_peers, list):
                    continue
                for peer_info in other_peers:
                    if peer_info.get("peer") == node_id_int:
                        other_node_int = int(other_node_key)
                        other_node_ip = node_to_ip.get(other_node_int)
                        if other_node_ip and other_node_ip not in peers_for_tc:
                            rank_name = peer_info.get("rank", "Excellent")
                            if rank_name in rank_params:
                                peers_for_tc[other_node_ip] = rank_name

            # Build tc script
            tc_lines = [
                "set -e",
                "tc qdisc del dev eth0 root 2>/dev/null || true",
                "tc qdisc add dev eth0 root handle 1: htb default 99",
                "tc class add dev eth0 parent 1: classid 1:99 htb rate 1000mbit",
            ]

            # Add classes and netem for each rank
            for rank_name, params in rank_params.items():
                tc_lines.append(f"tc class add dev eth0 parent 1: classid {params['classid']} htb rate {params['rate']}")
                tc_lines.append(f"tc qdisc add dev eth0 parent {params['classid']} handle {params['handle']}: netem delay {params['delay']} loss {params['loss']}")

            # Add filters for each peer
            for peer_ip, rank_name in peers_for_tc.items():
                if rank_name in rank_params:
                    classid = rank_params[rank_name]["classid"]
                    tc_lines.append(f"tc filter add dev eth0 protocol ip parent 1:0 prio 1 u32 match ip dst {peer_ip} flowid {classid}")

            tc_lines.append('echo "Dynamic TC applied successfully"')
            tc_script = "\n".join(tc_lines)

            cmd = f"docker exec {container_name} sh -c '{tc_script}'"

            try:
                ssh = self.ssh_manager.get_connection(agent.ip)

                stdin, stdout, stderr = ssh.exec_command(cmd)
                exit_status = stdout.channel.recv_exit_status()

                if exit_status == 0:
                    return (True, agent.name, len(peers_for_tc))
                else:
                    error = stderr.read().decode().strip()
                    return (False, agent.name, error)
            except Exception as e:
                return (False, agent.name, str(e))

        # Apply tc rules in parallel for all agents
        applied_count = 0
        failed_agents = []

        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            futures = {executor.submit(apply_tc_for_agent, agent): agent for agent in self.agents}
            for future in as_completed(futures):
                result = future.result()
                if result[0]:  # success
                    applied_count += 1
                    self.logger.debug(f"✅ Applied tc rules for agent {result[1]} (epoch {epoch}, {result[2]} peers)")
                else:
                    failed_agents.append(result[1])
                    self.logger.warning(f"⚠️ Failed to apply tc rules for agent {result[1]}: {result[2]}")

        if failed_agents:
            self.logger.warning(f"⚠️ Dynamic TC rules failed for {len(failed_agents)} agents")
        else:
            self.logger.info(f"✅ Dynamic network conditions applied to {applied_count}/{len(self.agents)} agents (epoch {epoch})")


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

        # Initialize defaults
        ssp_settings = experiment_parameters.get("ssp", {})
        udp_settings = {}
        tcp_settings = {}
        rudp_settings = {}
        comp_settings = {}

        tcp_enabled = True
        udp_enabled = False
        rudp_enabled = False
        comp_enabled = False
        rudp_mode = "rudp"

        # Parse method configuration
        if isinstance(method, str):
            method_str = method.lower()
            print(f"   - Configuration Mode: Simple ({method})")

            # Set enable flags based on method string
            if method_str == "dynamic":
                tcp_enabled = True
                udp_enabled = True  # Dynamic uses both
            elif method_str == "udp":
                tcp_enabled = False
                udp_enabled = True
            elif method_str == "rudp":
                tcp_enabled = False
                rudp_enabled = True
            elif method_str == "tcp":
                tcp_enabled = True

            # Simple schema assumes defaults for sub-settings
            if udp_enabled:
                udp_settings = {"fec_m": "auto"}
            if rudp_enabled:
                rudp_settings = {"mode": "rudp", "max_retries": 20, "window_size": 16}

        elif isinstance(method, dict):
            # 新形式: {"base": "udp", "fec": true, "compression": false, "nack": true}
            if "base" in method:
                base = method.get("base", "tcp")
                fec_enabled = method.get("fec", True)
                comp_enabled = method.get("compression", False)
                nack_enabled = method.get("nack", True)

                tcp_enabled = base == "tcp"
                udp_enabled = base == "udp"
                rudp_enabled = False

                print(f"   - Configuration Mode: Ablation (base={base})")
                print(f"   - FEC: {'Enabled' if fec_enabled else 'Disabled'}")
                print(f"   - Compression: {'Enabled' if comp_enabled else 'Disabled'}")
                print(f"   - NACK: {'Enabled' if nack_enabled else 'Disabled'}")
            else:
                raise ValueError('Unsupported method format. Use: {"base": "udp", "fec": true, "compression": false, "nack": true}')

        # SSP (Common logic)
        ssp_enabled = ssp_settings.get("enabled", False)
        print(f"   - SSP: {'Enabled' if ssp_enabled else 'Disabled'}")
        if ssp_enabled:
            print(f"     • SSP Threshold: {ssp_settings.get('ssp_threshold', 1.0)}")

        # UDP
        print(f"   - UDP: {'Enabled' if udp_enabled else 'Disabled'}")
        if udp_enabled:
            print(f"     • FEC M: {udp_settings.get('fec_m', 'auto')}")

        # RUDP / E-RUDP
        if rudp_enabled:
            mode_display = "E-RUDP" if rudp_mode == "erudp" else "RUDP"
            print(f"   - {mode_display}: Enabled")
            print(f"     • Window Size: {rudp_settings.get('window_size', 16)}")
            print(f"     • Max Retries: {rudp_settings.get('max_retries', 10)}")
            if rudp_mode == "erudp":
                print(f"     • Aging Limit: {rudp_settings.get('aging_limit', 0.5) * 1000:.0f}ms")
        else:
            print("   - RUDP: Disabled")

        # TCP Logic
        # RUDP takes precedence over TCP if strictly one protocol allowed,
        # but for logging we just show status.
        # Dynamic mode enables both TCP and UDP.

        print(f"   - TCP: {'Enabled' if tcp_enabled else 'Disabled'}")

        # Protocol validation / Display Active
        active_protocols = []
        if tcp_enabled:
            active_protocols.append("TCP")
        if udp_enabled:
            active_protocols.append("UDP")
        if rudp_enabled:
            active_protocols.append("RUDP")

        if isinstance(method, str) and method == "dynamic":
            print("\n✅ Active Transport Protocol: DYNAMIC (TCP/UDP switching)")
        elif len(active_protocols) == 1:
            print(f"\n✅ Active Transport Protocol: {active_protocols[0]}")
        elif len(active_protocols) > 1:
            # This happens in dynamic mode or misconfig (legacy)
            # If dynamic, handled above.
            print(f"\n✅ Active Transport Protocols: {', '.join(active_protocols)}")
        else:
            print("\n⚠️  WARNING: No transport protocol enabled!")

        # Compression
        print(f"   - Compression: {'Enabled' if comp_enabled else 'Disabled'}")
        if comp_enabled:
            print(f"     • Initial Method: {comp_settings.get('initial_method', 'zlib')}")

        # Create ControlServer instance
        controller = ControlServer()

        # Run experiment
        # Extract SSP settings - include 'enabled' flag for proper BSP/SSP mode selection
        ssp_threshold = ssp_settings.get("ssp_threshold", 1.0) if ssp_enabled else 1.0

        controller.run_experiment(
            epochs={"self": epochs_self, "wafl": epochs_wafl},
            wafl_phase=experiment_parameters["wafl_phase"],
            contact_pattern=experiment_parameters["contact_pattern"],
            ssp_config={
                "enabled": ssp_enabled,
                "ssp_threshold": ssp_threshold,
            },
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

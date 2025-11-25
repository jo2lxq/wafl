import datetime
import json
import logging
import os
import socket
import time
from typing import Any, Dict, List, Tuple

import paramiko


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
        timeout: int = 10,
        container_ctrl_port: int = None,
        host_p2p_port: int = None,
        container_p2p_port: int = None,
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

        # Deploy configurations during initialization
        self._deploy_configurations(experiment_parameters)

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
                "experiment_id": experiment_parameters.get("experiment_id"),
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
        self.logger.info(f"📋 Deploying configurations to agent {self.name}")

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
                files_to_deploy.append({"content": config_json, "filename": "config.json", "description": "agent configuration"})

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

                    deployed_files = []
                    for file_info in files_to_deploy:
                        try:
                            file_path = os.path.join(config_dir, file_info["filename"])
                            file_obj = io.BytesIO(file_info["content"].encode("utf-8"))

                            sftp.putfo(file_obj, file_path)
                            sftp.chmod(file_path, 0o644)

                            deployed_files.append(file_info["filename"])
                            self.logger.info(f"📋 Deployed {file_info['description']} to agent {self.name}")

                        except Exception as e:
                            self.logger.error(f"💥 Failed to deploy {file_info['description']}: {e}")
                            return False

                # Verify all deployed files
                verification_success = True
                for filename in deployed_files:
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

                self.logger.info(f"✅ All configurations deployed successfully to agent {self.name} ({len(deployed_files)} files)")
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
            self.logger.error(f"💥 Configuration deployment failed for agent {self.name}: {e}", exc_info=True)
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

    def start_remote_process(self, experiment_id: str) -> bool:
        """
        Start wafl/src/main.py with nohup via SSH on execution server.

        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.info(f"🚀 Verifying remote process for experiment '{experiment_id}' on {self.ip}")

        # Since we are using Docker, the process is already started by start_experiment.py
        # We just need to wait for the port to be ready.

        max_retries = 30
        retry_interval = 2

        for i in range(max_retries):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(2)
                    result = s.connect_ex((self.ip, self.container_ctrl_port))
                    if result == 0:
                        self.status = "READY"
                        self.logger.info(f"✅ Remote agent {self.name} is ready at {self.ip}:{self.container_ctrl_port}")
                        return True
            except Exception:
                pass

            self.logger.debug(f"⏳ Waiting for agent {self.name} to be ready ({i + 1}/{max_retries})...")
            time.sleep(retry_interval)

        self.logger.error(f"❌ Timed out waiting for agent {self.name} to be ready")
        self.status = "ERROR"
        return False

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
            self.logger.error(f"💥 Error parsing status response from agent {self.name}: {e}", exc_info=True)
            self.status = "ERROR"
            return "ERROR_PARSE", [f"Parse error: {e}"]

    def begin_epoch(self, phase: str, epoch: int) -> bool:
        """
        Send BEGIN command to start training epoch.

        Returns:
            bool: True if successful, False otherwise
        """
        command = f"BEGIN-{phase}-{epoch:05d}"

        self.logger.info(f"🎯 Starting epoch {epoch} for agent {self.name} (phase: {phase})")

        success, response = self._send_command(f"{command}\r\n")
        if not success:
            self.logger.error(f"❌ Failed to send BEGIN command to agent {self.name}: {response}")
            return False

        if response != "OK":
            self.logger.error(f"❌ Agent {self.name} rejected BEGIN command. Response: {response}")
            return False

        self.status = "RUNNING"
        self.logger.info(f"✅ Agent {self.name} accepted epoch {epoch} start command")
        return True

    def begin_evaluation(self, eval_name: str = "eval") -> bool:
        """
        Start evaluation routine with BEGIN-eval command.

        Returns:
            bool: True if successful, False otherwise
        """
        command = f"BEGIN-{eval_name}"
        self.logger.info(f"📊 Starting evaluation '{eval_name}' on agent {self.name}")

        success, response = self._send_command(f"{command}\r\n")
        if not success:
            self.logger.error(f"❌ Failed to send evaluation command to agent {self.name}: {response}")
            return False

        if response != "OK":
            self.logger.error(f"❌ Agent {self.name} rejected evaluation command. Response: {response}")
            return False

        self.logger.info(f"✅ Agent {self.name} started evaluation '{eval_name}'")
        return True

    def send_kill_command(self) -> bool:
        """
        Send KILL command to terminate process normally.
        """
        self.logger.warning(f"🛑 Sending graceful shutdown command to agent {self.name}")

        success, response = self._send_command("KILL\r\n")
        if success and response == "OK":
            self.logger.info(f"✅ Agent {self.name} acknowledged shutdown command")
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
                    self.logger.info(f"✅ Force kill successful for agent {self.name}")
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
        self._setup_logging()

    def _load_config(self) -> Dict[str, Any]:
        """Load config file (execution_config.json)."""
        config_path = os.path.join("ctrl", "execution_config.json")
        self.logger.info(f"📝 Loading configuration from {config_path}")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"🚫 Config file not found: {config_path}")

        try:
            with open(config_path, "r") as f:
                config = json.load(f)

            # Add some default values expected by other parts of the code
            config["EXPERIMENT_NAME"] = "wafl-experiment"

            self.logger.info("✅ Configuration loaded successfully from JSON")
            return config

        except Exception as e:
            self.logger.error(f"💥 Failed to load configuration: {e}", exc_info=True)
            raise

    def _generate_experiment_id(self, name: str) -> str:
        """Generate experiment ID in 'experiment-name-timestamp' format."""
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        experiment_id = f"{name}-{timestamp}"
        self.logger.info(f"🆔 Generated experiment ID: {experiment_id}")
        return experiment_id

    def _create_results_directory(self):
        """Create directory to save experiment results."""
        results_path = os.path.join("results", self.experiment_id, "summary")
        try:
            os.makedirs(results_path, exist_ok=True)
            self.logger.info(f"📁 Created results directory: {results_path}")
            return results_path
        except Exception as e:
            self.logger.error(f"💥 Failed to create results directory {results_path}: {e}")
            raise

    def _create_agents(self, experiment_parameters: Dict[str, Any]) -> List[WaflAgent]:
        """Create WaflAgent instance list based on config and experiment parameters."""

        # Check for execution_config.json
        json_config_path = os.path.join("ctrl", "execution_config.json")
        if os.path.exists(json_config_path):
            self.logger.info(f"📄 Loading topology from {json_config_path}")
            try:
                with open(json_config_path, "r") as f:
                    topology = json.load(f)

                nodes = topology.get("nodes", [])
                experiment_parameters["experiment_id"] = self.experiment_id
                experiment_parameters["results_dir"] = self.results_dir

                agents = []
                failed_agents = []
                for node in nodes:
                    try:
                        agent_index = node["id"]
                        device_name = str(agent_index)
                        ip = node["physical_ip"]
                        ctrl_port = node["host_port_ctrl"]
                        container_ctrl_port = node["container_port_ctrl"]
                        host_p2p_port = node["host_port_p2p"]
                        container_p2p_port = 10002  # Default or from JSON if added

                        agent = WaflAgent(
                            agent_index=agent_index,
                            device_name=device_name,
                            ip_address=ip,
                            ctrl_port=ctrl_port,
                            config=self.config,
                            experiment_parameters=experiment_parameters,
                            container_ctrl_port=container_ctrl_port,
                            host_p2p_port=host_p2p_port,
                            container_p2p_port=container_p2p_port,
                        )
                        agents.append(agent)
                        self.logger.info(f"🤖 Created agent '{device_name}' for {ip}:{ctrl_port}")
                    except Exception as e:
                        self.logger.error(f"💥 Failed to create agent '{node.get('id')}': {e}")
                        failed_agents.append(str(node.get("id")))

                if failed_agents:
                    raise RuntimeError(f"❌ Failed to create agents: {', '.join(failed_agents)}")

                self.logger.info(f"✅ Created {len(agents)} agents from JSON topology")
                return agents
            except Exception as e:
                self.logger.error(f"💥 Failed to load topology from JSON: {e}")
                raise

                self.logger.info(f"✅ Created {len(agents)} agents from JSON topology")
                return agents
            except Exception as e:
                self.logger.error(f"💥 Failed to load topology from JSON: {e}")
                raise

    def _setup_logging(self):
        """Setup experiment logging to file and console."""
        try:
            log_file = os.path.join(self.results_dir, "ctrl_output.log")
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            # Clear any existing handlers to avoid duplicate logs
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)

            log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
            level = getattr(logging, log_level, logging.INFO)
            logging.basicConfig(
                level=level,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
                force=True,
            )

            self.logger.info(f"📊 Logging configured. File: {log_file}")
            self.logger.info(f"🎯 Experiment ID: {self.experiment_id}")

        except Exception as e:
            print(f"💥 Failed to setup logging: {e}")
            raise

    def run_experiment(self, epochs: Dict[str, int], wafl_phase: Dict[str, Any], contact_pattern: str, ssp_config: Dict[str, Any] = None):
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
        self.logger.info(f"🚀 Starting experiment: {self.experiment_id} (SELF epochs: {epochs['self']}, WAFL epochs: {epochs['wafl']})")
        experiment_success = False

        # Set default SSP config if not provided
        if ssp_config is None:
            ssp_config = {"staleness": 0, "ssp_threshold": 1.0}

        try:
            # 0. Create agents with unified configuration deployment
            self.logger.info("📋 Phase 0: Creating agents and deploying configurations")
            experiment_parameters = {
                "epochs": epochs,
                "wafl_phase": wafl_phase,
                "contact_pattern": contact_pattern,
            }

            self.agents = self._create_agents(experiment_parameters)
            self.logger.info("✅ All agents created and configured successfully")

            # 1. Run SELF phase
            self.logger.info(f"🏃 Phase 1: Starting SELF phase ({epochs['self']} epochs)")
            # SELF phase is independent, so staleness is effectively infinite or irrelevant.
            # We use a large staleness to allow free running.
            self._run_phase("SELF", epochs["self"], staleness=999999)
            self.logger.info("🎉 All SELF training epochs completed successfully")

            # 2. Run WAFL phase
            self.logger.info(f"🤝 Phase 2: Starting WAFL phase ({epochs['wafl']} epochs)")

            staleness = ssp_config.get("staleness", 0)
            ssp_threshold = ssp_config.get("ssp_threshold", 1.0)

            self.logger.info(f"⚙️  Synchronization: SSP (Staleness: {staleness}, Threshold: {ssp_threshold})")

            self._run_phase("WAFL", epochs["wafl"], staleness=staleness, ssp_threshold=ssp_threshold)

            self.logger.info("🎉 All WAFL training epochs completed successfully")
            experiment_success = True

        except KeyboardInterrupt:
            self.logger.warning("⚠️ Experiment interrupted by user")
        except Exception as e:
            self.logger.error(f"💥 Experiment failed: {e}", exc_info=True)
        finally:
            # 3. Shutdown all agents
            self.logger.info("🛑 Phase 4: Shutting down all agents")
            self._shutdown_all_agents()

            status = "SUCCESS" if experiment_success else "FAILED"
            self.logger.info(f"🏁 Experiment {self.experiment_id} finished with status: {status}")

    def _run_phase(self, phase_name: str, total_epochs: int, staleness: int, ssp_threshold: float = 1.0):
        """
        Runs a training phase (SELF or WAFL) with SSP synchronization.
        ssp_threshold: Fraction of agents (0.0-1.0) required to complete an epoch before forcing others to skip.
        """
        agent_epochs = {agent.name: 0 for agent in self.agents}
        agent_status = {agent.name: "IDLE" for agent in self.agents}  # IDLE, RUNNING

        # Track completion count per epoch
        # We need to know how many agents have completed epoch X

        start_time = time.time()
        last_progress_log = 0

        while True:
            min_epoch = min(agent_epochs.values())

            # Check completion
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
                for e in agent_epochs.values():
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
                        for ae in agent_epochs.values():
                            if ae >= e:
                                count += 1

                        if count >= len(self.agents) * ssp_threshold:
                            target_epoch = e
                            break

                if target_epoch != -1:
                    # Force everyone < target_epoch to skip to target_epoch
                    self.logger.info(f"⚡ SSP Threshold reached for epoch {target_epoch}. Forcing slow agents to skip.")
                    for agent in self.agents:
                        if agent_epochs[agent.name] < target_epoch:
                            self.logger.warning(f"⏩ Forcing agent {agent.name} (epoch {agent_epochs[agent.name]}) to skip to {target_epoch}")
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
                            agent_epochs[agent.name] = target_epoch

            # Schedule agents
            for agent in self.agents:
                current_epoch = agent_epochs[agent.name]

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
                                self.logger.info(f"[{agent.name}] {log_line}")

                    if "ERROR" in status:
                        self.logger.error(f"Agent {agent.name} reported error: {status}")

                    if status.startswith("DONE"):
                        try:
                            parts = status.split("-")
                            if len(parts) >= 3:
                                done_epoch = int(parts[2])
                                if done_epoch > agent_epochs[agent.name]:
                                    agent_epochs[agent.name] = done_epoch
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
        """Terminates all agent processes, trying gracefully first, then forcefully."""
        self.logger.warning(f"🛑 Shutting down {len(self.agents)} agents")

        graceful_success = []
        force_kill_needed = []

        # Try graceful shutdown first
        for agent in self.agents:
            if agent.pid is None:
                self.logger.info(f"⏭️ Skipping agent {agent.name} (never started)")
                continue

            if agent.send_kill_command():
                graceful_success.append(agent.name)
            else:
                force_kill_needed.append(agent)

        if graceful_success:
            self.logger.info(f"✅ Graceful shutdown successful for: {', '.join(graceful_success)}")

        # Force kill remaining agents
        if force_kill_needed:
            self.logger.warning(f"💀 Force killing agents: {[a.name for a in force_kill_needed]}")

            for agent in force_kill_needed:
                try:
                    agent.force_kill_process(self.config)
                    self.logger.info(f"✅ Force kill successful for agent {agent.name}")
                except Exception as e:
                    self.logger.error(f"💥 Force kill failed for agent {agent.name}: {e}")

        self.logger.info("🏁 Agent shutdown process completed")


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
        print(f"📋 WAFL parameters: {experiment_parameters['wafl_phase']}")

        # Create ControlServer instance
        controller = ControlServer()

        # Run experiment
        # Extract SSP settings
        ssp_settings = experiment_parameters.get("method", {}).get("ssp", {})
        ssp_enabled = ssp_settings.get("enabled", True)
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

import datetime
import json
import logging
import os
import socket
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

import paramiko


class WaflAgent:
    """
    Represents each execution server (WAFL device) and manages communication.

    Attributes:
        name (str): Device name (e.g., "0", "1")
        ip (str): IP address
        ctrl_port (int): Control TCP port number
        status (str): Current status ("UNKNOWN", "READY", "RUNNING", "DONE", "ERROR", "TERMINATED")
        pid (str): Process ID of the remote wafl/main.py script.
    """

    def __init__(
        self,
        device_name: str,
        ip_address: str,
        ctrl_port: int,
        config: Dict[str, Any],
        experiment_parameters: Dict[str, Any],
        timeout: int = 10,
    ):
        self.name = device_name
        self.ip = ip_address
        self.ctrl_port = ctrl_port
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
                "device_name": self.name,
                "ip_address": self.ip,
            },
            "experiment_info": {
                "experiment_id": experiment_parameters.get("experiment_id"),
                "experiment_name": self.config.get("EXPERIMENT_NAME", "exp"),
            },
            "infrastructure": {
                "device_names": self.config["WAFL_DEVICE_NAMES"],
                "device_ips": self.config["WAFL_DEVICE_IPS"],
                "ctrl_port": self.config["WAFL_DEVICE_CTRL_PORT"],
                "p2p_port": self.config["WAFL_DEVICE_P2P_PORT"],
            },
            "experiment_parameters": {
                "epochs": experiment_parameters.get("epochs"),
                "wafl_phase_params": experiment_parameters.get("wafl_phase_params", {}),
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
            target_path = os.path.join(self.config["DEPLOYMENT_LOCATION"], "wafl")
            config_dir = os.path.join(target_path, "config", "common")

            self.logger.debug(f"🔗 Connecting to {username}@{self.ip} for configuration deployment")

            with paramiko.SSHClient() as ssh:
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(self.ip, port=ssh_port, username=username, pkey=key, timeout=30)

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
                    {"content": config_json, "filename": "config.json", "description": "agent configuration"}
                )

                # 2. Contact pattern (optional)
                contact_pattern = experiment_parameters.get("contact_pattern")
                if contact_pattern is None:
                    raise ValueError("contact_pattern cannot be None")
                contact_pattern_path = os.path.join("ctrl", "contact_pattern", contact_pattern)

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

                self.logger.info(
                    f"✅ All configurations deployed successfully to agent {self.name} ({len(deployed_files)} files)"
                )
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
            target_path = os.path.join(self.config["DEPLOYMENT_LOCATION"], "wafl")
            config_dir = os.path.join(target_path, "config", "common")
            config_file_path = os.path.join(config_dir, "config.json")

            self.logger.debug(f"🔗 Deploying config to {username}@{self.ip}:{config_file_path}")

            with paramiko.SSHClient() as ssh:
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(self.ip, port=ssh_port, username=username, pkey=key, timeout=30)

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

    def start_remote_process(self, experiment_id: str, args: Dict[str, Any]) -> bool:
        """
        Start wafl/src/main.py with nohup via SSH on execution server.

        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.info(f"🚀 Starting remote process for experiment '{experiment_id}' on {self.ip}")

        try:
            ssh_port = 22
            username = args["USER"]
            private_key_path = os.path.expanduser("~/.ssh/id_ed25519")

            if not os.path.exists(private_key_path):
                raise FileNotFoundError(f"🔑 SSH private key not found at {private_key_path}")

            key = paramiko.Ed25519Key.from_private_key_file(private_key_path)
            target_path = os.path.join(args["DEPLOYMENT_LOCATION"], "wafl")

            command_create_results = f"cd {os.path.join(target_path, 'results')} && mkdir -p {experiment_id}"
            command_start = (
                f"nohup python3 -u {os.path.join(target_path, 'src/main.py')} "
                f"> {os.path.join(target_path, 'results', experiment_id, 'log.txt')} "
                "2>&1 < /dev/null & echo $!"
            )

            self.logger.debug(f"🔗 Connecting to {username}@{self.ip}:{ssh_port}")

            with paramiko.SSHClient() as ssh:
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(self.ip, port=ssh_port, username=username, pkey=key, timeout=30)

                # Create results directory
                stdin, stdout, stderr = ssh.exec_command(command_create_results)
                exit_status = stdout.channel.recv_exit_status()
                if exit_status != 0:
                    error_msg = stderr.read().decode().strip()
                    self.logger.warning(f"⚠️ Results directory creation warning: {error_msg}")

                # Start main process
                stdin, stdout, stderr = ssh.exec_command(command_start)
                self.pid = stdout.readline().strip()
                exit_status = stdout.channel.recv_exit_status()

                if not self.pid or not self.pid.isdigit():
                    error_msg = stderr.read().decode().strip()
                    raise RuntimeError(f"❌ Failed to start remote process: {error_msg}")

                stdout.channel.close()
                self.status = "READY"
                self.logger.info(f"✅ Remote process started successfully with PID: {self.pid}")
                return True

        except FileNotFoundError as e:
            self.logger.error(f"🔑 SSH key error: {e}")
            self.status = "ERROR"
            return False
        except paramiko.AuthenticationException as e:
            self.logger.error(f"🔒 SSH authentication failed for {self.ip}: {e}")
            self.status = "ERROR"
            return False
        except paramiko.SSHException as e:
            self.logger.error(f"🌐 SSH connection error to {self.ip}: {e}")
            self.status = "ERROR"
            return False
        except Exception as e:
            self.logger.error(f"💥 Unexpected error starting remote process: {e}", exc_info=True)
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
            status_code = parts[0].split("-")[0]
            logs = lines[1:]

            # Validate log count if specified
            if len(parts) == 2 and parts[1].isdigit():
                expected_log_count = int(parts[1])
                if len(logs) != expected_log_count:
                    self.logger.warning(
                        f"📊 Log count mismatch for agent {self.name}. Expected {expected_log_count}, got {len(logs)}"
                    )

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

    def begin_epoch(self, phase: str, epoch: int, options: Optional[List[str]] = None) -> bool:
        """
        Send BEGIN command to start training epoch.

        Returns:
            bool: True if successful, False otherwise
        """
        command = f"BEGIN-{phase}-{epoch:05d}"
        if options:
            command += f":{str(options)}"

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

    def __init__(self, config_path: str):
        self.logger = logging.getLogger("ControlServer")
        self.config = self._load_config(config_path)
        self.experiment_id = self._generate_experiment_id(self.config.get("EXPERIMENT_NAME", "exp"))
        self.results_dir = self._create_results_directory()
        self.agents: List[WaflAgent] = []
        self._setup_logging()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load config file (.wafl_execution_config_base)."""
        self.logger.info(f"📝 Loading configuration from {config_path}")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"🚫 Config file not found: {config_path}")

        try:
            # Load environment variables from shell script
            self._load_shell_env_vars(config_path)

            # Validate required environment variables
            required_vars = [
                "WAFL_DEVICE_NAMES",
                "WAFL_DEVICE_IPS",
                "DEPLOYMENT_LOCATION",
                "USER",
            ]
            missing_vars = [var for var in required_vars if not os.environ.get(var)]

            if missing_vars:
                raise ValueError(f"🚫 Missing required environment variables: {', '.join(missing_vars)}")

            config = {
                "WAFL_DEVICE_NAMES": os.environ.get("WAFL_DEVICE_NAMES", "0").split(","),
                "WAFL_DEVICE_IPS": os.environ.get("WAFL_DEVICE_IPS", "localhost").split(","),
                "WAFL_DEVICE_CTRL_PORT": int(os.environ.get("WAFL_DEVICE_CTRL_PORT", "10001")),
                "WAFL_DEVICE_P2P_PORT": int(os.environ.get("WAFL_DEVICE_P2P_PORT", "10002")),
                "DEPLOYMENT_LOCATION": os.environ.get("DEPLOYMENT_LOCATION"),
                "USER": os.environ.get("USER"),
                "EXPERIMENT_NAME": os.environ.get("EXPERIMENT_NAME", "exp"),
            }

            # Validate device configuration
            if len(config["WAFL_DEVICE_NAMES"]) != len(config["WAFL_DEVICE_IPS"]):
                raise ValueError(
                    f"🚫 Device names and IPs count mismatch: "
                    f"{len(config['WAFL_DEVICE_NAMES'])} names vs {len(config['WAFL_DEVICE_IPS'])} IPs"
                )

            self.logger.info(f"✅ Configuration loaded successfully. Devices: {config['WAFL_DEVICE_NAMES']}")
            return config

        except Exception as e:
            self.logger.error(f"💥 Failed to load configuration: {e}", exc_info=True)
            raise

    def _load_shell_env_vars(self, config_path: str):
        """Load environment variables from shell script file using subprocess."""
        try:
            cmd = f"bash -c 'source {config_path} && env'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True, timeout=30)

            env_count = 0
            for line in result.stdout.strip().split("\n"):
                if "=" in line and not line.startswith("_"):
                    key, value = line.split("=", 1)
                    os.environ[key] = value
                    env_count += 1

            self.logger.debug(f"📝 Loaded {env_count} environment variables from {config_path}")

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"⏰ Timeout loading config file {config_path}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"💥 Error executing config file {config_path}: {e.stderr}")
        except Exception as e:
            raise RuntimeError(f"💥 Unexpected error loading config: {e}")

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
        agents = []
        names = self.config["WAFL_DEVICE_NAMES"]
        ips = self.config["WAFL_DEVICE_IPS"]
        port = self.config["WAFL_DEVICE_CTRL_PORT"]

        experiment_parameters["experiment_id"] = self.experiment_id
        experiment_parameters["results_dir"] = self.results_dir

        failed_agents = []
        for name, ip in zip(names, ips):
            try:
                agent = WaflAgent(name, ip, port, self.config, experiment_parameters)
                agents.append(agent)
                self.logger.info(f"🤖 Created and configured agent '{name}' for {ip}:{port}")
            except Exception as e:
                self.logger.error(f"💥 Failed to create agent '{name}': {e}")
                failed_agents.append(name)

        if failed_agents:
            raise RuntimeError(f"❌ Failed to create agents: {', '.join(failed_agents)}")

        self.logger.info(f"✅ Created {len(agents)} agents successfully with configurations deployed")
        return agents

    def _setup_logging(self):
        """Setup experiment logging to file and console."""
        try:
            log_file = os.path.join(self.results_dir, "control_server.log")
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

    def run_experiment(self, epochs: int, wafl_phase_params: Dict[str, Any], contact_pattern: str):
        """
        Execute entire experiment sequence (startup, training loop, shutdown).

        Note:
            Agent shutdown is always performed in the finally block, even if an exception occurs during the experiment.
        """
        self.logger.info(f"🚀 Starting experiment: {self.experiment_id} ({epochs} epochs)")
        experiment_success = False

        try:
            # 0. Create agents with unified configuration deployment
            self.logger.info("📋 Phase 0: Creating agents and deploying configurations")
            experiment_parameters = {
                "epochs": epochs,
                "wafl_phase_params": wafl_phase_params,
                "contact_pattern": contact_pattern,
            }

            self.agents = self._create_agents(experiment_parameters)
            self.logger.info("✅ All agents created and configured successfully")

            # 1. Start remote processes on all agents
            self.logger.info(f"🎬 Phase 1: Starting {len(self.agents)} remote processes")
            failed_agents = []

            for agent in self.agents:
                if not agent.start_remote_process(self.experiment_id, self.config):
                    failed_agents.append(agent.name)

            if failed_agents:
                raise RuntimeError(f"❌ Failed to start agents: {', '.join(failed_agents)}")

            self.logger.info("✅ All agents started successfully")

            # 2. Main training loop
            self.logger.info(f"🎓 Phase 2: Starting training loop ({epochs} epochs)")

            for epoch in range(1, epochs + 1):
                self.logger.info(f"📚 === Epoch {epoch}/{epochs} ===")

                # Send begin commands to all agents
                failed_commands = []
                for agent in self.agents:
                    peers = wafl_phase_params.get("peers")
                    if not agent.begin_epoch(phase="WAFL", epoch=epoch, options=peers):
                        failed_commands.append(agent.name)

                if failed_commands:
                    raise RuntimeError(f"❌ Failed to start epoch {epoch} on agents: {', '.join(failed_commands)}")

                # Wait for completion
                self._wait_for_all_agents_to_complete(current_epoch=epoch)
                self.logger.info(f"✅ Epoch {epoch}/{epochs} completed successfully")

            self.logger.info("🎉 All training epochs completed successfully")
            experiment_success = True

        except KeyboardInterrupt:
            self.logger.warning("⚠️ Experiment interrupted by user")
        except Exception as e:
            self.logger.error(f"💥 Experiment failed: {e}", exc_info=True)
        finally:
            # 3. Shutdown all agents
            self.logger.info("🛑 Phase 3: Shutting down all agents")
            self._shutdown_all_agents()

            status = "SUCCESS" if experiment_success else "FAILED"
            self.logger.info(f"🏁 Experiment {self.experiment_id} finished with status: {status}")

    def _wait_for_all_agents_to_complete(self, current_epoch: int, poll_interval: int = 15, timeout: int = 3600):
        """Polls agents until they all complete the current epoch."""
        self.logger.info(f"⏳ Waiting for all agents to complete epoch {current_epoch}")
        start_time = time.time()
        last_progress_log = 0

        while True:
            elapsed_time = time.time() - start_time

            if elapsed_time > timeout:
                raise TimeoutError(
                    f"⏰ Timeout waiting for epoch {current_epoch} completion after {timeout}s. Some agents may be stuck."
                )

            finished_agents = set()
            error_agents = []

            for agent in self.agents:
                try:
                    status_code, logs = agent.get_status()

                    # Log agent output
                    if logs:
                        for log_line in logs:
                            if log_line.strip():  # Skip empty lines
                                self.logger.info(f"[{agent.name}] {log_line}")

                    # Check for errors
                    if "ERROR" in status_code:
                        error_agents.append(f"{agent.name}({status_code})")
                        continue

                    # Check completion status
                    if status_code.startswith("DONE"):
                        try:
                            done_epoch = int(status_code.split("-")[-1])
                            if done_epoch >= current_epoch:
                                finished_agents.add(agent.name)
                        except (ValueError, IndexError):
                            self.logger.warning(f"⚠️ Could not parse epoch from status '{status_code}' for agent {agent.name}")

                except Exception as e:
                    self.logger.error(f"💥 Error getting status from agent {agent.name}: {e}")
                    error_agents.append(f"{agent.name}(COMM_ERROR)")

            # Report errors immediately
            if error_agents:
                raise RuntimeError(f"❌ Agents reported errors: {', '.join(error_agents)}")

            # Progress logging (every 60 seconds)
            if elapsed_time - last_progress_log >= 60:
                self.logger.info(
                    f"📊 Progress: {len(finished_agents)}/{len(self.agents)} agents completed (elapsed: {elapsed_time:.0f}s)"
                )
                last_progress_log = elapsed_time

            # Check if all completed
            if len(finished_agents) == len(self.agents):
                self.logger.info(f"✅ All agents completed epoch {current_epoch} in {elapsed_time:.1f}s")
                break

            time.sleep(poll_interval)

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
        required_params = ["epochs", "contact_pattern", "wafl_phase_params"]
        missing_params = [param for param in required_params if param not in experiment_parameters]

        if missing_params:
            print(f"💥 Missing required parameters in {PARAMETERS_PATH}: {', '.join(missing_params)}")
            exit(1)

        print(f"🚀 Starting experiment with {experiment_parameters['epochs']} epochs")
        print(f"📋 Contact pattern: {experiment_parameters['contact_pattern']}")
        print(f"📋 WAFL parameters: {experiment_parameters['wafl_phase_params']}")

        # Config file path
        CONFIG_PATH = "ctrl/wafl_execution_base_config"

        # Create ControlServer instance
        controller = ControlServer(config_path=CONFIG_PATH)

        # Run experiment
        controller.run_experiment(
            epochs=experiment_parameters["epochs"],
            wafl_phase_params=experiment_parameters["wafl_phase_params"],
            contact_pattern=experiment_parameters["contact_pattern"],
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

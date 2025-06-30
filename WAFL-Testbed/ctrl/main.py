import datetime
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

    def __init__(self, device_name: str, ip_address: str, ctrl_port: int, timeout: int = 10):
        self.name = device_name
        self.ip = ip_address
        self.ctrl_port = ctrl_port
        self.status = "UNKNOWN"
        self.logger = logging.getLogger(f"WaflAgent-{device_name}")
        self.pid = None
        self.timeout = timeout

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
            target_path = os.path.join(args["DEPLOYMENT_LOCATION"], "ctrl")

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
            if status_code not in valid_statuses and not any(status_code.startswith(s) for s in valid_statuses):
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
        self.agents: List[WaflAgent] = self._create_agents()
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
            required_vars = ["WAFL_DEVICE_NAMES", "WAFL_DEVICE_IPS", "WAFL_DEVICE_CTRL_PORT", "DEPLOYMENT_LOCATION", "USER"]
            missing_vars = [var for var in required_vars if not os.environ.get(var)]

            if missing_vars:
                raise ValueError(f"🚫 Missing required environment variables: {', '.join(missing_vars)}")

            config = {
                "WAFL_DEVICE_NAMES": os.environ.get("WAFL_DEVICE_NAMES", "0").split(","),
                "WAFL_DEVICE_IPS": os.environ.get("WAFL_DEVICE_IPS", "localhost").split(","),
                "WAFL_DEVICE_CTRL_PORT": int(os.environ.get("WAFL_DEVICE_CTRL_PORT", "10001")),
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

    def _create_agents(self) -> List[WaflAgent]:
        """Create WaflAgent instance list based on config."""
        agents = []
        names = self.config["WAFL_DEVICE_NAMES"]
        ips = self.config["WAFL_DEVICE_IPS"]
        port = self.config["WAFL_DEVICE_CTRL_PORT"]

        for name, ip in zip(names, ips):
            agent = WaflAgent(name, ip, port)
            agents.append(agent)
            self.logger.info(f"🤖 Created agent '{name}' for {ip}:{port}")

        self.logger.info(f"✅ Created {len(agents)} agents successfully")
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

    def run_experiment(self, epochs: int, wafl_phase_params: Dict[str, Any]):
        """
        Execute entire experiment sequence (startup, training loop, shutdown).
        """
        self.logger.info(f"🚀 Starting experiment: {self.experiment_id} ({epochs} epochs)")
        experiment_success = False

        try:
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
        # Config file path
        CONFIG_PATH = "ctrl/wafl_execution_base_config"

        # Create ControlServer instance
        controller = ControlServer(config_path=CONFIG_PATH)

        # Experiment parameters
        experiment_parameters = {
            "epochs": 50,
            "wafl_phase_params": {"aggregation_strategy": "FedAvg"},
        }

        # Run experiment
        controller.run_experiment(
            epochs=experiment_parameters["epochs"], wafl_phase_params=experiment_parameters["wafl_phase_params"]
        )

    except Exception as e:
        print(f"💥 Fatal error in main: {e}")
        exit(1)

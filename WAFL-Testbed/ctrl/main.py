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
        status (str): Current status (e.g., "UNKNOWN", "READY", "RUNNING", "DONE", "ERROR")
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

    def start_remote_process(self, experiment_id: str, args: Dict[str, Any]):
        """
        Start wafl/src/main.py with nohup via SSH on execution server.
        """
        self.logger.info(f"🚀 Starting execution program (experiment ID: {experiment_id})")

        ssh_port = 22

        username = args["USER"]

        private_key_path = os.path.expanduser("~/.ssh/id_ed25519")
        key = paramiko.Ed25519Key.from_private_key_file(private_key_path)

        target_path = os.path.join(args["DEPLOYMENT_LOCATION"], args["EXPERIMENT_NAME"])

        command_create_results = "cd " + os.path.join(target_path, "results") + " && mkdir " + experiment_id
        command_start = "nohup python3 -u " + os.path.join(target_path, "src/main.py") + " > log.txt 2>&1 < /dev/null & echo $!"

        self.logger.info(f"Connecting to {self.ip}...")

        with paramiko.SSHClient() as ssh:
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.ip, port=ssh_port, username=username, pkey=key)
            ssh.exec_command(command_create_results)
            stdin, stdout, stderr = ssh.exec_command(command_start)
            self.pid = stdout.readline().strip()
            stdout.channel.close()
            self.logger.info(f"Started process with PID: {self.pid}")

    def _send_command(self, command: str) -> Tuple[bool, str]:
        """
        Send command to control TCP port and receive response.

        Returns:
            Tuple[bool, str]: (success flag, response string)
        """
        self.logger.debug(f"🔌 TCP Send to {self.ip}:{self.ctrl_port}: {command.strip()}")
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
                self.logger.debug(f"📡 TCP Recv from {self.ip}:{self.ctrl_port}: {full_response}")
                return True, full_response
        except socket.timeout:
            self.logger.error(f"TCP connection to {self.ip}:{self.ctrl_port} timed out.")
            return False, "ERROR:TIMEOUT"
        except socket.error as e:
            self.logger.error(f"TCP socket error on {self.ip}:{self.ctrl_port}: {e}")
            return False, f"ERROR:{e}"

    def get_status(self) -> Tuple[str, List[str]]:
        """
        Send STAT command and get status and stdout.

        Returns:
            Tuple[str, List[str]]: (status string, stdout list)
        """
        success, response = self._send_command("STAT\r\n")

        if not success:
            return "ERROR_COMM", [response]  # Communication error

        lines = response.split("\n")
        if not lines or not lines[0]:
            return "ERROR_PARSE", ["Received empty response from agent."]

        first_line = lines[0].strip()
        parts = first_line.split(":", 1)

        status_code = parts[0]
        logs = lines[1:]

        # Validate format
        if len(parts) == 2 and parts[1].isdigit():
            expected_log_count = int(parts[1])
            if len(logs) != expected_log_count:
                self.logger.warning(f"Log line count mismatch. Expected {expected_log_count}, got {len(logs)}.")
        elif status_code not in ["OK", "ERROR"]:
            self.logger.warning(f"Unrecognized status format: {first_line}")

        self.logger.debug(f"📈 Status received: {status_code}")
        return status_code, logs

    def begin_epoch(self, phase: str, epoch: int, options: Optional[List[str]] = None):
        """
        Send BEGIN command to start training epoch.
        Pass options like model exchange target list.
        """
        command = f"BEGIN-{phase}-{epoch:05d}"
        if options:
            command += f":{str(options)}"

        self.logger.info(f"📤 Sending command: {command}")
        success, response = self._send_command(f"{command}\r\n")
        if not success or response != "OK":
            self.logger.error(f"Failed to start epoch. Response: {response}")

    def begin_evaluation(self, eval_name: str = "eval"):
        """
        Start evaluation routine with BEGIN-eval command.
        """
        command = f"BEGIN-{eval_name}"
        self.logger.info(f"📊 Starting evaluation routine '{eval_name}'")
        success, response = self._send_command(f"{command}\r\n")
        if not success or response != "OK":
            self.logger.error(f"Failed to start evaluation. Response: {response}")

    def send_kill_command(self) -> bool:
        """
        Send KILL command to terminate process normally.
        """
        self.logger.warning("🔪 Sending KILL command.")
        success, response = self._send_command("KILL\r\n")
        if success and response == "OK":
            self.logger.info("✅ KILL command acknowledged.")
            self.status = "TERMINATED"
            return True
        else:
            self.logger.error(f"KILL command failed or was not acknowledged. Response: {response}")
            return False

    def force_kill_process(self, args: Dict[str, Any]):
        """
        Force kill process via SSH.
        """
        self.logger.error("💀 Force killing process")

        ssh_port = 22

        username = args["USER"]

        private_key_path = os.path.expanduser("~/.ssh/id_ed25519")
        key = paramiko.Ed25519Key.from_private_key_file(private_key_path)

        command_kill = "kill " + self.pid

        with paramiko.SSHClient() as ssh:
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.ip, port=ssh_port, username=username, pkey=key)
            if self.pid is not None:
                ssh.exec_command(command_kill)


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
        print(f"📝 Loading config file {config_path}")

        # run source wafl_execution_base_config
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load environment variables from shell script
        self._load_shell_env_vars(config_path)

        # Return config dictionary from environment variables
        return {
            "WAFL_DEVICE_NAMES": os.environ.get("WAFL_DEVICE_NAMES", "0").split(","),
            "WAFL_DEVICE_IPS": os.environ.get("WAFL_DEVICE_IPS", "localhost").split(","),
            "WAFL_DEVICE_CTRL_PORT": int(os.environ.get("WAFL_DEVICE_CTRL_PORT", "10001")),
            "DEPLOYMENT_LOCATION": os.environ.get("DEPLOYMENT_LOCATION"),
            "USER": os.environ.get("USER"),
            "EXPERIMENT_NAME": os.environ.get("EXPERIMENT_NAME", "exp"),
        }

    def _load_shell_env_vars(self, config_path: str):
        """Load environment variables from shell script file using subprocess."""
        try:
            # Use subprocess to source the file and dump environment
            cmd = f"bash -c 'source {config_path} && env'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)

            # Parse the output and set environment variables
            for line in result.stdout.strip().split("\n"):
                if "=" in line and not line.startswith("_"):  # Skip shell internal variables
                    key, value = line.split("=", 1)
                    os.environ[key] = value
        except:
            self.logger.error(f"Failed to load config from {config_path}. Ensure it exists and is valid.")
            raise

    def _generate_experiment_id(self, name: str) -> str:
        """Generate experiment ID in 'experiment-name-timestamp' format."""
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        return f"{name}-{timestamp}"

    def _create_results_directory(self):
        """Create directory to save experiment results."""
        results_path = os.path.join("results", self.experiment_id, "summary")
        os.makedirs(results_path, exist_ok=True)
        print(f"📁 Created results directory: {results_path}")
        return results_path

    def _create_agents(self) -> List[WaflAgent]:
        """Create WaflAgent instance list based on config."""
        agents = []
        names = self.config["WAFL_DEVICE_NAMES"]
        ips = self.config["WAFL_DEVICE_IPS"]
        port = self.config["WAFL_DEVICE_CTRL_PORT"]
        for name, ip in zip(names, ips):
            agents.append(WaflAgent(name, ip, port))
        return agents

    def _setup_logging(self):
        """Setup experiment logging to file and console."""
        log_file = os.path.join(self.results_dir, "control_server.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
        )

        self.logger.info(f"📊 Logging to {log_file}")
        self.logger.info(f"🎯 Experiment ID: {self.experiment_id}")

    def run_experiment(self, epochs: int, wafl_phase_params: Dict[str, Any]):
        """
        Execute entire experiment sequence (startup, training loop, shutdown).
        """
        self.logger.info(f"🚀 Starting experiment: {self.experiment_id}")
        try:
            # 1. Start remote processes on all agents
            for agent in self.agents:
                agent.start_remote_process(self.experiment_id, self.config)

            # Check if all agents started successfully
            if any(agent.status == "ERROR" for agent in self.agents):
                raise RuntimeError("One or more agents failed to start.")

            self.logger.info("✅ All agents started successfully.")

            # 2. Main training loop
            for epoch in range(1, epochs + 1):
                self.logger.info(f"--- Starting Epoch {epoch}/{epochs} ---")

                # Instruct all agents to begin the epoch
                for agent in self.agents:
                    # Logic to determine peers can be added here
                    peers = wafl_phase_params.get("peers")
                    agent.begin_epoch(phase="WAFL", epoch=epoch, options=peers)

                # Wait for all agents to complete the epoch
                self._wait_for_all_agents_to_complete(current_epoch=epoch)
                self.logger.info(f"--- ✅ Completed Epoch {epoch}/{epochs} ---")

            self.logger.info("🎉 All training epochs completed successfully.")

        except Exception as e:
            self.logger.error(f"💥 An error occurred during the experiment: {e}", exc_info=True)
        finally:
            # 3. Shutdown all agents
            self._shutdown_all_agents()
            self.logger.info(f"--- Experiment {self.experiment_id} finished ---")

    def _wait_for_all_agents_to_complete(self, current_epoch: int, poll_interval: int = 15, timeout: int = 3600):
        """Polls agents until they all complete the current epoch."""
        self.logger.info(f"⏳ Waiting for all agents to complete epoch {current_epoch}...")
        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Waiting for agents to complete epoch {current_epoch} timed out after {timeout} seconds.")

            finished_agents = set()
            for agent in self.agents:
                status_code, logs = agent.get_status()

                if logs:  # Log any output from agents
                    for log_line in logs:
                        self.logger.info(f"[{agent.name}] {log_line}")

                # Check for errors first
                if "ERROR" in status_code:
                    raise RuntimeError(f"Agent {agent.name} reported an error state: {status_code}")

                # Check if agent is done with the current or a later epoch
                if status_code.startswith("DONE"):
                    try:
                        done_epoch = int(status_code.split("-")[-1])
                        if done_epoch >= current_epoch:
                            finished_agents.add(agent.name)
                    except (ValueError, IndexError):
                        self.logger.warning(f"Could not parse epoch from status '{status_code}' for agent {agent.name}")

            self.logger.info(f"Completion status: {len(finished_agents)}/{len(self.agents)} agents done.")
            if len(finished_agents) == len(self.agents):
                break

            time.sleep(poll_interval)

    def _shutdown_all_agents(self):
        """Terminates all agent processes, trying gracefully first, then forcefully."""
        self.logger.warning("🛑 Starting shutdown of all agents")
        for agent in self.agents:
            if agent.pid is None:  # Skip agents that never started
                continue

            if not agent.send_kill_command():
                self.logger.warning(f"Graceful shutdown failed for agent {agent.name}. Attempting force kill.")
                agent.force_kill_process(self.config)


if __name__ == "__main__":
    # Config file path
    CONFIG_PATH = "ctrl/wafl_execution_base_config"

    # Create ControlServer instance
    controller = ControlServer(config_path=CONFIG_PATH)

    # dummy
    experiment_parameters = {
        "epochs": 50,
        "wafl_phase_params": {"aggregation_strategy": "FedAvg"},
    }

    # Run experiment
    controller.run_experiment(
        epochs=experiment_parameters["epochs"], wafl_phase_params=experiment_parameters["wafl_phase_params"]
    )

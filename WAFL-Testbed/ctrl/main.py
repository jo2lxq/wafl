import datetime
import logging
import os
import socket
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

        private_key_path = "/home/denjo/.ssh/id_ed25519"  # dummy
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

        private_key_path = "/home/denjo/.ssh/id_ed25519"  # dummy
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

        # dummy
        return {
            "WAFL_DEVICE_NAMES": ["0", "1", "2"],
            "WAFL_DEVICE_IPS": ["192.168.11.100", "192.168.11.101", "192.168.11.102"],
            "WAFL_DEVICE_CTRL_PORT": 10001,
            "DEPLOYMENT_LOCATION": "/home/denjo/testbed",
            "USER": "denjo",
            "EXPERIMENT_NAME": "exp",
        }

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
        pass

    def _wait_for_all_agents_to_complete(self):
        """Poll and wait until all agents complete processing."""
        self.logger.info("⏳ Waiting for all agents to complete...")
        pass

    def _shutdown_all_agents(self):
        """Terminate all agent processes."""
        self.logger.warning("🛑 Starting shutdown of all agents")
        pass


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

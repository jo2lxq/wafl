import datetime
import logging
import os
from typing import Any, Dict, List, Optional, Tuple


class WaflAgent:
    """
    Represents each execution server (WAFL device) and manages communication.

    Attributes:
        name (str): Device name (e.g., "0", "1")
        ip (str): IP address
        ctrl_port (int): Control TCP port number
        status (str): Current status (e.g., "UNKNOWN", "READY", "RUNNING", "DONE", "ERROR")
    """

    def __init__(self, device_name: str, ip_address: str, ctrl_port: int):
        self.name = device_name
        self.ip = ip_address
        self.ctrl_port = ctrl_port
        self.status = "UNKNOWN"
        self.logger = logging.getLogger(f"WaflAgent-{device_name}")

    def start_remote_process(self, experiment_id: str, args: Dict[str, Any]):
        """
        Start wafl/src/main.py with nohup via SSH on execution server.
        """
        self.logger.info(f"🚀 Starting execution program (experiment ID: {experiment_id})")
        pass

    def _send_command(self, command: str) -> Tuple[bool, str]:
        """
        Send command to control TCP port and receive response.

        Returns:
            Tuple[bool, str]: (success flag, response string)
        """
        self.logger.debug(f"🔌 TCP send: {command.strip()}")
        response = "OK"
        return True, response

    def get_status(self) -> Tuple[str, List[str]]:
        """
        Send STAT command and get status and stdout.

        Returns:
            Tuple[str, List[str]]: (status string, stdout list)
        """
        # dummy
        status_code = "DONE-SELF-00001"
        logs = ["log line 1", "log line 2"]
        self.logger.debug(f"📈 Status retrieved: {status_code}")
        return status_code, logs

    def begin_epoch(self, phase: str, epoch: int, options: Optional[List[str]] = None):
        """
        Send BEGIN command to start training epoch.
        Pass options like model exchange target list.
        """
        command = f"BEGIN-{phase}-{epoch:05d}"
        if options:
            command += f":{options}"
        self.logger.info(f"📤 Sending command: {command}")
        success, _ = self._send_command(f"{command}\r\n")
        pass

    def begin_evaluation(self, eval_name: str = "eval"):
        """
        Start evaluation routine with BEGIN-eval command.
        """
        self.logger.info(f"📊 Starting evaluation routine '{eval_name}'")
        success, _ = self._send_command(f"BEGIN-{eval_name}\r\n")
        pass

    def send_kill_command(self) -> bool:
        """
        Send KILL command to terminate process normally.
        """
        self.logger.warning("🔪 Sending KILL command")
        success, _ = self._send_command("KILL\r\n")
        return success

    def force_kill_process(self):
        """
        Force kill process via SSH.
        """
        self.logger.error("💀 Force killing process")
        pass


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
            "EXPERIMENT_NAME": "exp",
        }

    def _generate_experiment_id(self, name: str) -> str:
        """Generate experiment ID in 'experiment-name-timestamp' format."""
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        return f"{name}-{timestamp}"

    def _create_results_directory(self):
        """Create directory to save experiment results."""
        summary_path = os.path.join("results", self.experiment_id, "summary")
        print(f"📁 Creating results directory: {summary_path}")
        return summary_path

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

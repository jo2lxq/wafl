import logging
import os
import socket
import subprocess
import threading
import time
from typing import Any, Dict

"""
mock of wafl/main.py
"""


class WaflExecution(threading.Thread):
    """
    Execute 1 epoch of learning.

    Attributes:
        name (str): Device name (e.g., "0", "1")
        p2p_port (int): P2P TCP port number
    """

    def __init__(self, device_name: str, p2p_port: int):
        self.name = device_name
        self.p2p_port = p2p_port

    def run(self):
        time.sleep(10)  # dumny


class SendParameter(threading.Thread):
    """
    Recieve request from other device and send parameters.

    Attributes:
        name (str): Device name (e.g., "0", "1")
        p2p_port (int): P2P TCP port number
    """

    def __init__(self, device_name: str, p2p_port: int):
        self.name = device_name
        self.ctrl_port = p2p_port
        self.running = True

    def run(self):
        while self.running:
            time.sleep(5)  # dumny

    def stop(self):
        self.running = False


class ExecutionServer:
    """
    Main implementation of wafl/main.py.
    """

    def __init__(self, config_path: str):
        self.device_name = "0"  # dummy
        self.logger = logging.getLogger("ExecutionServer")
        self.config = self._load_config(config_path)
        self.experiment_id = "exp-20250630T011525"  # dummy
        self.results_dir = os.path.join("results", self.experiment_id)
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
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}. Ensure it exists and is valid. Exception: {e}")
            raise

    def _setup_logging(self):
        """Setup experiment logging to file and console."""
        log_file = os.path.join(self.results_dir, "execution_server.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
        )

        self.logger.info(f"📊 Logging to {log_file}")
        self.logger.info(f"🎯 Device: {self.device_name}")
        self.logger.info(f"🎯 Experiment ID: {self.experiment_id}")

    def run(self):
        """
        Run execution server
        """
        self.logger.info(f"🚀 Starting execution server {self.device_name}: {self.experiment_id}")

        send_parameter = SendParameter(self.device_name, self.config["WAFL_DEVICE_P2P_PORT"])
        send_parameter.start()

        host = "127.0.0.1"
        ctrl_port = self.config["WAFL_DEVICE_CTRL_PORT"]

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, ctrl_port))
            s.listen()
            while True:
                try:
                    conn, addr = s.accept()
                    data = conn.recv(4096)
                    if not data:
                        conn.close()
                        continue
                    command = data.decode("utf-8")
                    if command.startswith("BEGIN"):
                        pass

                    elif command == "STAT":
                        pass

                    elif command == "KILL":
                        send_parameter.stop()
                        pass
                    conn.close()

                except Exception as e:
                    time.sleep(1)
                    self.logger.error(f"The Exception occurred. Exception: {e}")


if __name__ == "__main__":
    # Config file path
    CONFIG_PATH = "wafl_execution_base_config"

    # Create ControlServer instance
    executor = ExecutionServer(config_path=CONFIG_PATH)

    # Run experiment
    executor.run()

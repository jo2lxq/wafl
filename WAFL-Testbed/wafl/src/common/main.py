import io
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


def count_lines_in_string(text_string: str) -> int:
    """
    Returns the number of lines in the given string.
    Returns 0 for an empty string.
    """
    if not text_string:
        return 0
    return len(text_string.splitlines(keepends=True))


class WaflExecution(threading.Thread):
    """
    Execute 1 epoch of learning.

    Attributes:
        name (str): Device name (e.g., "0", "1")
        p2p_port (int): P2P TCP port number
    """

    def __init__(self, device_name: str, p2p_port: int):
        super().__init__()
        self.name = device_name
        self.p2p_port = p2p_port

    def run(self):
        time.sleep(5)  # dumny


class SendParameter(threading.Thread):
    """
    Recieve request from other device and send parameters.

    Attributes:
        name (str): Device name (e.g., "0", "1")
        p2p_port (int): P2P TCP port number
    """

    def __init__(self, device_name: str, p2p_port: int):
        super().__init__()
        self.name = device_name
        self.p2p_port = p2p_port
        self.running = True

    def run(self):
        while self.running:
            time.sleep(1)  # dumny

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
        self.results_dir = os.path.join("/home/denjo/testbed/DEMO_zenko/results", self.experiment_id)
        self.log_stream = io.StringIO()
        self.phase = "READY"
        self.epoch = "00000"
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
            "WAFL_DEVICE_P2P_PORT": int(os.environ.get("WAFL_DEVICE_P2P_PORT", "10002")),
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
        log_file = os.path.join("/home/denjo/testbed/DEMO_zenko/results", "execution_server.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler(),
                logging.StreamHandler(self.log_stream),
            ],
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

        host = "0.0.0.0"
        ctrl_port = self.config["WAFL_DEVICE_CTRL_PORT"]
        execute_epoch = None

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, ctrl_port))
            s.listen()
            while True:
                try:
                    self.logger.info(f"Waiting at port {ctrl_port}...")
                    conn, addr = s.accept()
                    data = conn.recv(4096)
                    if not data:
                        conn.close()
                        continue
                    command = data.decode("utf-8")
                    if command.startswith("BEGIN"):
                        self.logger.info("Recieve BEGIN.")
                        parts = command.split("-")
                        self.phase = parts[1].strip()
                        self.epoch = parts[2].strip()
                        response = "UNDEFINED\r\n"
                        try:
                            self.logger.info(f"Start {self.phase} epoch {self.epoch}.")
                            execute_epoch = WaflExecution(self.device_name, self.config["WAFL_DEVICE_P2P_PORT"])
                            execute_epoch.start()
                            response = "OK\r\n"

                        except Exception as e:
                            time.sleep(1)
                            self.logger.error(f"The Exception occurred at BEGIN. Exception: {e}")
                            response = "ERROR\r\n"

                        conn.sendall(response.encode("utf-8"))

                    elif command.startswith("STAT"):
                        self.logger.info("Recieve STAT.")
                        stat = "????"
                        if execute_epoch is not None and execute_epoch.is_alive():
                            stat = "EXEC"
                        else:
                            stat = "DONE"
                        response = "-".join([stat, self.phase, self.epoch])
                        captured_logs = self.log_stream.getvalue()
                        self.log_stream.seek(0)
                        self.log_stream.truncate(0)
                        z = count_lines_in_string(captured_logs)
                        response += ":" + str(z) + "\r\n"
                        response += captured_logs
                        conn.sendall(response.encode("utf-8"))

                    elif command.startswith("KILL"):
                        self.logger.info("Recieve KILL.")
                        response = "UNDEFINED\r\n"
                        try:
                            send_parameter.stop()
                            execute_epoch.join()
                            time.sleep(2)
                            if send_parameter.is_alive():
                                raise Exception("failed to kill send_parameter")
                            if execute_epoch is not None and execute_epoch.is_alive():
                                raise Exception("failed to kill execute_epoch")
                            response = "OK\r\n"
                        except Exception as e:
                            self.logger.error(f"The Exception occurred at KILL. Exception: {e}")
                            response = "ERROR\r\n"
                        conn.sendall(response.encode("utf-8"))
                        conn.close()
                        break

                    conn.close()

                except Exception as e:
                    time.sleep(1)
                    self.logger.error(f"The Exception occurred. Exception: {e}")


if __name__ == "__main__":
    # Config file path
    # CONFIG_PATH = "wafl_execution_base_config"
    CONFIG_PATH = "/home/denjo/testbed/DEMO_zenko/wafl_execution_base_config"

    # Create ControlServer instance
    executor = ExecutionServer(config_path=CONFIG_PATH)

    # Run experiment
    executor.run()

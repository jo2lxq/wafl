import io
import json
import logging
import os
import socket
import threading
import time
from typing import Any, Dict

"""
mock of wafl/main.py
"""

DEPLOYMENT_LOCATION = "/home/denjo/workspace/ktakahashi"  # dummy


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
        time.sleep(5)  # dummy


class SendParameter(threading.Thread):
    """
    Receive request from other device and send parameters.

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
            time.sleep(1)  # dummy

    def stop(self):
        self.running = False


class ExecutionServer:
    """
    Main implementation of wafl/main.py.
    """

    def __init__(self, config_path: str):
        self.logger = logging.getLogger("ExecutionServer")
        self.config = self._load_config(config_path)
        self.device_name = self.config["DEVICE_NAME"]
        self.experiment_id = self.config["EXPERIMENT_ID"]
        self.results_dir = os.path.join(DEPLOYMENT_LOCATION, self.config["PROJECT_NAME"], "results", self.experiment_id)
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

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)

            # Extract relevant configuration values from JSON structure
            return {
                "WAFL_DEVICE_NAMES": config_data["infrastructure"]["device_names"],
                "WAFL_DEVICE_IPS": config_data["infrastructure"]["device_ips"],
                "WAFL_DEVICE_CTRL_PORT": config_data["infrastructure"]["ctrl_port"],
                "WAFL_DEVICE_P2P_PORT": config_data["infrastructure"]["p2p_port"],
                "PROJECT_NAME": config_data["experiment_info"]["project_name"],
                "DEVICE_NAME": config_data["agent_info"]["device_name"],
                "EXPERIMENT_ID": config_data["experiment_info"]["experiment_id"],
                "EXPERIMENT_NAME": config_data["experiment_info"]["experiment_name"],
                "EPOCHS": config_data["experiment_parameters"]["epochs"],
                "WAFL_PHASE_PARAMS": config_data["experiment_parameters"]["wafl_phase_params"],
                "LOG_LEVEL": config_data["runtime"]["log_level"],
            }

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in config file: {e}")
        except KeyError as e:
            raise ValueError(f"Missing required key in config file: {e}")

    def _setup_logging(self):
        """Setup experiment logging to file and console."""
        log_file = os.path.join(self.results_dir, "execution_server.log")
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
                        self.logger.info("Receive BEGIN.")
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
                        self.logger.info("Receive STAT.")
                        stat = "????"
                        if execute_epoch is not None and execute_epoch.is_alive():
                            stat = "EXEC"
                        else:
                            stat = "DONE"
                        response = "-".join([stat, self.phase, self.epoch])
                        # captured_logs = self.log_stream.getvalue()
                        # self.log_stream.seek(0)
                        # self.log_stream.truncate(0)
                        z = count_lines_in_string("")
                        response += ":" + str(z) + "\r\n"
                        response += ""
                        self.logger.info(f"STAT response: {response}")
                        conn.sendall(response.encode("utf-8"))

                    elif command.startswith("KILL"):
                        self.logger.info("Receive KILL.")
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
    # Create ControlServer instance
    print(os.getcwd())
    executor = ExecutionServer(config_path=os.path.join(DEPLOYMENT_LOCATION, "WAFL-Testbed", "config", "config.json"))

    # Run experiment
    executor.run()

from __future__ import annotations

import copy
import json
import logging
import os
import pickle
import socket
import threading
import time
import zlib
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from net import Net


class PickledDataset(torch.utils.data.Dataset):
    """
    Wrapper for the Subset.
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ModelLearningUtils:
    """
    A class for the WAFL model learning process.
    ACCESS: in the '../' format (Please run the main.py file from PROJECT/src directory).
    DATASET: PROJECT/dataset/dataset.pickled (Pickle of the subset: torch.utils.data.Dataset);
    MODEL (to be saved as): PROJECT/results/model_instance.pth;
    CONTACT PATTERN: PROJECT/config/contact_pattern.json (Dict of "XXXXX" : [] (epoch: neighbour list));
    CONFIG FILE: PROJECT/config/config_local.json
    """

    # Hyperparameter configuration
    # The hyperparameter values supplied by the config file will take precedence.
    # Also, some of these variables, such as cMAX_EPOCH are not relevant here, since
    # the epochs are initiated by the control server.
    # Such variables have been defined here to ensure consistency with the original
    # WAFL-MLP project.
    cBATCH_SIZE = 32  # default 32
    cLEARNING_RATE = 0.001  # default 0.001
    cFL_COEFFICIENCY = 1.0  # default 1.0  (WAFL's aggregation co efficiency)
    cSELF_TRAIN_EPOCHS = 50  # default 50
    cMAX_EPOCH = 5000  # default 5000

    def __init__(
        self,
        dataset_path: str,
        model_instance_path: str,
        contact_pattern_path: str,
        model_sharing: ModelSharingUtils,
        ctrl_tcp: CTRL_TCP,
    ) -> None:
        """
        Initialize the learning process instance.
        """
        torch.random.manual_seed(1)
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)
        self.model_sharing = model_sharing
        self.ctrl_tcp = ctrl_tcp
        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=ModelLearningUtils.cBATCH_SIZE, shuffle=False, num_workers=2
        )
        self.logger = logging.getLogger("ModelLearningUtils")
        self.model_instance_path = model_instance_path
        with open(contact_pattern_path, "r") as f:
            self.neighbour_map = json.load(f)
        if not isinstance(self.neighbour_map, list) or len(self.neighbour_map) == 0:
            self.logger.error("Contact pattern must be a non-empty dictionary")
        else:
            self.logger.info(f"Contact pattern loaded: {len(self.neighbour_map)} epochs")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = Net().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=ModelLearningUtils.cLEARNING_RATE)
        self.logger.info("Initialized the Model Learning Utils instance.")

    def get_neighbour_list(self, five_digit_number_str: str = "99999") -> List[int]:
        """
        Return the neighbour list for the current epoch from the contact pattern file.
        """
        neighbour_list = []
        try:
            neighbour_list = self.neighbour_map[int(five_digit_number_str)].get(str(self.model_sharing.name), [])
        except Exception as exc:
            self.logger.error(f"Error retrieving neighbour list: {str(exc)[:100]}...")
        return neighbour_list

    def self_learn(self, five_digit_number_str: str = "99999", WAFL_LEARN=False) -> bool:
        """
        Implementation of the Self-Learning Epoch Logic for WAFL-MLP.
        """
        if not WAFL_LEARN:
            self.logger.info(f"🧠 Beginning the Self-Learning Epoch: {five_digit_number_str}")
        SUCCESS = False
        try:
            running_loss = 0.0
            for i, data in enumerate(self.data_loader, 0):
                x_train, y_train = data
                x_train = x_train.to(self.device)
                y_train = y_train.to(self.device)

                self.optimizer.zero_grad()

                y_output = self.net(x_train)
                loss = self.criterion(y_output, y_train)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if (i + 1) % 50 == 0:
                    self.logger.debug(f"Running Loss: {running_loss / 50:.5f}")
                    running_loss = 0.0
            if not WAFL_LEARN:
                self.logger.info(f"🏁 Completed the Self-Learning Epoch: {five_digit_number_str}")
            torch.save(
                self.net.state_dict(),
                self.model_instance_path,
            )
            self.model_sharing.update_model_instance(self.net.state_dict(), "Testing")
            SUCCESS = True
        except Exception as exc:
            self.logger.error(f"The following error occurred in self_learn: {str(exc)[:100]}...")
            SUCCESS = False
        return SUCCESS

    def wafl_learn(self, five_digit_number_str: str) -> bool:
        """
        Implementation of the Epoch-by-Epoch Learning Process for WAFL-MLP.
        """
        self.logger.info(f"🎯 Beginning the WAFL-Learning Epoch: {five_digit_number_str}")
        SUCCESS = False
        try:
            neighbours = self.get_neighbour_list(five_digit_number_str)
            n_nbr = len(neighbours)
            local_model = copy.deepcopy(self.net.state_dict())
            for neighbour in neighbours:
                received_model = self.model_sharing.request_model_from_peer(
                    self.ctrl_tcp.get_device_ip(str(neighbour)), "&purpose=testing"
                )
                for key in self.net.state_dict():
                    model_difference = received_model[key] - self.net.state_dict()[key]
                    local_model[key] += model_difference * self.cFL_COEFFICIENCY / (n_nbr + 1)
            self.net.load_state_dict(local_model)
            SELF_LEARN_FLAG = self.self_learn(five_digit_number_str, WAFL_LEARN=True)
            if not SELF_LEARN_FLAG:
                raise Exception("SELF-LEARNING ERROR")
            self.logger.info(f"✅ Completed the WAFL-Learning Epoch: {five_digit_number_str}")
            SUCCESS = True
        except Exception as exc:
            self.logger.error(f"The following error occurred in wafl_learn: {str(exc)[:100]}...")
            SUCCESS = False
        return SUCCESS


class ModelSharingUtils:
    """
    A class for handling peer-to-peer WAFL model sharing.
    """

    cMDLREQ = "MDLREQ"

    def __init__(self, name: str, addr: str, port: int, timeout: float = 10.0) -> None:
        """
        Initialize the instance attributes.
        """
        self.vMODEL_INSTANCE = None
        self.vMODEL_INSTANCE_CACHE = None
        self.vMODEL_METADATA = ""
        self.fLISTENER_ACTIVE = True
        self.name = name
        self.addr = addr
        self.port = port
        self.timeout = timeout
        self.logger = logging.getLogger("ModelSharingUtils")
        self.logger.info("Initialized the Model Sharing Utils instance.")
        self.listener_thread = threading.Thread(target=self._socket_listener_thread, daemon=False, name="P2P_Listener")
        self.listener_thread.start()
        self.logger.info("🚀 Launched the P2P Transfer Thread.")

    def _serialize_model(self, LE_model: Any) -> bytes:
        """
        Serialize the WAFL model for sharing
        (from the last completed epoch).
        """
        self.logger.debug("🔢 Serializing the model for transfer.")
        try:
            serialized_output = pickle.dumps(LE_model)
            return serialized_output
        except Exception as exc:
            self.logger.error(f"The following error occurred: {str(exc)[:100]}...")
            return b"ERROR"

    def _deserialize_model(self, SR_model: bytes) -> Any:
        """
        De-serialize the received WAFL model
        (from the last completed epoch).
        """
        self.logger.debug("🔢 De-serializing the received model.")
        try:
            deserialized_output = pickle.loads(SR_model)
            return deserialized_output
        except Exception as exc:
            self.logger.error(f"The following error occurred: {str(exc)[:100]}...")
            return b"ERROR"

    def _compress_model(self, LE_model: bytes) -> bytes:
        """
        Lossless compression of the WAFL model for transfer.
        """
        self.logger.info("📦 Compressing the model for transfer.")
        try:
            compressed_output = zlib.compress(LE_model)
            original_size_megabytes = len(LE_model) / 1e6
            compressed_size_megabytes = len(compressed_output) / 1e6
            self.logger.debug(f"🗜️ Compressed from {original_size_megabytes:.2f}MB to {compressed_size_megabytes:.2f}MB.")
            return compressed_output
        except Exception as exc:
            self.logger.error(f"The following error occurred: {str(exc)[:100]}...")
            return b"ERROR"

    def _decompress_model(self, SR_Model: bytes) -> bytes:
        """
        De-compression of the received WAFL model using ZLib.
        """
        self.logger.debug("📦 De-compressing the received model.")
        try:
            decompressed_output = zlib.decompress(SR_Model)
            return decompressed_output
        except Exception as exc:
            self.logger.error(f"The following error occurred: {str(exc)[:100]}...")
            return b"ERROR"

    def _fetch_model(self, peer_IP: str, other_options: str = "") -> Tuple[bool, Any]:
        """
        Implementation of the Model Request (MDLREQ) command.
        Requests the specified peer device for model data.
        other_options attribute, if non-empty, should be prefixed by a '&' character.
        Format of parameters: &param1=val1&param2=val2...
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                self.logger.debug(f"📥 Requesting WAFL model from peer: {str(peer_IP)}")
                command = f"{ModelSharingUtils.cMDLREQ}:src={self.addr}{other_options}\r\n"
                s.settimeout(self.timeout)
                s.connect((peer_IP, self.port))
                s.sendall(command.encode("utf-8"))
                data = []
                while True:
                    packet = s.recv(4096)
                    if not packet:
                        break
                    data.append(packet)
            data = b"".join(data)
            data = self._decompress_model(data)
            data = self._deserialize_model(data)
            if data == b"ERROR" or data is None:
                raise Exception("FETCH ERROR")
            return True, data
        except Exception as exc:
            self.logger.error(f"The following error occurred in _fetch_model: {str(exc)[:100]}...")
            return False, b"ERROR"

    def _dispatch_model(self, conn: socket, options: str) -> bool:
        """
        Utility function for sending the model data to the peer.
        Depending on the WAFL project, the options parameter may
        determine the processing that takes place inside this
        function.
        """
        try:
            self.logger.debug("⏳ Preparing the WAFL model data to be dispatched.")
            self.logger.debug(f"🖨️ The received OPTIONS for dispatch: {options}")
            # OPTIONS-specific processing
            # code should be added here.
            # For now the entire model is dispatched.
            model_data = self.vMODEL_INSTANCE
            if self.vMODEL_INSTANCE_CACHE is None:
                model_data = self._serialize_model(model_data)
                model_data = self._compress_model(model_data)
                self.vMODEL_INSTANCE_CACHE = model_data
            else:
                model_data = self.vMODEL_INSTANCE_CACHE
            if model_data == b"ERROR":
                self.vMODEL_INSTANCE_CACHE = None
                raise Exception("DISPATCH ERROR")
            conn.sendall(model_data)
            self.logger.debug("✅ Successfully sent the model data to the peer.")
            return True
        except Exception as exc:
            self.logger.error(f"The following error occurred in _dispatch_model: {str(exc)[:100]}...")
            return False

    def _socket_listener_thread(self) -> None:
        """
        Implemenation of the Peer-to-Peer Sharing Listener Thread.
        Will run as a non-daemon thread for processing MDLREQ requests.
        Will pass on the received OPTIONS to the dispatch model utility.
        Will be run from the __init__() function.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.addr, self.port))
            s.settimeout(self.timeout)
            s.listen()
            self.logger.info(f"🔗 Socket bound at {self.addr}:{self.port} and listening.")
            while self.fLISTENER_ACTIVE:
                try:
                    conn, addr_info = s.accept()
                    conn.settimeout(self.timeout)
                    self.logger.info(f"📶 Connection Established with {addr_info[0]}:{addr_info[1]}.")
                    data = []
                    while True:
                        packet = conn.recv(4096)
                        if not packet:
                            break
                        data.append(packet)
                        if packet[-2:] == b"\r\n":
                            break
                    data = b"".join(data).decode("utf-8").strip()
                    self.logger.debug(f"'{data}' command received from peer.")
                    command, options = data.split(":")
                    if command != ModelSharingUtils.cMDLREQ:
                        raise Exception("COMMAND MISMATCH")
                    self.logger.info("📡 Dispatching Model Data to the WAFL Peer.")
                    DISPATCHED = self._dispatch_model(conn, options)
                    if not DISPATCHED:
                        raise Exception("NOT DISPATCHED")
                    conn.close()
                except socket.timeout:
                    # This is expected to happen when no connection is made within the timeout.
                    # It allows the while loop to check the fLISTENER_ACTIVE flag.
                    continue
                except Exception as exc:
                    # Avoid logging minor errors that could spam the log.
                    if self.fLISTENER_ACTIVE:
                        self.logger.error(f"The following error occurred in _socket_listener_thread: {str(exc)[:100]}...")
                    time.sleep(1.0)
            self.logger.info("P2P listener thread has been terminated.")

    def update_model_instance(self, LE_model: Any, metadata: str = "") -> None:
        """
        Updates the WAFL model instance that is
        to be dispatched.
        """
        self.vMODEL_INSTANCE = copy.deepcopy(LE_model)
        self.vMODEL_INSTANCE_CACHE = None
        self.vMODEL_METADATA = metadata

    def request_model_from_peer(self, peer_IP: str, other_options: str = "") -> Any:
        """
        The wrapper function for retrieving model parameters from a WAFL peer device.
        options attribute, if non-empty, should be prefixed by a '&' character.
        Format of options: &param1=val1&param2=val2...
        Keeps requesing for parameters until they are retrieved successfully.
        Uses Exponential Backoff Mechanism for waiting between retries.
        """
        FETCHED = False
        WAIT_TIME = 2.0
        GROWTH_FACTOR = 1.5
        while not FETCHED and self.fLISTENER_ACTIVE:
            FETCHED, model_data = self._fetch_model(peer_IP, other_options)
            if FETCHED:
                self.logger.info(f"✅ Retrieved model parameters from peer: {str(peer_IP)}")
                return model_data
            time.sleep(WAIT_TIME)
            WAIT_TIME **= GROWTH_FACTOR
        # Handling KILL command from Control Server.
        return "Fetch Operation Terminated Abruptly."


class CTRL_TCP:
    """
    A class for handling the TCP connection between ctrl server.
    """

    def __init__(self, config_path: str):
        """
        Initialize the TCP connection parameters.
        """
        self.logger = logging.getLogger("CTRL_TCP")
        self.device_names: Optional[List[str]] = None
        self.device_ips: Optional[List[str]] = None
        self.name: Optional[str] = None
        self.addr: Optional[str] = None
        self.ctrl_port: Optional[int] = None
        self.p2p_port: Optional[int] = None
        self.timeout: Optional[float] = None

        # Flag to control the main listener loop.
        self.fLISTENER_ACTIVE = True

        # Variables for tracking the current status.
        self.is_epoch_running: bool = False
        self.current_epoch_type: Optional[str] = None  # "SELF" or "WAFL"
        self.current_epoch_number: Optional[str] = None  # 5-digit string
        self.status_logs: List[str] = []

        # Variable to hold the learning thread object.
        self.learning_thread: Optional[threading.Thread] = None

        self._load_config(config_path)
        self.ctrl_listener_thread = threading.Thread(target=self.wait_ctrl, daemon=False, name="CTRL_TCP_Listener")
        self.ctrl_listener_thread.start()
        self.logger.info("Initialized the CTRL_TCP instance with configuration parameters.")
        self.setup_local_wafl_node(self.name)

    def _load_config(self, config_path: str) -> bool:
        """
        Loads configuration information from the specified JSON file
        and stores it in member variables.
        Returns:
            bool: True if the configuration was successfully loaded and stored, False otherwise.
        """
        if not os.path.exists(config_path):
            self.logger.error(f"Specified file not found: {config_path}")
            return False

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            self.logger.info(f"Configuration file '{config_path}' loaded.")
            wafl_devices_data = config_data.get("infrastructure")
            if not isinstance(wafl_devices_data, dict):
                self.logger.error("Invalid format for 'wafl_devices' in JSON. Dictionary required.")
                return False
            self.device_names = wafl_devices_data.get("device_names")
            if not (isinstance(self.device_names, list) and all(isinstance(n, str) for n in self.device_names)):
                self.logger.error("Invalid format for 'wafl_devices.name' in JSON. List of integers required.")
                return False
            self.device_ips = wafl_devices_data.get("device_ips")
            if not (isinstance(self.device_ips, list) and all(isinstance(ip, str) for ip in self.device_ips)):
                self.logger.error("Invalid format for 'wafl_devices.ip' in JSON. List of strings required.")
                return False
            agent_info = config_data.get("agent_info")
            if not isinstance(agent_info, dict):
                self.logger.error("Invalid format for 'agent_info' in JSON. Dictionary required.")
                return False
            self.name = agent_info.get("device_name")
            if not isinstance(self.name, str):
                self.logger.error("Invalid format for 'wafl_devices.self_name' in JSON. Integer required.")
                return False
            self.addr = agent_info.get("ip_address")
            if not isinstance(self.addr, str):
                self.logger.error("Invalid format for 'wafl_devices.addr' in JSON. String required.")
                return False
            self.ctrl_port = wafl_devices_data.get("ctrl_port")
            if not isinstance(self.ctrl_port, int):
                self.logger.error("Invalid format for 'wafl_devices.ctrl_port' in JSON. Integer required.")
                return False
            self.p2p_port = wafl_devices_data.get("p2p_port")
            if not isinstance(self.p2p_port, int):
                self.logger.error("Invalid format for 'wafl_devices.p2p_port' in JSON. Integer required.")
                return False
            self.timeout = 10.0  # dummy
            if not isinstance(self.timeout, float):
                self.logger.error("Invalid format for 'wafl_devices.timeout' in JSON. Float required.")
                return False
            self.logger.info("All configurations successfully stored in member variables.")
            return True

        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON file: {e}")
            return False
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while loading configuration: {e}")
            return False

    def get_device_ip(self, device_name: str) -> Optional[str]:
        """
        Returns the IP address corresponding to the given device name.
        Returns:
            Optional[str]: The corresponding IP address string, or None if not found.
        """
        try:
            index = self.device_names.index(device_name)
            return self.device_ips[index]
        except ValueError:
            self.logger.warning(f"Device name {device_name} not found.")
            return None
        except IndexError:
            self.logger.warning(
                f"Index for device name {device_name} is out of bounds for the IP list. Data inconsistency might exist."
            )
            return None

    def setup_local_wafl_node(self, device_name: str) -> None:
        """
        Sets up the node with the given device name.
        This function should be called before starting the WAFL model training.
        """
        self.logger.info(f"Setting up the node with device name: {device_name}")
        self.model_sharing = ModelSharingUtils(device_name, self.get_device_ip(device_name), self.p2p_port, 10.0)
        self.model_learning = ModelLearningUtils(
            "./dataset/dataset.pickled",
            "./results/model_instance.pth",
            "./config/contact_pattern.json",
            self.model_sharing,
            self,
        )

    def _receive_command(self, conn: socket.socket) -> Optional[str]:
        """
        Receives a command string from the client connection.
        Returns:
            Optional[str]: The decoded and stripped command string, or None if an error occurs
                           or connection is closed.
        """
        data_buffer = []
        try:
            while True:
                packet = conn.recv(4096)
                if not packet:
                    self.logger.info("Client closed connection.")
                    break
                data_buffer.append(packet)
                if packet.endswith(b"\r\n"):
                    break

            received_command = b"".join(data_buffer).decode("utf-8").strip()
            self.logger.info(f"Received raw command: '{received_command}'")
            return received_command
        except socket.timeout:
            self.logger.warning("Socket receive timed out while waiting for command.")
            return None
        except UnicodeDecodeError:
            self.logger.error("Failed to decode received data as UTF-8. Invalid data format.")
            return None
        except Exception as exc:
            self.logger.error(f"An unexpected error occurred during command reception: {str(exc)[:100]}...")
            return None

    def wait_ctrl(self) -> None:
        """
        Waits for the control server to be ready and processes incoming commands.
        This function should be called before starting the WAFL model training.
        """
        self.logger.info("Waiting for the control server to be ready...")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.addr, self.ctrl_port))
            s.listen()
            self.logger.info(f"🔗 Socket bound at {self.addr}:{self.ctrl_port} and listening.")

            while self.fLISTENER_ACTIVE:
                try:
                    conn, addr_info = s.accept()
                    conn.settimeout(self.timeout)
                    self.logger.info(f"📶 Connection Established with {addr_info[0]}:{addr_info[1]}.")

                    received_command = self._receive_command(conn)

                    if received_command:
                        if not self._process_command(received_command, conn):
                            self.logger.warning(f"Failed to process command: '{received_command}'")
                    else:
                        self.logger.warning("No valid command received from client or reception failed.")

                    conn.close()
                    self.logger.info(f"Connection with {addr_info[0]}:{addr_info[1]} closed.")

                except socket.timeout:
                    # This is expected behavior to allow the loop to check fLISTENER_ACTIVE.
                    continue
                except Exception as exc:
                    if self.fLISTENER_ACTIVE:
                        self.logger.error(f"An error occurred in the socket listener: {str(exc)[:100]}...")

            self.logger.info("Control listener thread has been terminated.")

    def _process_command(self, command_str: str, conn: socket.socket) -> bool:
        """
        Parses the command string and dispatches to appropriate functions.
        Returns:
            bool: True if the command was successfully processed, False otherwise.
        """
        self.logger.info(f"Processing command: '{command_str}'")
        parts = command_str.split("-")

        main_command = parts[0] if len(parts) > 0 else ""

        try:
            if main_command == "KILL":
                if len(parts) != 1:
                    self.logger.warning(f"Incorrect form of KILL command: {command_str}")
                    conn.sendall("ERROR\r\n".encode("utf-8"))
                    return False
                # Handle the KILL command by setting flags to terminate loops.
                self.logger.info("KILL command received. Shutting down listeners and ongoing tasks.")
                self.fLISTENER_ACTIVE = False
                self.model_sharing.fLISTENER_ACTIVE = False
                threads_to_join = []
                if hasattr(self.model_sharing, "listener_thread"):
                    threads_to_join.append(self.model_sharing.listener_thread)
                if self.learning_thread and self.learning_thread.is_alive():
                    threads_to_join.append(self.learning_thread)

                # Wait for threads to finish, with a timeout.
                thread_joined = 0
                for thread in threads_to_join:
                    self.logger.info(f"Waiting for thread {thread.name} to terminate...")
                    thread.join(timeout=self.timeout)
                    if thread.is_alive():
                        self.logger.warning(f"Thread {thread.name} did not terminate in time.")
                    else:
                        thread_joined += 1
                        self.logger.info(f"Thread {thread.name} has terminated successfully.")

                if thread_joined == len(threads_to_join):
                    self.logger.info("All stoppable threads have been processed.")
                    conn.sendall("OK\r\n".encode("utf-8"))
                    return True
                else:
                    self.logger.warning("Not all threads could be stopped. Some might still be running.")
                    conn.sendall("ERROR\r\n".encode("utf-8"))
                    return False

            elif main_command == "STAT":
                if len(parts) != 1:
                    self.logger.warning(f"Incorrect form of STAT command: {command_str}")
                    conn.sendall("ERROR\r\n".encode("utf-8"))
                    return False
                status_response = self._get_status()
                conn.sendall(status_response.encode("utf-8"))
                return True

            elif main_command == "BEGIN":
                if len(parts) != 3:
                    self.logger.warning(f"Incorrect form of BEGIN command: {command_str}. Expected BEGIN-TYPE-NUMBER.")
                    conn.sendall("ERROR\r\n".encode("utf-8"))
                    return False
                if self.learning_thread and self.learning_thread.is_alive():
                    self.logger.warning("Cannot start new learning task. A task is already running.")
                    conn.sendall("ERROR\r\n".encode("utf-8"))
                    return False

                sub_command = parts[1]
                five_digit_number_str = parts[2]

                if not (five_digit_number_str.isdigit() and len(five_digit_number_str) == 5):
                    self.logger.warning(f"Invalid five-digit number format in BEGIN command: {command_str}")
                    conn.sendall("ERROR\r\n".encode("utf-8"))
                    return False

                if sub_command == "SELF":
                    try:
                        self.learning_thread = threading.Thread(
                            target=self._self_learn,
                            daemon=False,
                            args=(five_digit_number_str,),
                            name=f"SelfLearn-{five_digit_number_str}",
                        )
                        self.learning_thread.start()
                        conn.sendall("OK\r\n".encode("utf-8"))
                        return True
                    except Exception as e:
                        self.logger.error(f"Failed to start self learning thread: {e}")
                        conn.sendall("ERROR\r\n".encode("utf-8"))
                        return False
                elif sub_command == "WAFL":
                    try:
                        self.learning_thread = threading.Thread(
                            target=self._wafl_learn,
                            daemon=False,
                            args=(five_digit_number_str,),
                            name=f"WAFL_Learn_{five_digit_number_str}",
                        )
                        self.learning_thread.start()
                        conn.sendall("OK\r\n".encode("utf-8"))
                    except Exception as e:
                        self.logger.error(f"Failed to start WAFL learning thread: {e}")
                        conn.sendall("ERROR\r\n".encode("utf-8"))
                        return False
                else:
                    self.logger.warning(f"Unknown BEGIN subcommand: {sub_command} in command: {command_str}")
                    conn.sendall("ERROR\r\n".encode("utf-8"))
                    return False

            else:
                self.logger.warning(f"Unknown command received: {command_str}")
                conn.sendall("ERROR\r\n".encode("utf-8"))
                return False
        except Exception as e:
            self.logger.error(f"Error during command processing for '{command_str}': {e}", exc_info=True)
            conn.sendall("ERROR\r\n".encode("utf-8"))
            return False

    def _get_status(self) -> str:
        """
        Constructs the status response string based on the current state.
        Format: EXEC-XXXX-YYYYY-Z or DONE-XXXX-YYYYY-Z
        """
        current_logs = self.status_logs.copy()
        if self.learning_thread:
            thread_status = "Active" if self.learning_thread.is_alive() else "Finished"
            current_logs.append(f"Learning Thread Status: {thread_status}")
        if self.current_epoch_type is None or self.current_epoch_number is None:
            # Handle case where no epoch has run yet.
            return "DONE-NONE--1-0\r\n"

        log_line_count = len(self.status_logs)

        if self.is_epoch_running or (self.learning_thread and self.learning_thread.is_alive()):
            # Format: EXEC-XXXX-YYYYY-Z
            header = f"EXEC-{self.current_epoch_type}-{self.current_epoch_number}-{log_line_count}"
        else:
            # Format: DONE-XXXX-YYYYY-Z
            header = f"DONE-{self.current_epoch_type}-{self.current_epoch_number}-{log_line_count}"

        # Combine header and logs
        logs = "\n".join(current_logs)
        response = f"{header}\n{logs}\r\n"
        return f"{response}"

    def _self_learn(self, five_digit_number_str: str) -> bool:
        """
        Handles the BEGIN-SELF command.
        """
        self.logger.info(f"Start self learning epoch: {five_digit_number_str}")

        # --- Update status at the beginning of the epoch ---
        self.is_epoch_running = True
        self.current_epoch_type = "SELF"
        self.current_epoch_number = five_digit_number_str
        self.status_logs = [f"Log for {self.current_epoch_type} epoch {self.current_epoch_number} started at {time.ctime()}"]
        # ---

        FLAG = self.model_learning.self_learn(five_digit_number_str)
        # --- Update status at the end of the epoch ---
        self.status_logs.append(f"Epoch completion successful: {FLAG} at {time.ctime()}")
        self.is_epoch_running = False
        # ---
        return FLAG

    def _wafl_learn(self, five_digit_number_str: str) -> bool:
        """
        Handles the BEGIN-WAFL command.
        """
        self.logger.info(f"Start WAFL learning epoch: {five_digit_number_str}")

        # --- Update status at the beginning of the epoch ---
        self.is_epoch_running = True
        self.current_epoch_type = "WAFL"
        self.current_epoch_number = five_digit_number_str
        self.status_logs = [f"Log for {self.current_epoch_type} epoch {self.current_epoch_number} started at {time.ctime()}"]
        # ---

        # Implement the logic for WAFL learning here
        FLAG = self.model_learning.wafl_learn(five_digit_number_str)

        # --- Update status at the end of the epoch ---
        self.status_logs.append(f"Epoch completion successful: {FLAG} at {time.ctime()}")
        self.is_epoch_running = False
        # ---
        return FLAG


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    comm_interface = CTRL_TCP("./config/config.json")

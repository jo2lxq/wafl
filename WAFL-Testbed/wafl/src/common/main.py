import copy
import json
import logging
import os
import pickle
import socket
import threading
import time
import zlib
from typing import Any, Dict, List, Optional, Tuple


# Class for supplying Dummy Data for Local Testing.
# Should be removed once the work on the file is completed.
class DUMMY:
    IP_ADDR = "127.0.0.1"
    MODEL_DATA = pickle.load(open("DEMODATA.47MB", "rb"))


class ModelSharingUtils:
    """
    A class for handling peer-to-peer WAFL model sharing.
    """

    cMDLREQ = "MDLREQ"

    def __init__(self, addr: str, port: int, timeout: float = 10.0) -> None:
        """
        Initialize the instance attributes.
        """
        self.vMODEL_INSTANCE = None
        self.vMODEL_INSTANCE_CACHE = None
        self.vMODEL_METADATA = ""
        self.fLISTENER_ACTIVE = True
        self.addr = addr
        self.port = port
        self.timeout = timeout
        self.logger = logging.getLogger("ModelSharingUtils")
        self.logger.info("Initialized the Model Sharing Utils instance.")
        threading.Thread(target=self._socket_listener_thread, daemon=False, args=[]).start()
        self.logger.info("🚀 Launched the P2P Transfer Thread.")

    def _serialize_model(self, LE_model: Any) -> bytes:
        """
        Serialize the WAFL model for sharing
        (from the last completed epoch).
        """
        self.logger.info("🔢 Serializing the model for transfer.")
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
        self.logger.info("🔢 De-serializing the received model.")
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
            self.logger.info(f"🗜️ Compressed from {original_size_megabytes:.2f}MB to {compressed_size_megabytes:.2f}MB.")
            return compressed_output
        except Exception as exc:
            self.logger.error(f"The following error occurred: {str(exc)[:100]}...")
            return b"ERROR"

    def _decompress_model(self, SR_Model: bytes) -> bytes:
        """
        De-compression of the received WAFL model using ZLib.
        """
        self.logger.info("📦 De-compressing the received model.")
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
                self.logger.info(f"📥 Requesting WAFL model from peer: {str(peer_IP)}")
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
            self.logger.error(f"The following error occurred: {str(exc)[:100]}...")
            return False, b"ERROR"

    def _dispatch_model(self, conn: socket, options: str) -> bool:
        """
        Utility function for sending the model data to the peer.
        Depending on the WAFL project, the options parameter may
        determine the processing that takes place inside this
        function.
        """
        try:
            self.logger.info("⏳ Preparing the WAFL model data to be dispatched.")
            self.logger.info(f"🖨️ The received OPTIONS for dispatch: {options}")
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
            self.logger.info("✅ Successfully sent the model data to the peer.")
            return True
        except Exception as exc:
            self.logger.error(f"The following error occurred: {str(exc)[:100]}...")
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
                except Exception as exc:
                    self.logger.error(f"The following error occurred: {str(exc)[:100]}...")
                    time.sleep(1.0)

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
        while not FETCHED:
            FETCHED, model_data = self._fetch_model(peer_IP, other_options)
            if FETCHED:
                self.logger.info(f"✅ Retrieved model parameters from peer: {str(peer_IP)}")
                return model_data
            time.sleep(WAIT_TIME)
            WAIT_TIME **= GROWTH_FACTOR


class CTRL_TCP:
    """
    A class for handling the TCP connection between ctrl server.
    """

    # Work in Progress.
    def __init__(self, config_path: str):
        """
        Initialize the TCP connection parameters.
        """
        self.logger = logging.getLogger("CTRL_TCP")
        self.ctrl_server_ip: Optional[str] = None
        self.device_names: Optional[List[int]] = None
        self.device_ips: Optional[List[str]] = None
        self.name: Optional[int] = None
        self.addr: Optional[str] = None
        self.ctrl_port: Optional[int] = None
        self.p2p_port: Optional[int] = None
        self.timeout: Optional[float] = None
        self._load_config(config_path)
        threading.Thread(target=self.wait_ctrl, daemon=False, args=[]).start()
        self.logger.info("Initialized the CTRL_TCP instance with configuration parameters.")
        self.model_sharing = ModelSharingUtils(DUMMY.IP_ADDR, self.p2p_port, self.timeout)

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
            self.ctrl_server_ip = config_data.get("ctrl_server_ip")
            if not isinstance(self.ctrl_server_ip, str):
                self.logger.error("Invalid format for 'ctrl_server_ip' in JSON. String required.")
                return False
            wafl_devices_data = config_data.get("wafl_devices")
            if not isinstance(wafl_devices_data, dict):
                self.logger.error("Invalid format for 'wafl_devices' in JSON. Dictionary required.")
                return False
            self.device_names = wafl_devices_data.get("name_list")
            if not (isinstance(self.device_names, list) and all(isinstance(n, int) for n in self.device_names)):
                self.logger.error("Invalid format for 'wafl_devices.name' in JSON. List of integers required.")
                return False
            self.device_ips = wafl_devices_data.get("ip_list")
            if not (isinstance(self.device_ips, list) and all(isinstance(ip, str) for ip in self.device_ips)):
                self.logger.error("Invalid format for 'wafl_devices.ip' in JSON. List of strings required.")
                return False
            self.name = wafl_devices_data.get("self_name")
            if not isinstance(self.name, int):
                self.logger.error("Invalid format for 'wafl_devices.self_name' in JSON. Integer required.")
                return False
            self.addr = wafl_devices_data.get("self_addr")
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
            self.timeout = wafl_devices_data.get("timeout")
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

    def get_device_ip(self, device_name: int) -> Optional[str]:
        """
        Returns the IP address corresponding to the given device name.
        Returns:
            Optional[str]: The corresponding IP address string, or None if not found.
        """
        try:
            index = self.wafl_device_names.index(device_name)
            return self.wafl_device_ips[index]
        except ValueError:
            self.logger.warning(f"Device name {device_name} not found.")
            return None
        except IndexError:
            self.logger.warning(
                f"Index for device name {device_name} is out of bounds for the IP list. Data inconsistency might exist."
            )
            return None

    def setup_local_wafl_node(self, device_name: int) -> None:
        """
        Sets up the node with the given device name.
        This function should be called before starting the WAFL model training.
        """
        self.logger.info(f"Setting up the node with device name: {device_name}")
        local_node = ModelSharingUtils(addr=self.get_device_ip(device_name), port=self.wafl_p2p_port, timeout=10)
        return local_node

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
            self.logger.info(f"🔗 Socket bound at {self.addr}:{self.ctrl_port} and listening.")  # ctrl_portに修正

            while True:
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
                    self.logger.warning("No incoming connection within the timeout period. Retrying...")
                except Exception as exc:
                    self.logger.error(f"An error occurred in the socket listener: {str(exc)[:100]}...")

    def _process_command(self, command_str: str, conn: socket.socket) -> bool:
        """
        Parses the command string and dispatches to appropriate functions.
        Returns:
            bool: True if the command was successfully processed, False otherwise.
        """
        self.logger.info(f"Processing command: '{command_str}'")
        parts = command_str.split("-")

        main_command = parts[0] if len(parts) > 0 else ""

        if main_command == "KILL":
            if len(parts) != 1:
                self.logger.warning(f"Incorrect form of KILL command: {command_str}")
                return False
            return self._kill_process()
        elif main_command == "STAT":
            if len(parts) != 1:
                self.logger.warning(f"Incorrect form of STAT command: {command_str}")
                return False
            status_response = self._get_status()
            conn.sendall(status_response.encode("utf-8"))
            return True
        elif main_command == "BEGIN":
            if len(parts) != 3:
                self.logger.warning(f"Incorrect form of BEGIN command: {command_str}")
                return False

            sub_command = parts[1]
            five_digit_number_str = parts[2]

            if not (len(five_digit_number_str) == 5 and five_digit_number_str.isdigit()):
                self.logger.warning(f"BEGIN command's number must be a 5-digit number string: {command_str}")
                return False

            if sub_command == "SELF":
                return self._handle_begin_self(five_digit_number_str)
            elif sub_command == "WAFL":
                return self._handle_begin_wafl(five_digit_number_str)
            else:
                self.logger.warning(f"Unknown BEGIN subcommand: {sub_command} in command: {command_str}")
                return False
        else:
            self.logger.warning(f"Unknown command received: {command_str}")
            return False

    def _kill_process(self) -> bool:
        pass

    def _get_status(self) -> Dict[str, Any]:
        pass

    def _self_learn(self, five_digit_number_str: str) -> bool:
        """
        Handles the BEGIN-SELF command.
        """
        self.logger.info(f"Start self learning epoch: {five_digit_number_str}")
        # Implement the logic for self-learning here
        pass

    def _wafl_learn(self, five_digit_number_str: str) -> bool:
        """
        Handles the BEGIN-WAFL command.
        """
        self.logger.info(f"Start WAFL learning epoch: {five_digit_number_str}")
        # Implement the logic for WAFL learning here
        # Dummy Test Operations BEGIN
        self.model_sharing.update_model_instance(DUMMY.MODEL_DATA, "Testing")
        MODEL = self.model_sharing.request_model_from_peer(DUMMY.IP_ADDR, "&purpose=testing")
        print(f"Model Received: {type(MODEL)} | Length: {len(pickle.dumps(MODEL)) / 1e6:.2f}MB")
        # Dummy Test Operations END


if __name__ == "__main__":
    # Testing the Module
    logging.basicConfig(level=logging.DEBUG)
    comm_interface = CTRL_TCP("config.json")
    comm_interface._wafl_learn("00000")
    comm_interface.model_sharing.fLISTENER_ACTIVE = False

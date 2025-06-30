import copy
import logging
import pickle
import socket
import threading
import time
import zlib
from typing import Any, Tuple

logging.basicConfig(level=logging.DEBUG)
# This is a dummy class.
# I have created it for specifying the current
# component's parameters. It should be removed
# once the entire file is ready.


class GLOBAL_PARAMETERS:
    IP_ADDR = "127.0.0.1"
    IP_PORT = 10002
    LAST_EPOCH_MODEL_INSTANCE = "MODEL DATA, COULD BE AN INSTANCE OF ANY CLASS"


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


if __name__ == "__main__":
    # Testing the Module
    model_sharing = ModelSharingUtils(GLOBAL_PARAMETERS.IP_ADDR, GLOBAL_PARAMETERS.IP_PORT, 10.0)
    model_sharing.update_model_instance(GLOBAL_PARAMETERS.LAST_EPOCH_MODEL_INSTANCE, "Testing")
    MODEL = model_sharing.request_model_from_peer(GLOBAL_PARAMETERS.IP_ADDR, "&purpose=testing")
    model_sharing.fLISTENER_ACTIVE = False
    print(f"Model Received: {type(MODEL)} | Length: {len(pickle.dumps(MODEL)) / 1e6:.2f}MB")

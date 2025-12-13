from __future__ import annotations

import copy
import csv
import json
import logging
import os
import pickle
import socket
import threading
import time
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from .compression_manager import CompressionManager
from .logger import MetricsLogger
from .net import Net
from .udp_model_sharing import UDPModelSharing


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

    def __init__(
        self,
        dataset_path: str,
        model_instance_path: str,
        contact_pattern_path: str,
        model_sharing: ModelSharingUtils,
        ctrl_tcp: CTRL_TCP,
        experiment_id,
        wafl_phase_params: dict,
    ) -> None:
        """
        Initialize the learning process instance.
        """
        torch.random.manual_seed(1)
        train_path = os.path.join(dataset_path, "train/train.pkl")
        test_path = os.path.join(dataset_path, "test/test.pkl")
        with open(train_path, "rb") as f:
            train_dataset = pickle.load(f)
        with open(test_path, "rb") as f:
            test_dataset = pickle.load(f)
        self.model_sharing = model_sharing
        self.ctrl_tcp = ctrl_tcp
        self.wafl_phase_params = wafl_phase_params
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.wafl_phase_params["batch_size"],
            shuffle=False,
            num_workers=2,
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.wafl_phase_params["batch_size"],
            shuffle=False,
            num_workers=2,
        )
        self.logger = logging.getLogger("ModelLearningUtils")
        self.model_instance_path = model_instance_path
        with open(contact_pattern_path, "r") as f:
            self.neighbour_map = json.load(f)
        if not isinstance(self.neighbour_map, list) or len(self.neighbour_map) == 0:
            self.logger.error("Contact pattern must be a non-empty list")
        else:
            self.logger.info(f"Contact pattern loaded: {len(self.neighbour_map)} epochs")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = Net().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.wafl_phase_params["learning_rate"])
        self.experiment_id = experiment_id
        # Metrics logger is initialized in WaflAgent but we need access to it or create one
        # Actually ModelLearningUtils is initialized inside WaflAgent (in ctrl/main.py) or used by it?
        # Wait, this is running on the node.
        # The MetricsLogger is passed or initialized?
        # In the original code, MetricsLogger was initialized in CTRL_TCP or similar?
        # Let's check where MetricsLogger is instantiated.
        # It seems it was instantiated in CTRL_TCP.__init__ in the previous `view_file` of main.py (lines 670+).
        # But ModelLearningUtils is independent.
        # We should probably initialize MetricsLogger here if we want to use it, OR pass it.
        # However, to keep it simple and avoid changing signatures too much, let's instantiate it here.
        from .logger import MetricsLogger

        self.metrics_logger = MetricsLogger(experiment_id, "node")  # node name is not used in filename

        self.logger.debug("Initialized Model Learning Utils")

        # Log wafl_phase parameters
        self.logger.info("📊 WAFL Phase Parameters Applied:")
        self.logger.info(f"   - Aggregation Strategy: {self.wafl_phase_params.get('aggregation_strategy', 'FedAvg')}")
        self.logger.info(f"   - Batch Size: {self.wafl_phase_params['batch_size']}")
        self.logger.info(f"   - Learning Rate: {self.wafl_phase_params['learning_rate']}")
        self.logger.info(f"   - Coefficiency: {self.wafl_phase_params['coefficiency']}")

        self.train_loss = 0
        self.train_accuracy = 0
        self.epoch_number = 0
        self.stop_requested = False

        # SSP metrics tracking
        self.epoch_start_time = None
        self.batches_processed = 0
        self.current_gradient_norm = 0.0

    def stop_current_epoch(self) -> dict:
        """Stops the current learning epoch immediately and returns wasted metrics.

        Returns:
            dict: Contains 'wasted_ms' and 'wasted_norm' for SSP logging
        """
        self.stop_requested = True
        self.logger.info("🛑 Stop requested for current epoch (SSP Reset).")

        # Calculate wasted metrics
        wasted_ms = 0.0
        if self.epoch_start_time is not None:
            wasted_ms = (time.time() - self.epoch_start_time) * 1000

        wasted_metrics = {
            "wasted_ms": wasted_ms,
            "wasted_norm": self.current_gradient_norm,
            "batches_processed": self.batches_processed,
        }

        self.logger.info(f"⚠️ SSP Reset: Wasted {wasted_ms:.2f}ms, {self.batches_processed} batches, gradient_norm={self.current_gradient_norm:.6f}")
        return wasted_metrics

    def get_neighbour_list(self, five_digit_number_str: str = "99999") -> List[int]:
        """
        Return the neighbour list for the current epoch from the contact pattern file.
        """
        neighbour_list = []
        try:
            neighbour_list = self.neighbour_map[int(five_digit_number_str)].get(str(self.model_sharing.agent_index), [])
        except Exception as exc:
            self.logger.error(f"Error retrieving neighbour list: {str(exc)[:100]}...")
        return neighbour_list

    def self_learn(self, five_digit_number_str: str = "99999", WAFL_LEARN=False) -> bool:
        """
        Implementation of the Self-Learning Epoch Logic for WAFL-MLP.
        """
        phase_name = "WAFL" if WAFL_LEARN else "SELF"
        if not WAFL_LEARN:
            self.logger.debug(f"Training [SELF] - Epoch {five_digit_number_str} started")
        SUCCESS = False

        # Reset SSP tracking
        self.stop_requested = False
        self.epoch_start_time = time.time()
        self.batches_processed = 0
        self.current_gradient_norm = 0.0

        try:
            running_loss = 0.0
            total_loss = 0.0
            num_batches = 0
            correct_train = 0
            total_train = 0
            for i, data in enumerate(self.train_loader, 0):
                x_train, y_train = data
                x_train = x_train.to(self.device)
                y_train = y_train.to(self.device)
                self.optimizer.zero_grad()
                y_output = self.net(x_train)
                loss = self.criterion(y_output, y_train)
                loss.backward()

                # Track gradient norm for SSP wasted computation
                total_norm = 0.0
                for p in self.net.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                self.current_gradient_norm = total_norm**0.5

                self.optimizer.step()
                batch_loss = loss.item()
                running_loss += batch_loss
                total_loss += batch_loss
                num_batches += 1
                self.batches_processed = num_batches
                _, predicted = torch.max(y_output.data, 1)
                total_train += y_train.size(0)
                correct_train += (predicted == y_train).sum().item()
                if (i + 1) % 50 == 0:
                    self.logger.debug(f"Running Loss: {running_loss / 50:.5f}")
                    running_loss = 0.0

                if self.stop_requested:
                    self.logger.warning("⚠️ Epoch interrupted by stop request (SSP Reset).")
                    return False
            self.train_loss = total_loss / num_batches if num_batches > 0 else 0.0
            self.train_accuracy = correct_train / total_train if total_train > 0 else 0.0

            # Always evaluate the model to get test metrics
            test_loss, test_accuracy = self._evaluate_model()

            if not WAFL_LEARN:
                self.logger.info(f"🏁 Completed the Self-Learning Epoch: {five_digit_number_str}")

            self.logger.info(f"📉 Training Loss: {self.train_loss:.6f}")
            self.logger.info(f"📈 Training Accuracy: {self.train_accuracy:.4f}")
            self.logger.info(f"📉 Test Loss: {test_loss:.4f}")
            self.logger.info(f"📈 Test Accuracy: {test_accuracy:.4f}")

            # Log metrics to CSV
            epoch_duration_ms = (time.time() - self.epoch_start_time) * 1000 if self.epoch_start_time else 0.0
            metrics = {
                "train_loss": self.train_loss,
                "train_accuracy": self.train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "epoch_duration_ms": epoch_duration_ms,
                # SSP metrics (no waste if epoch completed normally)
                "batches_processed": self.batches_processed,
            }
            # Add communication and compression metrics
            comm_metrics = self.model_sharing.get_epoch_metrics()
            metrics.update(comm_metrics)

            self.ctrl_tcp.metrics_logger.log_epoch(phase_name, self.epoch_number + 1, metrics)

            self.epoch_number += 1
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
                peer_ip = self.ctrl_tcp.get_device_ip(neighbour)
                if peer_ip is None:
                    self.logger.error(f"Could not get IP for neighbour {neighbour}")
                    continue
                received_model = self.model_sharing.request_model_from_peer(peer_ip, "&purpose=testing")
                if isinstance(received_model, str):
                    self.logger.error(f"Failed to receive model from {neighbour}")
                    continue
                for key in self.net.state_dict():
                    model_difference = received_model[key] - self.net.state_dict()[key]
                    local_model[key] += model_difference * self.wafl_phase_params["coefficiency"] / (n_nbr + 1)
            self.net.load_state_dict(local_model)

            # Always call self_learn to continue training, regardless of neighbor count
            if n_nbr:
                self.logger.info(f"📡 Exchanged models with {n_nbr} neighbors")
            else:
                self.logger.info("No neighbors available for model exchange in this epoch")

            # Call self_learn with WAFL_LEARN=True to continue training
            SELF_LEARN_FLAG = self.self_learn(five_digit_number_str, WAFL_LEARN=True)

            if not SELF_LEARN_FLAG:
                raise Exception("SELF-LEARNING ERROR")
            self.logger.info(f"✅ Completed the WAFL-Learning Epoch: {five_digit_number_str}")
            SUCCESS = True
        except Exception as exc:
            self.logger.error(f"The following error occurred in wafl_learn: {str(exc)[:100]}...")
            SUCCESS = False
        return SUCCESS

    def _evaluate_model(self) -> Tuple[float, float]:
        """
        Evaluate model on test dataset and return test loss and accuracy.
        """
        self.net.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.net(images)
                loss = self.criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                total_loss += loss.item()
                num_batches += 1
        self.net.train()
        accuracy = correct / total
        avg_test_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.logger.debug(f"Model evaluation - Test loss: {avg_test_loss:.6f}")
        return avg_test_loss, accuracy


class ModelSharingUtils:
    """
    A class for handling peer-to-peer WAFL model sharing.
    """

    cMDLREQ = "MDLREQ"

    def __init__(self, index: int, name: str, addr: str, port: int, timeout: float) -> None:
        """
        Initialize the instance attributes.
        """
        self.vMODEL_INSTANCE = None
        self.vMODEL_INSTANCE_CACHE = None
        self.vMODEL_METADATA = ""
        self.fLISTENER_ACTIVE = True
        self.agent_index = index
        self.name = name
        self.addr = addr
        self.port = port
        self.timeout = timeout
        self.logger = logging.getLogger("ModelSharingUtils")
        self.logger.debug("Initialized Model Sharing Utils")

        # Load method settings from parameters.json
        try:
            with open("ctrl/parameters.json", "r") as f:
                params = json.load(f)
                method_config = params.get("method", {})
                udp_config = method_config.get("udp", {})
                comp_config = method_config.get("compression", {})

                self.udp_enabled = udp_config.get("enabled", False)
                self.udp_fec_m = udp_config.get("fec_m", 9)

                self.compression_enabled = comp_config.get("enabled", False)
                self.initial_comp_method = comp_config.get("initial_method", "zlib")

        except Exception as e:
            self.logger.warning(f"Failed to load method settings from parameters.json: {e}. Using defaults.")
            self.udp_enabled = False
            self.udp_fec_m = 9
            self.compression_enabled = False
            self.initial_comp_method = "zlib"

        # Initialize UDP and Compression
        if self.udp_enabled:
            self.logger.info(f"🚀 UDP Enabled (FEC M={self.udp_fec_m})")
            self.udp_sharing = UDPModelSharing(self.addr, self.port, fec_m=self.udp_fec_m)
            # Start listener callback
            self.udp_sharing.start_listener(self._on_udp_model_received)
        else:
            self.logger.info("UDP Disabled")
            self.udp_sharing = None

        if self.compression_enabled:
            self.logger.info(f"🗜️ Compression Enabled (Initial: {self.initial_comp_method})")
            self.compression_manager = CompressionManager(initial_method=self.initial_comp_method)
        else:
            self.logger.info("Compression Disabled")
            self.compression_manager = None

        # UDP Received models buffer
        self.received_models = {}  # {ip: data}
        self.received_models_lock = threading.Lock()

        # TCP traffic statistics (for when UDP is disabled)
        self.tcp_stats = {
            "bytes_sent": 0,
            "bytes_received": 0,
            "models_sent": 0,
            "models_received": 0,
        }

        self.listener_thread = threading.Thread(target=self._socket_listener_thread, daemon=False, name="P2P_Listener")
        self.listener_thread.start()

        self.logger.info("🚀 Launched the P2P Transfer Threads (TCP & UDP).")

    def _on_udp_model_received(self, data: bytes, source_ip: str):
        """Callback for UDP model reception."""
        self.logger.info(f"📦 Received UDP model from {source_ip} ({len(data)} bytes)")
        with self.received_models_lock:
            self.received_models[source_ip] = data

    def _serialize_model(self, LE_model: Any) -> bytes:
        """
        Serialize the WAFL model for sharing
        (from the last completed epoch).
        """
        self.logger.debug("🔢 Serializing the model for transfer.")
        try:
            serialized_output = pickle.dumps(LE_model)
            # Apply compression if enabled
            if self.compression_manager:
                compressed_output = self.compression_manager.compress(serialized_output)
                return compressed_output
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
            # Decompress if enabled
            if self.compression_manager:
                decompressed_data = self.compression_manager.decompress(SR_model)
                deserialized_output = pickle.loads(decompressed_data)
            else:
                deserialized_output = pickle.loads(SR_model)
            return deserialized_output
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

                # Try UDP if enabled
                if self.udp_enabled and "protocol=udp" not in other_options:
                    other_options += "&protocol=udp"

                command = f"{ModelSharingUtils.cMDLREQ}:src={self.addr}{other_options}\r\n"
                s.settimeout(self.timeout)
                s.connect((peer_IP, self.port))
                s.sendall(command.encode("utf-8"))

                # Check response
                # If UDP, we expect "OK_UDP" or similar, then wait for UDP.
                # If TCP, we get data directly.

                # Read first chunk to see if it is OK_UDP
                first_packet = s.recv(4096)
                if b"OK_UDP" in first_packet:
                    self.logger.info(f"Waiting for UDP model from {peer_IP}...")
                    # Wait for UDP data
                    start_wait = time.time()
                    while time.time() - start_wait < self.timeout * 2:  # Give more time for UDP
                        with self.received_models_lock:
                            if peer_IP in self.received_models:
                                data = self.received_models.pop(peer_IP)
                                break
                        time.sleep(0.1)
                    else:
                        raise Exception("UDP RECEIVE TIMEOUT")
                else:
                    # TCP fallback or standard TCP
                    data = [first_packet]
                    while True:
                        packet = s.recv(4096)
                        if not packet:
                            break
                        data.append(packet)
                    data = b"".join(data)
                    # Record TCP bytes received
                    self.tcp_stats["bytes_received"] += len(data)
                    self.tcp_stats["models_received"] += 1

            data = self._deserialize_model(data)
            if data == b"ERROR" or data is None:
                raise Exception("FETCH ERROR")
            return True, data
        except Exception as exc:
            self.logger.error(f"The following error occurred in _fetch_model: {str(exc)[:100]}...")
            return False, b"ERROR"

    def _dispatch_model(self, conn: socket.socket, options: str) -> bool:
        """
        Utility function for sending the model data to the peer.
        Depending on the WAFL project, the options parameter may
        determine the processing that takes place inside this
        function.
        """
        try:
            self.logger.debug("⏳ Preparing the WAFL model data to be dispatched.")
            self.logger.debug(f"🖨️ The received OPTIONS for dispatch: {options}")

            use_udp = "protocol=udp" in options and self.udp_enabled

            # OPTIONS-specific processing
            # code should be added here.
            # For now the entire model is dispatched.
            model_data = self.vMODEL_INSTANCE
            if self.vMODEL_INSTANCE_CACHE is None:
                model_data = self._serialize_model(model_data)
                # Compression is already applied in _serialize_model
                self.vMODEL_INSTANCE_CACHE = model_data
            else:
                model_data = self.vMODEL_INSTANCE_CACHE

            if model_data == b"ERROR":
                self.vMODEL_INSTANCE_CACHE = None
                raise Exception("DISPATCH ERROR")

            if use_udp:
                # Get peer IP from connection
                peer_ip = conn.getpeername()[0]
                self.logger.info(f"📡 Dispatching model via UDP to {peer_ip}")
                # Send OK via TCP first? Or just send UDP?
                # Usually we should confirm receipt of request.
                # But _dispatch_model is called inside the loop.
                # If we send UDP, we might not send anything on TCP, or send "OK".
                # But the requester expects data on TCP if not UDP.
                # If UDP, requester is waiting on UDP.
                # We should send a small confirmation on TCP.
                conn.sendall(b"OK_UDP\r\n")

                # Send via UDP
                success = self.udp_sharing.send_model(model_data, peer_ip, self.port)  # Use same port
                if not success:
                    raise Exception("UDP DISPATCH ERROR")
            else:
                conn.sendall(model_data)
                # Record TCP bytes sent
                self.tcp_stats["bytes_sent"] += len(model_data)
                self.tcp_stats["models_sent"] += 1

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
            s.bind(("0.0.0.0", self.port))
            s.settimeout(self.timeout)
            s.listen()
            self.logger.info(f"P2P server listening on {self.addr}:{self.port}")
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
                    self.logger.info("P2P server - Sending model data to peer")
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
        WAIT_TIME = 1.0
        GROWTH_FACTOR = 1.5
        while not FETCHED and self.fLISTENER_ACTIVE:
            FETCHED, model_data = self._fetch_model(peer_IP, other_options)
            if FETCHED:
                self.logger.info(f"Model received from peer {str(peer_IP)}")
                return model_data
            time.sleep(WAIT_TIME)
            WAIT_TIME **= GROWTH_FACTOR
        # Handling KILL command from Control Server.
        return "Fetch Operation Terminated Abruptly."

    def get_epoch_metrics(self) -> dict:
        """
        Get communication and compression metrics for the current epoch.
        Returns a dictionary with UDP/FEC and compression stats.
        """
        metrics = {}

        # UDP/FEC metrics
        if self.udp_sharing is not None:
            udp_stats = self.udp_sharing.stats
            metrics.update(
                {
                    "survival_rate": self.udp_sharing.get_survival_rate(),
                    "sent_models": udp_stats.get("sent_models", 0),
                    "sent_failed": udp_stats.get("sent_failed", 0),
                    "received_models": udp_stats.get("received_models", 0),
                    "fec_recovery_success": udp_stats.get("fec_recovery_success", 0),
                    "fec_recovery_fail": udp_stats.get("fec_recovery_fail", 0),
                    "bytes_sent": udp_stats.get("bytes_sent", 0),
                    "bytes_received": udp_stats.get("bytes_received", 0),
                }
            )
        else:
            # TCP only - use tcp_stats
            metrics.update(
                {
                    "survival_rate": 1.0,  # TCP is reliable
                    "sent_models": self.tcp_stats.get("models_sent", 0),
                    "sent_failed": 0,  # TCP doesn't track failures the same way
                    "received_models": self.tcp_stats.get("models_received", 0),
                    "fec_recovery_success": 0,
                    "fec_recovery_fail": 0,
                    "bytes_sent": self.tcp_stats.get("bytes_sent", 0),
                    "bytes_received": self.tcp_stats.get("bytes_received", 0),
                }
            )

        # Compression metrics
        if self.compression_manager is not None:
            comp_stats = self.compression_manager.get_epoch_stats()
            metrics.update(
                {
                    "compression_method": self.compression_manager.method,
                    "compression_ratio": comp_stats.get("ratio", 1.0),
                    "compression_time_ms": comp_stats.get("time_ms", 0),
                    "original_size": comp_stats.get("original_size", 0),
                    "compressed_size": comp_stats.get("compressed_size", 0),
                }
            )
        else:
            # No compression - default values
            metrics.update(
                {
                    "compression_method": "none",
                    "compression_ratio": 1.0,
                    "compression_time_ms": 0,
                    "original_size": 0,
                    "compressed_size": 0,
                }
            )

        # Reset epoch-level statistics after collecting
        self._reset_epoch_stats()

        return metrics

    def _reset_epoch_stats(self):
        """Reset epoch-level statistics for the next epoch."""
        # Reset TCP stats
        self.tcp_stats = {
            "bytes_sent": 0,
            "bytes_received": 0,
            "models_sent": 0,
            "models_received": 0,
        }
        # Reset UDP stats if enabled
        if self.udp_sharing is not None:
            self.udp_sharing.stats = {
                "sent_models": 0,
                "sent_failed": 0,
                "received_models": 0,
                "fec_recovery_success": 0,
                "fec_recovery_fail": 0,
                "total_chunks_received": 0,
                "failed_chunks": 0,
                "bytes_sent": 0,
                "bytes_received": 0,
            }
        # Reset compression stats if enabled
        if self.compression_manager is not None:
            self.compression_manager.reset_epoch_stats()


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
        self.agent_index: Optional[int] = None
        self.name: Optional[str] = None
        self.addr: Optional[str] = None
        self.ctrl_port: Optional[int] = None
        self.p2p_port: Optional[int] = None
        self.timeout: Optional[float] = None
        self.experiment_id: Optional[str] = None

        # Flag to control the main listener loop.
        self.fLISTENER_ACTIVE = True

        # Variables for tracking the current status.
        self.is_epoch_running: bool = False
        self.current_epoch_type: Optional[str] = None  # "SELF" or "WAFL"
        self.current_epoch_number: Optional[str] = None  # 5-digit string
        self.status_logs: List[str] = []

        # Variable to hold the learning thread object.
        self.learning_thread: Optional[threading.Thread] = None

        # Initialize model_sharing and model_learning attributes
        self.model_sharing: Optional[ModelSharingUtils] = None
        self.model_learning: Optional[ModelLearningUtils] = None

        if not self._load_config(config_path):
            raise ValueError("Failed to load configuration file")
        if self.agent_index is None:
            raise ValueError("Device index not properly loaded from configuration")
        if self.name is None:
            raise ValueError("Device name not properly loaded from configuration")

        # Setup the WAFL node before starting the control thread
        self.setup_local_wafl_node(self.agent_index, self.name)

        self.ctrl_listener_thread = threading.Thread(target=self.wait_ctrl, daemon=False, name="CTRL_TCP_Listener")
        self.ctrl_listener_thread.start()

        # Initialize Metrics Logger
        self.metrics_logger = MetricsLogger(self.experiment_id, self.name, self.start_timestamp)

        # Start Resource Monitor
        self.monitor_thread = threading.Thread(
            target=self._monitor_resources,
            args=(
                f"./results/{self.experiment_id}/resources.csv",
                1.0,
                self.start_timestamp,
            ),
            daemon=True,
            name="ResourceMonitor",
        )
        self.monitor_thread.start()

        self.logger.debug("Initialized CTRL_TCP instance")

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
            experiment_info = config_data.get("experiment_info")
            if not isinstance(experiment_info, dict):
                self.logger.error("Invalid format for 'experiment_info' in JSON. Dictionary required.")
                return False
            self.experiment_id = experiment_info.get("experiment_id")
            if not isinstance(self.experiment_id, str):
                self.logger.error("Invalid format for 'experiment_info.experiment_id' in JSON. String required.")
                return False

            self.start_timestamp = experiment_info.get("start_timestamp")
            if self.start_timestamp is None:
                self.logger.warning("start_timestamp not found in config, using current time fallback")
                self.start_timestamp = time.time()

            wafl_devices_data = config_data.get("infrastructure")
            if not isinstance(wafl_devices_data, dict):
                self.logger.error("Invalid format for 'wafl_devices' in JSON. Dictionary required.")
                return False
            self.device_names = wafl_devices_data.get("device_names")
            if not (isinstance(self.device_names, list) and all(isinstance(n, str) for n in self.device_names)):
                self.logger.error("Invalid format for 'wafl_devices.name' in JSON. List of strings required.")
                return False
            self.device_ips = wafl_devices_data.get("device_ips")
            if not (isinstance(self.device_ips, list) and all(isinstance(ip, str) for ip in self.device_ips)):
                self.logger.error("Invalid format for 'wafl_devices.ip' in JSON. List of strings required.")
                return False
            agent_info = config_data.get("agent_info")
            if not isinstance(agent_info, dict):
                self.logger.error("Invalid format for 'agent_info' in JSON. Dictionary required.")
                return False
            self.agent_index = agent_info.get("index")
            if not isinstance(self.agent_index, int):
                self.logger.error("Invalid format for 'wafl_devices.index' in JSON. Integer required.")
                return False
            self.name = agent_info.get("device_name")
            if not isinstance(self.name, str):
                self.logger.error("Invalid format for 'wafl_devices.self_name' in JSON. String required.")
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
            experiment_parameters = config_data.get("experiment_parameters")
            self.wafl_phase_params = experiment_parameters.get("wafl_phase")
            if not isinstance(self.wafl_phase_params, dict):
                self.logger.error("Invalid format for 'wafl_phase_params' in JSON. Dictionary required.")
                return False
            for key in ["batch_size", "learning_rate", "coefficiency"]:
                if key not in self.wafl_phase_params:
                    raise ValueError(f"Missing required key '{key}' in wafl_phase_params")
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

    def get_device_ip(self, agent_index: int) -> Optional[str]:
        """
        Returns the IP address corresponding to the given agent index.
        Returns:
            Optional[str]: The corresponding IP address string, or None if not found.
        """
        if self.device_names is None or self.device_ips is None:
            self.logger.warning("Device names or IPs list is not initialized.")
            return None
        try:
            return self.device_ips[agent_index]
        except IndexError:
            self.logger.warning(f"Index for agent {agent_index} is out of bounds for the IP list. Data inconsistency might exist.")
            return None

    def _monitor_resources(self, output_file: str, interval: float, start_timestamp: float):
        """Monitor system resources and write to CSV."""
        import psutil

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Write header if file doesn't exist
        if not os.path.exists(output_file):
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "cpu_percent", "memory_percent", "memory_used_mb"])

        while True:
            try:
                # Calculate relative time
                current_time = time.time()
                relative_time = current_time - start_timestamp

                cpu_percent = psutil.cpu_percent(interval=None)
                memory_info = psutil.virtual_memory()
                memory_percent = memory_info.percent
                memory_used_mb = memory_info.used / (1024 * 1024)

                with open(output_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([relative_time, cpu_percent, memory_percent, memory_used_mb])

                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error in resource monitor: {e}")
                time.sleep(interval)
                timestamp = time.time()
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()

                with open(output_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            timestamp,
                            cpu_percent,
                            memory.percent,
                            memory.used / (1024 * 1024),
                        ]
                    )

                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error monitoring resources: {e}")
                time.sleep(interval)

    def setup_local_wafl_node(self, agent_index: int, device_name: str) -> None:
        """
        Sets up the node with the given device name.
        This function should be called before starting the WAFL model training.
        """
        self.logger.info(f"Setting up the node with device name: {device_name}")
        device_ip = self.get_device_ip(agent_index)
        if device_ip is None:
            raise ValueError(f"Could not find IP address for device: {device_name}")
        if self.p2p_port is None:
            raise ValueError("P2P port is not properly configured")
        if self.timeout is None:
            raise ValueError("Timeout is not properly configured")

        model_dir = f"./results/{self.experiment_id}"
        os.makedirs(model_dir, exist_ok=True)
        self.model_sharing = ModelSharingUtils(agent_index, device_name, device_ip, self.p2p_port, self.timeout)
        self.model_learning = ModelLearningUtils(
            "./dataset",
            f"./results/{self.experiment_id}/model.pth",
            "./contact_pattern.json",
            self.model_sharing,
            self,
            self.experiment_id,
            self.wafl_phase_params,
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
            s.bind(("0.0.0.0", self.ctrl_port))
            s.listen()
            self.logger.info(f"Control server listening on {self.addr}:{self.ctrl_port}")

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
                    self.logger.info(f"Control connection closed - {addr_info[0]}:{addr_info[1]}")

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
        self.logger.info(f"Command received: {command_str}")
        parts = command_str.split("-")

        main_command = parts[0] if len(parts) > 0 else ""

        try:
            if main_command == "KILL":
                if len(parts) != 1:
                    self.logger.warning(f"Incorrect form of KILL command: {command_str}")
                    conn.sendall("ERROR\r\n".encode("utf-8"))
                    return False
                # Handle the KILL command by setting flags to terminate loops.
                self.logger.info("KILL command - Shutting down gracefully")
                self.fLISTENER_ACTIVE = False
                if self.model_sharing is not None:
                    self.model_sharing.fLISTENER_ACTIVE = False
                threads_to_join = []
                if self.model_sharing is not None and hasattr(self.model_sharing, "listener_thread"):
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

            elif main_command == "FORCE_NEXT":
                if len(parts) != 1:
                    self.logger.warning(f"Incorrect form of FORCE_NEXT command: {command_str}")
                    conn.sendall("ERROR\r\n".encode("utf-8"))
                    return False

                self.logger.info("⚠️ Received FORCE_NEXT command. Resetting current epoch (SSP Reset).")
                wasted_metrics = None

                if self.learning_thread and self.learning_thread.is_alive():
                    if self.model_learning:
                        # Get wasted metrics before stopping
                        wasted_metrics = self.model_learning.stop_current_epoch()

                    # Wait for thread to finish
                    self.learning_thread.join(timeout=10)
                    if self.learning_thread.is_alive():
                        self.logger.error("Failed to stop learning thread gracefully.")
                    else:
                        self.logger.info("Learning thread stopped successfully.")

                    # Log detailed wasted computation metrics
                    if wasted_metrics:
                        epoch_num = int(self.current_epoch_number) if self.current_epoch_number else 0
                        phase = self.current_epoch_type or "UNKNOWN"
                        self.metrics_logger.log_ssp_metrics(
                            phase=phase,
                            epoch=epoch_num,
                            wasted_ms=wasted_metrics.get("wasted_ms", 0.0),
                            wasted_norm=wasted_metrics.get("wasted_norm", 0.0),
                            batches_processed=wasted_metrics.get("batches_processed", 0),
                            was_force_stopped=True,
                        )
                        self.logger.info(f"📊 SSP Metrics: wasted_ms={wasted_metrics['wasted_ms']:.2f}, wasted_norm={wasted_metrics['wasted_norm']:.6f}, batches={wasted_metrics['batches_processed']}")
                    else:
                        # Fallback logging if metrics not available
                        epoch_num = int(self.current_epoch_number) if self.current_epoch_number else 0
                        phase = self.current_epoch_type or "UNKNOWN"
                        self.metrics_logger.log_ssp_metrics(
                            phase=phase,
                            epoch=epoch_num,
                            wasted_ms=0.0,
                            wasted_norm=0.0,
                            batches_processed=0,
                            was_force_stopped=True,
                        )

                conn.sendall("OK\r\n".encode("utf-8"))
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
                        return True
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
            self.logger.error(
                f"Error during command processing for '{command_str}': {e}",
                exc_info=True,
            )
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

        if self.model_learning is None:
            self.logger.error("Model learning is not initialized")
            return False

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

        if self.model_learning is None:
            self.logger.error("Model learning is not initialized")
            return False

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
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    comm_interface = CTRL_TCP("./config.json")

    # Keep the main thread alive while the control server thread runs
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        logging.info("Shutting down...")

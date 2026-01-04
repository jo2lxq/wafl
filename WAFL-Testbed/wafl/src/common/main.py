from __future__ import annotations

import copy
import csv
import gc
import json
import logging
import os
import pickle
import socket
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from .compression_manager import CompressionManager
from .logger import MetricsLogger
from .net import Net
from .udp_model_sharing import UDPModelSharing


def background_deserialize(data: bytes, use_compression: bool) -> Any:
    """
    Deserializes model data in a background process.
    Handles decompression (zlib/lz4) and unpickling.
    """
    import pickle
    import zlib

    if not data:
        return b"ERROR"

    try:
        decompressed = data
        if use_compression:
            # CompressionManager header format: [MethodID:1B][Payload]
            mid = data[0]
            payload = data[1:]

            if mid == 1:  # Zlib
                decompressed = zlib.decompress(payload)
            elif mid == 2:  # LZ4
                try:
                    import lz4.frame

                    decompressed = lz4.frame.decompress(payload)
                except ImportError:
                    return b"ERROR:LZ4_MISSING"
            else:
                # mid=0 (None) or unknown
                decompressed = payload

        return pickle.loads(decompressed)
    except Exception:
        # Return error string on failure
        return b"ERROR"


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
        self_epochs: int = 0,
    ) -> None:
        """
        Initialize the learning process instance.

        Args:
            self_epochs: Number of SELF phase epochs, used as offset for WAFL phase epoch numbering
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
        self.self_epochs = self_epochs  # Offset for WAFL phase epoch numbering
        self.current_phase = "SELF"  # Track current phase for epoch calculation

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
        self.current_phase = phase_name

        # Calculate unified epoch number (1-indexed)
        # epoch_number is 0-indexed cumulative counter across all phases
        # display_epoch = epoch_number + 1 (no offset needed, epoch_number accumulates)
        display_epoch = self.epoch_number + 1

        if not WAFL_LEARN:
            self.logger.debug(f"Training [SELF] - Epoch {display_epoch} started")
        SUCCESS = False

        # Reset SSP tracking
        self.stop_requested = False
        self.epoch_start_time = time.time()
        self.batches_processed = 0
        self.current_gradient_norm = 0.0

        try:
            # Disable GC during training to reduce latency spikes
            gc.disable()

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
                    # Re-enable GC before returning
                    gc.enable()
                    gc.collect()
                    return False
            self.train_loss = total_loss / num_batches if num_batches > 0 else 0.0
            self.train_accuracy = correct_train / total_train if total_train > 0 else 0.0

            # Always evaluate the model to get test metrics
            test_loss, test_accuracy = self._evaluate_model()

            if not WAFL_LEARN:
                self.logger.info(f"🏁 Completed the Self-Learning Epoch: {display_epoch}")

            self.logger.info(f"📉 Training Loss: {self.train_loss:.6f}")
            self.logger.info(f"📈 Training Accuracy: {self.train_accuracy:.4f}")
            self.logger.info(f"📉 Test Loss: {test_loss:.4f}")
            self.logger.info(f"📈 Test Accuracy: {test_accuracy:.4f}")

            # Log metrics to CSV
            # self_learn内の時間 = 純粋な計算時間 (training + evaluation)
            compute_time_ms = (time.time() - self.epoch_start_time) * 1000 if self.epoch_start_time else 0.0

            # WAFLフェーズでは、通信時間はself_learnの前にwafl_learnで計測されている
            # epoch_duration_ms = compute_time_ms + comm_time_ms (合計時間)
            comm_time_ms = getattr(self, "_wafl_comm_time_ms", 0.0)
            waiting_time_ms = getattr(self, "_wafl_waiting_time_ms", 0.0)
            epoch_duration_ms = compute_time_ms + comm_time_ms if WAFL_LEARN else compute_time_ms

            metrics = {
                "train_loss": self.train_loss,
                "train_accuracy": self.train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "epoch_duration_ms": epoch_duration_ms,
                "compute_time_ms": compute_time_ms,
                "comm_time_ms": comm_time_ms,
                "waiting_time_ms": waiting_time_ms,
                # SSP metrics (no waste if epoch completed normally)
                "batches_processed": self.batches_processed,
            }
            # Add communication and compression metrics
            comm_metrics = self.model_sharing.get_epoch_metrics()
            metrics.update(comm_metrics)

            # Reset wafl comm time for next epoch
            self._wafl_comm_time_ms = 0.0
            self._wafl_waiting_time_ms = 0.0

            self.ctrl_tcp.metrics_logger.log_epoch(phase_name, display_epoch, metrics)

            self.epoch_number += 1
            torch.save(
                self.net.state_dict(),
                self.model_instance_path,
            )
            self.model_sharing.update_model_instance(self.net.state_dict(), "Testing", epoch_number=display_epoch)
            SUCCESS = True
        except Exception as exc:
            self.logger.error(f"The following error occurred in self_learn: {str(exc)[:100]}...")
            SUCCESS = False
        finally:
            # Re-enable GC after training epoch
            gc.enable()
            gc.collect()
        return SUCCESS

    def wafl_learn(self, five_digit_number_str: str) -> bool:
        """
        Implementation of the Epoch-by-Epoch Learning Process for WAFL-MLP.
        Uses parallel model fetching from all neighbors for improved throughput.
        """
        # Calculate unified epoch number for display (1-indexed)
        # epoch_number accumulates across phases, so no offset needed
        display_epoch = self.epoch_number + 1
        self.logger.info(f"🎯 Beginning the WAFL-Learning Epoch: {display_epoch}")

        # Note: Adaptive FEC is now updated at epoch END in get_epoch_metrics()
        # This ensures stats are available before reset

        SUCCESS = False
        try:
            neighbours = self.get_neighbour_list(five_digit_number_str)
            n_nbr = len(neighbours)
            local_model = copy.deepcopy(self.net.state_dict())

            # Measure communication time for model exchange
            # Includes overhead, waiting time, and processing
            comm_start_time = time.time()
            waiting_time_ms = 0.0

            # Parallel model fetching from all neighbors
            received_models = []

            def fetch_from_peer(neighbour):
                """Fetch model from a single peer. Returns (neighbour, model) or (neighbour, None)."""
                peer_ip = self.ctrl_tcp.get_device_ip(neighbour)
                if peer_ip is None:
                    self.logger.error(f"Could not get IP for neighbour {neighbour}")
                    return (neighbour, None)
                try:
                    # Pass explicit timeout to request_model_from_peer if needed,
                    # but self.model_sharing.timeout is used by socket.
                    # We rely on strict future timeout for overall control.
                    model = self.model_sharing.request_model_from_peer(peer_ip, "&purpose=testing")
                    if isinstance(model, str):
                        self.logger.error(f"Failed to receive model from {neighbour}: {model[:50]}")
                        return (neighbour, None)
                    return (neighbour, model)
                except Exception as e:
                    self.logger.error(f"Exception fetching model from {neighbour}: {e}")
                    return (neighbour, None)

            if n_nbr > 0:
                # Use ThreadPoolExecutor for parallel model fetching
                max_workers = min(n_nbr, 16)  # Cap at 16 concurrent connections

                # Use the configured timeout from ModelSharingUtils
                # This ensures we don't wait longer than the allowed epoch timeout
                strict_timeout = self.model_sharing.timeout

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_nbr = {executor.submit(fetch_from_peer, n): n for n in neighbours}
                    futures = list(future_to_nbr.keys())

                    wait_start = time.time()

                    # Enforce strict timeout using wait()
                    # return_when=ALL_COMPLETED not used because we want to stop exactly at timeout
                    # actually wait() returns (done, not_done) after timeout
                    from concurrent.futures import wait

                    done, not_done = wait(futures, timeout=strict_timeout)

                    waiting_time_ms = (time.time() - wait_start) * 1000

                    # Process completed futures
                    for future in done:
                        try:
                            neighbour, model = future.result()
                            if model is not None:
                                received_models.append(model)
                        except Exception as e:
                            nbr = future_to_nbr[future]
                            self.logger.error(f"Error getting result from {nbr}: {e}")

                    # Handle timed out futures
                    if not_done:
                        self.logger.warning(f"⚠️ Timeout waiting for {len(not_done)} neighbors (>{strict_timeout}s)")
                        for future in not_done:
                            nbr = future_to_nbr[future]
                            future.cancel()  # Verify if this actually kills the thread (it mostly doesn't in Python TPE)
                            # But we stop waiting for it.
                            self.logger.warning(f"  ❌ Cancelled fetch from {nbr} due to timeout")
                            # Count as fetch failure in stats
                            if self.model_sharing.udp_sharing is None:
                                self.model_sharing.tcp_stats["fetch_failed"] += 1
                            else:
                                # UDP モードでもタイムアウトをカウント
                                self.model_sharing.udp_sharing.stats["timeout_models"] += 1

            # Aggregate received models (optimized with no_grad and in-place operations)
            n_received = len(received_models)
            with torch.no_grad():
                coeff = self.wafl_phase_params["coefficiency"] / (n_nbr + 1)
                for received_model in received_models:
                    for key in self.net.state_dict():
                        model_difference = received_model[key] - self.net.state_dict()[key]
                        local_model[key].add_(model_difference * coeff)

            comm_time_ms = (time.time() - comm_start_time) * 1000
            self.net.load_state_dict(local_model)

            # Log communication stats
            if n_nbr > 0:
                self.logger.info(f"📡 Parallel model exchange: {n_received}/{n_nbr} neighbors (comm_time: {comm_time_ms:.1f}ms, waiting: {waiting_time_ms:.1f}ms)")
            else:
                self.logger.info("No neighbors available for model exchange in this epoch")

            # Store communication time for metrics (to be picked up by self_learn)
            self._wafl_comm_time_ms = comm_time_ms
            self._wafl_waiting_time_ms = waiting_time_ms  # Store for metrics

            # Call self_learn with WAFL_LEARN=True to continue training
            SELF_LEARN_FLAG = self.self_learn(five_digit_number_str, WAFL_LEARN=True)

            if not SELF_LEARN_FLAG:
                raise Exception("SELF-LEARNING ERROR")
            self.logger.info(f"✅ Completed the WAFL-Learning Epoch: {display_epoch}")
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
        self.vMODEL_EPOCH = 0  # Track the epoch number of the current model
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
                fec_m_config = udp_config.get("fec_m", 16)

                # Support "auto" or fixed integer value for fec_m
                if fec_m_config == "auto":
                    self.udp_fec_m = 16  # Optimal fixed value for all conditions
                    self.udp_adaptive_fec = True
                else:
                    self.udp_fec_m = int(fec_m_config)
                    self.udp_adaptive_fec = False

                # Load timeout configurations
                # Common timeout for model exchange (applies to both TCP and UDP)
                self.model_fetch_timeout = method_config.get("model_fetch_timeout", 3.0)

                # UDP-specific timeouts
                udp_timeouts = udp_config.get("timeouts", {})
                self.udp_initial_packet_timeout = udp_timeouts.get("initial_packet", 2.0)
                self.udp_inter_packet_timeout = udp_timeouts.get("inter_packet", 0.2)
                self.udp_model_completion_timeout = udp_timeouts.get("model_completion", 3.0)

                self.compression_enabled = comp_config.get("enabled", False)
                self.initial_comp_method = comp_config.get("initial_method", "zlib")

        except Exception as e:
            self.logger.warning(f"Failed to load method settings from parameters.json: {e}. Using defaults.")
            self.udp_enabled = False
            self.udp_fec_m = 16
            self.udp_adaptive_fec = False
            self.model_fetch_timeout = 3.0
            self.udp_initial_packet_timeout = 2.0
            self.udp_inter_packet_timeout = 0.2
            self.udp_model_completion_timeout = 3.0
            self.compression_enabled = False
            self.initial_comp_method = "zlib"

        # Override the constructor timeout with configurable model_fetch_timeout
        # This applies to both TCP and UDP for fairness
        self.timeout = self.model_fetch_timeout

        # Log timeout configuration
        self.logger.info(f"⏱️ Model fetch timeout (TCP/UDP): {self.model_fetch_timeout}s")
        if self.udp_enabled:
            self.logger.info(f"⏱️ UDP extra timeouts: inter_packet={self.udp_inter_packet_timeout}s")

        # Initialize UDP and Compression
        if self.udp_enabled:
            mode_str = "Auto" if self.udp_adaptive_fec else "Fixed"
            self.logger.info(f"🚀 UDP Enabled (FEC M={self.udp_fec_m}, Mode={mode_str})")
            self.udp_sharing = UDPModelSharing(self.addr, self.port, fec_m=self.udp_fec_m, timeout=self.udp_model_completion_timeout, inter_packet_timeout=self.udp_inter_packet_timeout)
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

        # Async Deserialization Executor
        self.executor = ProcessPoolExecutor(max_workers=2)

        # Smoothed Packet Loss (EWMA) for stable Adaptive FEC
        self.smoothed_packet_loss = 0.0
        self.packet_loss_alpha = 0.2  # EWMA weight for new observations (stabilized adaptation)

        # TCP traffic statistics (for when UDP is disabled)
        self.tcp_stats = {
            "bytes_sent": 0,
            "bytes_received": 0,
            "models_sent": 0,
            "models_received": 0,
            "fetch_failed": 0,
        }

        # Initialize physical network counters
        # We need to define get_sys_net_io first or use inline logic briefly,
        # but since methods are defined later, we can't call them in __init__ easily if not bound?
        # Python methods are bound at runtime. But get_sys_net_io is defined below in the class.
        # Actually it's better to just init to 0 and handle first call in get_epoch_metrics
        # OR define a valid initial value here.
        self.last_sys_tx = 0
        self.last_sys_rx = 0

        # Try to read initial values to avoid counting boot-up traffic
        try:
            with open("/sys/class/net/eth0/statistics/tx_bytes", "r") as f:
                self.last_sys_tx = int(f.read().strip())
            with open("/sys/class/net/eth0/statistics/rx_bytes", "r") as f:
                self.last_sys_rx = int(f.read().strip())
        except Exception:
            pass

        self.listener_thread = threading.Thread(target=self._socket_listener_thread, daemon=False, name="P2P_Listener")
        self.listener_thread.start()

        self.logger.info("🚀 Launched the P2P Transfer Threads (TCP & UDP).")

    def _on_udp_model_received(self, data: bytes, source_ip: str):
        """Callback for UDP model reception."""
        self.logger.info(f"📦 Received UDP model from {source_ip} ({len(data)} bytes) - Starting async deserialization")

        # Determine if compression is used (based on manager presence)
        use_compression = self.compression_manager is not None

        # Submit deserialization task to background process
        future = self.executor.submit(background_deserialize, data, use_compression)

        with self.received_models_lock:
            self.received_models[source_ip] = future

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
                    self.logger.debug("🔧 UDP enabled, adding protocol=udp to options")

                command = f"{ModelSharingUtils.cMDLREQ}:src={self.addr}{other_options}\r\n"
                self.logger.debug(f"📤 Sending MDLREQ command: {command.strip()}")

                # Set per-recv timeout to 5s, but enforce overall deadline
                per_recv_timeout = min(5.0, self.timeout)
                s.settimeout(per_recv_timeout)
                s.connect((peer_IP, self.port))
                self.logger.debug(f"🔌 Connected to {peer_IP}:{self.port}, timeout={per_recv_timeout}s")
                s.sendall(command.encode("utf-8"))

                # Overall transfer deadline - this is the key fix for TCP timeout
                transfer_deadline = time.time() + self.timeout

                # Check response
                # If UDP, we expect "OK_UDP" or similar, then wait for UDP.
                # If TCP, we get data directly.

                # Read first chunk to see if it is OK_UDP
                first_packet = s.recv(4096)
                self.logger.debug(f"📨 First packet received: {len(first_packet)} bytes")

                if b"OK_UDP" in first_packet:
                    self.logger.info(f"📡 UDP mode confirmed, waiting for UDP model from {peer_IP}...")

                    # Send READY acknowledgment so sender knows we're prepared
                    s.sendall(b"READY\r\n")
                    self.logger.debug("📤 Sent READY ack to sender")

                    # Wait for UDP data
                    start_wait = time.time()
                    while time.time() - start_wait < self.timeout:
                        with self.received_models_lock:
                            if peer_IP in self.received_models:
                                data_or_future = self.received_models.pop(peer_IP)
                                elapsed = time.time() - start_wait
                                self.logger.debug(f"📥 UDP model received from {peer_IP} in {elapsed:.2f}s, deserializing...")

                                # If it's a Future (Async Deserialization), wait for result
                                if isinstance(data_or_future, (bytes, bytearray)):
                                    # Legacy/Fallback if not future (safe fallback)
                                    return True, self._deserialize_model(data_or_future)
                                else:
                                    # Assume it's a Future
                                    try:
                                        deserialized_output = data_or_future.result(timeout=10.0)
                                        if isinstance(deserialized_output, bytes) and deserialized_output.startswith(b"ERROR"):
                                            self.logger.error("Async deserialization returned ERROR")
                                            return False, b"ERROR"
                                        return True, deserialized_output
                                    except Exception as e:
                                        self.logger.error(f"Async deserialization failed or timed out: {e}")
                                        return False, b"ERROR"
                                break
                        time.sleep(0.02)  # 20ms polling for faster response
                    else:
                        raise Exception("UDP RECEIVE TIMEOUT")
                else:
                    # TCP fallback or standard TCP - enforce STRICT overall deadline
                    self.logger.debug(f"📡 TCP mode: receiving model data from {peer_IP}")
                    # This is critical: even if packets trickle in slowly, we must
                    # abort if total transfer time exceeds deadline
                    data = [first_packet]
                    bytes_received_in_loop = len(first_packet)

                    while True:
                        # ALWAYS check overall deadline BEFORE attempting recv
                        current_time = time.time()
                        if current_time > transfer_deadline:
                            self.logger.warning(f"⏱️ TCP strict deadline exceeded after receiving {bytes_received_in_loop} bytes")
                            raise TimeoutError(f"TCP transfer deadline exceeded ({self.timeout}s)")

                        # Calculate remaining time and set socket timeout accordingly
                        remaining_time = transfer_deadline - current_time
                        s.settimeout(min(remaining_time, 0.5))  # Max 0.5s per recv

                        try:
                            packet = s.recv(4096)
                            if not packet:
                                self.logger.debug(f"📥 TCP transfer complete: {bytes_received_in_loop} bytes total")
                                break  # Connection closed by peer - normal end
                            data.append(packet)
                            bytes_received_in_loop += len(packet)
                        except socket.timeout:
                            # Socket timeout occurred - check if overall deadline exceeded
                            if time.time() > transfer_deadline:
                                self.logger.warning(f"⏱️ TCP deadline exceeded during recv ({bytes_received_in_loop} bytes received)")
                                raise TimeoutError(f"TCP transfer deadline exceeded ({self.timeout}s)")
                            # Do NOT continue - if timeout occurred, deadline is imminent
                            # Re-raise to abort this transfer attempt
                            raise TimeoutError(f"TCP recv timeout, aborting transfer ({bytes_received_in_loop} bytes received)")

                    data = b"".join(data)
                    # Record TCP bytes received
                    self.tcp_stats["bytes_received"] += len(data)
                    self.tcp_stats["models_received"] += 1
                    self.logger.debug(f"📊 TCP stats updated: received={len(data)} bytes, total_recv={self.tcp_stats['bytes_received']}")

            data = self._deserialize_model(data)
            if data == b"ERROR" or data is None:
                raise Exception("FETCH ERROR")
            self.logger.debug(f"✅ Model fetched from {peer_IP} successfully")
            return True, data
        except TimeoutError as te:
            self.logger.warning(f"⏱️ TCP timeout in _fetch_model: {te}")
            return False, b"ERROR"
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
                self.logger.debug(f"📤 Model size: {len(model_data)} bytes, UDP mode")

                # Send OK_UDP notification via TCP
                conn.sendall(b"OK_UDP\r\n")
                self.logger.debug("📤 Sent OK_UDP to receiver")

                # Wait for READY acknowledgment with extended timeout
                # This prevents race condition where UDP send starts before receiver is listening
                conn.settimeout(2.0)  # Reduced timeout for faster failure detection
                try:
                    ready_ack = conn.recv(16)
                    if b"READY" in ready_ack:
                        self.logger.debug("📨 Receiver confirmed READY for UDP")
                except (socket.timeout, Exception):
                    # Proceed even without ACK - receiver should be listening by now
                    self.logger.debug("⚠️ No READY ack received, proceeding with UDP send")
                    pass

                # Send via UDP immediately
                self.logger.debug(f"📤 Starting UDP send to {peer_ip}:{self.port}")
                success = self.udp_sharing.send_model(model_data, peer_ip, self.port)  # Use same port
                if not success:
                    raise Exception("UDP DISPATCH ERROR")
                self.logger.debug("✅ UDP send completed")
            else:
                self.logger.debug(f"📡 Dispatching model via TCP: {len(model_data)} bytes")
                conn.sendall(model_data)
                # Record TCP bytes sent
                self.tcp_stats["bytes_sent"] += len(model_data)
                self.tcp_stats["models_sent"] += 1
                self.logger.debug(f"📊 TCP stats updated: sent={len(model_data)} bytes, total_sent={self.tcp_stats['bytes_sent']}")

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

    def update_model_instance(self, LE_model: Any, metadata: str = "", epoch_number: int = 0) -> None:
        """
        Updates the WAFL model instance that is
        to be dispatched.

        Args:
            LE_model: The model state dict to share
            metadata: Optional metadata string
            epoch_number: The epoch number when this model was created (for staleness tracking)
        """
        self.vMODEL_INSTANCE = copy.deepcopy(LE_model)
        self.vMODEL_INSTANCE_CACHE = None
        self.vMODEL_METADATA = metadata
        self.vMODEL_EPOCH = epoch_number
        self.logger.debug(f"Updated model instance (epoch={epoch_number})")

    def request_model_from_peer(self, peer_IP: str, other_options: str = "", max_retries: int = 1) -> Any:
        """
        The wrapper function for retrieving model parameters from a WAFL peer device.
        options attribute, if non-empty, should be prefixed by a '&' character.
        Format of options: &param1=val1&param2=val2...

        Args:
            peer_IP: Target peer IP address
            other_options: Additional options string
            max_retries: Maximum number of retry attempts (default: 3)

        Returns:
            Model data if successful, error message string if failed
        """
        FETCHED = False
        WAIT_TIME = 1.0
        GROWTH_FACTOR = 1.5
        retries = 0

        while not FETCHED and self.fLISTENER_ACTIVE and retries < max_retries:
            FETCHED, model_data = self._fetch_model(peer_IP, other_options)
            if FETCHED:
                self.logger.info(f"Model received from peer {str(peer_IP)}")
                return model_data
            retries += 1
            if retries < max_retries:
                time.sleep(WAIT_TIME)
                WAIT_TIME = min(WAIT_TIME * GROWTH_FACTOR, 5.0)  # Cap wait time at 5s

        # Record fetch failure for TCP stats (used for survival rate calculation)
        if not FETCHED and self.udp_sharing is None:
            self.tcp_stats["fetch_failed"] = self.tcp_stats.get("fetch_failed", 0) + 1
            self.logger.warning(f"Failed to fetch model from {peer_IP} after {max_retries} retries")

        return "Fetch Operation Failed."

    def _read_sys_net_stats(self) -> tuple:
        """
        Read physical network interface statistics from /sys/class/net/eth0/statistics.
        Returns (tx_bytes, rx_bytes) representing total bytes sent/received on the interface.
        This includes ALL overhead: TCP/UDP headers, retransmissions, etc.
        """
        try:
            with open("/sys/class/net/eth0/statistics/tx_bytes", "r") as f:
                tx_bytes = int(f.read().strip())
            with open("/sys/class/net/eth0/statistics/rx_bytes", "r") as f:
                rx_bytes = int(f.read().strip())
            return tx_bytes, rx_bytes
        except FileNotFoundError:
            # Fallback for non-container environments
            try:
                import psutil

                net = psutil.net_io_counters()
                return net.bytes_sent, net.bytes_recv
            except Exception:
                return 0, 0

    def get_epoch_metrics(self) -> dict:
        """
        Get communication and compression metrics for the current epoch.
        Uses PHYSICAL interface counters from /sys/class/net/eth0/statistics
        to accurately capture all traffic including TCP retransmissions and headers.
        """
        # --- Physical Interface Statistics (Primary Metrics) ---
        current_tx, current_rx = self._read_sys_net_stats()

        # Calculate delta since last epoch
        phy_bytes_sent = max(0, current_tx - self.last_sys_tx)
        phy_bytes_received = max(0, current_rx - self.last_sys_rx)

        # Update baseline for next epoch
        self.last_sys_tx = current_tx
        self.last_sys_rx = current_rx

        self.logger.info(f"📊 Physical Network Stats: Sent={phy_bytes_sent / 1024:.1f}KB, Recv={phy_bytes_received / 1024:.1f}KB")

        metrics = {
            "bytes_sent": phy_bytes_sent,
            "bytes_received": phy_bytes_received,
        }

        # --- Survival Rate and Model Counts (Application Layer) ---
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
                }
            )
        else:
            # TCP mode
            fetch_success = self.tcp_stats.get("models_received", 0)
            fetch_failed = self.tcp_stats.get("fetch_failed", 0)
            fetch_attempts = fetch_success + fetch_failed
            survival_rate = fetch_success / fetch_attempts if fetch_attempts > 0 else 1.0

            metrics.update(
                {
                    "survival_rate": survival_rate,
                    "sent_models": self.tcp_stats.get("models_sent", 0),
                    "sent_failed": fetch_failed,
                    "received_models": fetch_success,
                    "fec_recovery_success": 0,
                    "fec_recovery_fail": 0,
                }
            )

        # --- Compression Metrics ---
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
            metrics.update(
                {
                    "compression_method": "none",
                    "compression_ratio": 1.0,
                    "compression_time_ms": 0,
                    "original_size": 0,
                    "compressed_size": 0,
                }
            )

        # Update adaptive FEC based on current epoch's packet loss (BEFORE reset)
        if self.udp_adaptive_fec and self.udp_sharing is not None:
            packet_loss = self._calculate_packet_loss()
            if packet_loss > 0:
                self.logger.info(f"📊 Epoch packet loss: {packet_loss:.1%}")
            self._update_adaptive_fec(packet_loss)

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
            "fetch_failed": 0,  # Must be reset to prevent accumulation across epochs
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
                "timeout_models": 0,
                "bytes_sent": 0,
                "bytes_received": 0,
            }
        # Reset compression stats if enabled
        if self.compression_manager is not None:
            self.compression_manager.reset_epoch_stats()

    def _calculate_packet_loss(self) -> float:
        """
        Calculate packet loss rate based on FEC recovery statistics.

        This estimates the NETWORK packet loss rate by analyzing how often
        FEC recovery was needed to reconstruct chunks. When packets are lost
        but FEC can recover, it indicates packet loss occurred.

        Now also accounts for:
        - Timeouts (complete model failure) = severe loss (treated as 50%+)
        - FEC decode failures = severe loss (treated as 25%+)

        Returns:
            float: Estimated packet loss rate (0.0-1.0)
        """
        if self.udp_sharing is None:
            return 0.0

        stats = self.udp_sharing.stats
        total_chunks = stats.get("total_chunks_received", 0)
        fec_success = stats.get("fec_recovery_success", 0)  # Chunks recovered via FEC
        fec_fail = stats.get("fec_recovery_fail", 0)  # Chunks that failed FEC decode
        timeout_models = stats.get("timeout_models", 0)  # Complete model timeouts
        received_models = stats.get("received_models", 0)  # Successfully received models

        # Check for severe conditions first
        # If we have timeouts but no successful receives, assume very high loss
        if timeout_models > 0 and received_models == 0:
            # All models timed out - severe packet loss (50%+)
            self.logger.warning(f"⚠️ All models timed out ({timeout_models}), assuming 50% packet loss")
            return 0.50

        # If we have FEC failures, it means loss exceeded FEC capacity
        if fec_fail > 0:
            # FEC couldn't recover some chunks - estimate 25%+ loss
            fail_ratio = fec_fail / max(1, total_chunks + fec_fail) if total_chunks + fec_fail > 0 else 0.25
            estimated_loss = max(0.25, fail_ratio)
            self.logger.info(f"📊 FEC failures detected ({fec_fail}), estimating {estimated_loss:.1%} loss")
            return min(1.0, estimated_loss)

        if total_chunks == 0:
            # No data yet - use moderate default if there were timeout attempts
            if timeout_models > 0:
                return 0.30  # Some models failed, assume moderate loss
            return 0.0

        # Calculate the fraction of chunks that needed FEC recovery
        # This directly correlates with network packet loss rate
        current_loss = fec_success / total_chunks if total_chunks > 0 else 0.0

        # Apply EWMA smoothing for stability
        if self.smoothed_packet_loss == 0.0:
            self.smoothed_packet_loss = current_loss
        else:
            self.smoothed_packet_loss = self.packet_loss_alpha * current_loss + (1 - self.packet_loss_alpha) * self.smoothed_packet_loss

        return min(1.0, self.smoothed_packet_loss)

    def _update_adaptive_fec(self, packet_loss: float) -> None:
        """
        Update FEC parameters based on measured packet loss.

        Strategy: Dynamic parity calculation.
        - k: Standard block size (usually 8-16)
        - parity: Redundant packets, increased when loss is high.

        Args:
            packet_loss: Measured packet loss rate (0.0-1.0)
        """
        if not self.udp_adaptive_fec or self.udp_sharing is None:
            return

        # Adaptive Logic based on Loss Tiers
        # Goal: Maintain recovery despite packet loss
        #
        # CRITICAL OPTIMIZATION: Disable tier switching to prevent Pacing oscillation.
        # Use fixed high-efficiency FEC (k=16, parity=2, ~12.5% overhead).
        # RTT CC and NACK-based ARQ handle congestion and packet loss.
        # This ensures stable Pacing and maximum throughput.

        # Fixed optimal parameters for all conditions

        # Call update with both k and parity
        # Ensure parity is at least 1

        # Simplified Adaptive FEC: k is fixed, only adjust Parity based on Loss.
        # This prevents oscillation and ensures efficiency.

        # Note: packet_loss argument is already smoothed by _calculate_packet_loss

        # Target k is fixed for optimal performance
        target_k = 16

        # Adjust Parity based on smoothed loss
        # Strategy: Keep overhead slightly above loss rate, with minimum safety
        # Start conservative (high parity) when no data is available
        if packet_loss < 0.03:
            target_parity = 1  # ~6% overhead (Excellent/Good)
        elif packet_loss < 0.08:
            target_parity = 2  # ~12.5% overhead (Moderate)
        elif packet_loss < 0.15:
            target_parity = 4  # ~25% overhead (Fair)
        elif packet_loss < 0.25:
            target_parity = 8  # ~50% overhead (Poor)
        else:
            target_parity = 16  # ~100% overhead (Very Poor)

        # Conservative startup: Use minimum Parity=4 until we have reliable loss data
        # (smoothed_packet_loss = 0 means no FEC feedback yet)
        if self.smoothed_packet_loss == 0.0:
            target_parity = max(target_parity, 4)
            self.logger.info(f"🛡️ Conservative FEC: No loss data yet, using Parity={target_parity}")

        # Apply update if parity changed
        current_parity = self.udp_sharing.parity

        if target_parity != current_parity:
            self.logger.info(f"📊 Adaptive FEC: Loss={self.smoothed_packet_loss:.1%} -> Parity {current_parity}->{target_parity} (k={target_k})")
            # Don't reset pacing, let RTT CC handle it
            self.udp_sharing.update_network_params(target_k, new_parity=target_parity)
            self.udp_fec_m = target_k


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

        # SSP: Pending metrics from FORCE_NEXT (to be logged at next epoch start)
        self._pending_ssp_metrics: Optional[dict] = None

        if not self._load_config(config_path):
            raise ValueError("Failed to load configuration file")
        if self.agent_index is None:
            raise ValueError("Device index not properly loaded from configuration")
        if self.name is None:
            raise ValueError("Device name not properly loaded from configuration")

        # Setup file logging for output.log (before initializing other components)
        # This ensures all logs from ModelSharingUtils, ModelLearningUtils, etc. are captured
        log_dir = f"./results/{self.experiment_id}"
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, "output.log")
        file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(file_handler)
        self.logger.info(f"📝 Log file initialized: {log_file_path}")

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

            # Load epochs configuration for unified epoch numbering
            epochs_config = experiment_parameters.get("epochs", {})
            self.self_epochs = epochs_config.get("self", 0)
            self.wafl_epochs = epochs_config.get("wafl", 0)
            self.logger.info(f"Epochs configuration: SELF={self.self_epochs}, WAFL={self.wafl_epochs}")

            # Load model exchange timeout from parameters
            method_config = config_data.get("method", {})
            self.timeout = method_config.get("model_exchange_timeout", 5.0)
            self.logger.info(f"Model exchange timeout set to {self.timeout}s")

            if not isinstance(self.timeout, (int, float)):
                self.logger.error("Invalid format for 'method.model_exchange_timeout' in JSON. Number required.")
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
            self_epochs=self.self_epochs,
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

                    # Store SSP metrics to be logged at next epoch start
                    # This ensures 1 epoch = 1 record
                    epoch_num = int(self.current_epoch_number) if self.current_epoch_number else 0
                    phase = self.current_epoch_type or "UNKNOWN"
                    self._pending_ssp_metrics = {
                        "epoch": epoch_num,
                        "phase": phase,
                        "wasted_ms": wasted_metrics.get("wasted_ms", 0.0) if wasted_metrics else 0.0,
                        "wasted_norm": wasted_metrics.get("wasted_norm", 0.0) if wasted_metrics else 0.0,
                        "batches_processed": wasted_metrics.get("batches_processed", 0) if wasted_metrics else 0,
                        "was_force_stopped": True,
                    }
                    self.logger.info(f"📊 SSP Metrics stored for epoch {epoch_num}: wasted_ms={self._pending_ssp_metrics['wasted_ms']:.2f}, batches={self._pending_ssp_metrics['batches_processed']}")

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

        # --- Log pending SSP metrics from previous FORCE_NEXT (if any) ---
        if self._pending_ssp_metrics:
            pending = self._pending_ssp_metrics
            self.logger.info(f"📝 Logging pending SSP metrics for epoch {pending['epoch']} (was force-stopped)")
            self.metrics_logger.log_epoch(
                phase=pending["phase"],
                epoch=pending["epoch"],
                metrics={
                    "wasted_ms": pending["wasted_ms"],
                    "wasted_norm": pending["wasted_norm"],
                    "batches_processed": pending["batches_processed"],
                    "was_force_stopped": 1,
                },
            )
            self._pending_ssp_metrics = None
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

        # --- Log pending SSP metrics from previous FORCE_NEXT (if any) ---
        if self._pending_ssp_metrics:
            pending = self._pending_ssp_metrics
            self.logger.info(f"📝 Logging pending SSP metrics for epoch {pending['epoch']} (was force-stopped)")
            self.metrics_logger.log_epoch(
                phase=pending["phase"],
                epoch=pending["epoch"],
                metrics={
                    "wasted_ms": pending["wasted_ms"],
                    "wasted_norm": pending["wasted_norm"],
                    "batches_processed": pending["batches_processed"],
                    "was_force_stopped": 1,
                },
            )
            self._pending_ssp_metrics = None
        # ---

        # Implement the logic for WAFL learning here
        FLAG = self.model_learning.wafl_learn(five_digit_number_str)

        # --- Update status at the end of the epoch ---
        self.status_logs.append(f"Epoch completion successful: {FLAG} at {time.ctime()}")
        self.is_epoch_running = False
        # ---
        return FLAG


if __name__ == "__main__":
    log_level = os.environ.get("LOG_LEVEL", "DEBUG").upper()
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

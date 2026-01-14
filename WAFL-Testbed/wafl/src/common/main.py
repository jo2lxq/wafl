from __future__ import annotations

import copy
import csv
import gc
import json
import logging
import os
import pickle
import re
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
from .network_estimator import get_network_estimator
from .rudp_model_sharing import RUDPModelSharing
from .udp_model_sharing import UDPModelSharing


def _get_measured_network_quality() -> str:
    """Return network quality based on per-node measured metrics.

    IMPORTANT: Must not rely on ctrl-provided static network conditions.
    """
    try:
        return get_network_estimator().get_metrics().get_quality_level()
    except Exception:
        return "poor"


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
        # In the original code, MetricsLogger was instantiated in CTRL_TCP or similar?
        # Let's check where MetricsLogger is instantiated.
        # It seems it was instantiated in CTRL_TCP.__init__ in the previous `view_file` of main.py (lines 670+).
        # But ModelLearningUtils is independent.
        # We should probably initialize MetricsLogger here if we want to use it, OR pass it.
        # However, to keep it simple and avoid changing signatures too much, let's instantiate it here.
        # Use shared metrics logger from CTRL_TCP to valid duplicate WandB runs
        self.metrics_logger = ctrl_tcp.metrics_logger

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

    def _validate_model_weights(self, model: dict, peer_name: str) -> bool:
        """
        Validate model weights to prevent propagation of corrupted/exploding models.
        Checks for NaN, Inf, and extremely large values.
        """
        try:
            for key, tensor in model.items():
                if not isinstance(tensor, torch.Tensor):
                    continue

                # specific check for float tensors
                if tensor.is_floating_point():
                    # Check for NaN/Inf
                    if not torch.isfinite(tensor).all():
                        self.logger.error(f"❌ Model validation failed for {peer_name}: NaN/Inf detected in {key}")
                        return False

                    # Check for exploding weights (absolute threshold)
                    # Normal weights are usually < 10.0. A threshold of 1e5 is extremely safe.
                    # 1e34 is definitely corruption.
                    if tensor.abs().max() > 1e6:
                        self.logger.error(f"❌ Model validation failed for {peer_name}: Exploding weights detected in {key} (max={tensor.abs().max():.2e})")
                        return False
            return True
        except Exception as e:
            self.logger.error(f"❌ Error during model validation for {peer_name}: {e}")
            return False

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

            # Resolve current epoch peers (for connection management + logging)
            peer_ips: list[str] = []
            missing_ip_neighbours: list[int] = []
            for n in neighbours:
                ip = self.ctrl_tcp.get_device_ip(n)
                if ip:
                    peer_ips.append(ip)
                else:
                    missing_ip_neighbours.append(n)

            # Resolve next epoch peers from contact pattern (if available)
            next_peer_ips: list[str] = []
            next_missing_ip_neighbours: list[int] = []
            try:
                cur_idx = int(five_digit_number_str)
                next_idx = cur_idx + 1
                if isinstance(self.neighbour_map, list) and 0 <= next_idx < len(self.neighbour_map):
                    next_neighbours = self.neighbour_map[next_idx].get(str(self.model_sharing.agent_index), [])
                    for n in next_neighbours:
                        ip = self.ctrl_tcp.get_device_ip(n)
                        if ip:
                            next_peer_ips.append(ip)
                        else:
                            next_missing_ip_neighbours.append(n)
            except Exception as e:
                self.logger.debug(f"Failed to resolve next epoch neighbours: {e}")

            # Measured network metrics (per-node)
            try:
                m = get_network_estimator().get_metrics()
                q = m.get_quality_level()
                self.logger.info(
                    f"🧭 Epoch plan: idx={five_digit_number_str}, neighbours={n_nbr}, peers_resolved={len(peer_ips)}, missing_ip={len(missing_ip_neighbours)}, "
                    f"next_peers={len(next_peer_ips)}, measured(q={q}, loss={m.packet_loss_rate * 100:.2f}%, rtt={m.rtt_ms:.1f}ms, bw={m.bandwidth_mbps:.2f}Mbps)"
                )
            except Exception:
                self.logger.info(f"🧭 Epoch plan: idx={five_digit_number_str}, neighbours={n_nbr}, peers_resolved={len(peer_ips)}, missing_ip={len(missing_ip_neighbours)}, next_peers={len(next_peer_ips)}")

            if missing_ip_neighbours:
                self.logger.warning(f"⚠️ Neighbours with missing IP mapping (first 5): {missing_ip_neighbours[:5]}")
            if next_missing_ip_neighbours:
                self.logger.warning(f"⚠️ Next-epoch neighbours with missing IP mapping (first 5): {next_missing_ip_neighbours[:5]}")

            # Epoch start hook (for RUDP logging only)
            try:
                self.model_sharing.on_epoch_start(epoch=display_epoch, planned_peer_ips=peer_ips)
            except Exception as e:
                self.logger.debug(f"on_epoch_start hook failed: {e}")

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
                strict_timeout = self.model_sharing.get_parallel_fetch_timeout()

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_nbr = {executor.submit(fetch_from_peer, n): n for n in neighbours}
                    futures = list(future_to_nbr.keys())

                    wait_start = time.time()

                    # Enforce strict timeout using wait()
                    # return_when=ALL_COMPLETED not used because we want to stop exactly at timeout
                    from concurrent.futures import wait

                    done, not_done = wait(futures, timeout=strict_timeout)

                    waiting_time_ms = (time.time() - wait_start) * 1000

                    # Process completed futures
                    for future in done:
                        try:
                            neighbour, model = future.result()
                            if model is not None:
                                # Validate received model to prevent corruption (Exploding Gradient/Weights)
                                if self._validate_model_weights(model, neighbour):
                                    received_models.append(model)
                                else:
                                    self.logger.warning(f"⚠️ Discarded corrupted model from {neighbour} (NaN/Inf or Exploding Weights)")
                        except Exception as e:
                            nbr = future_to_nbr[future]
                            self.logger.error(f"Error getting result from {nbr}: {e}")

                    if not_done:
                        self.logger.warning(f"⚠️ Timeout waiting for {len(not_done)} neighbors (>{strict_timeout}s)")
                        for future in not_done:
                            nbr = future_to_nbr[future]
                            future.cancel()  # Verify if this actually kills the thread (it mostly doesn't in Python TPE)
                            # But we stop waiting for it.
                            self.logger.warning(f"  ❌ Cancelled fetch from {nbr} due to timeout")
                            # Count as fetch failure in stats
                            if self.model_sharing.rudp_sharing is None and self.model_sharing.udp_sharing is None:
                                # TCP モードのみここでカウント (RUDP/UDP は内部タイムアウトでカウント)
                                self.model_sharing.tcp_stats["fetch_failed"] += 1

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

            # Epoch end: prune RUDP connections based on next epoch planned peers.
            try:
                self.model_sharing.on_epoch_end(epoch=display_epoch, next_epoch_peer_ips=next_peer_ips)
            except Exception as e:
                self.logger.debug(f"on_epoch_end hook failed: {e}")

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
        if timeout is None:
            self.timeout = 10.0  # デフォルトタイムアウト（4.0s -> 10.0s に変更してプロトコル内部タイムアウトを優先）
        else:
            self.timeout = timeout
        self.logger = logging.getLogger("ModelSharingUtils")
        self.logger.debug("Initialized Model Sharing Utils")

        # Initialize accumulators to avoid AttributeError if on_epoch_start is skipped (e.g. SELF phase)
        self._last_reported_tcp = {}
        self._last_reported_udp = {}
        self._last_reported_rudp = {}
        self.protocol_counts = {"TCP": 0, "UDP": 0, "RUDP": 0}

        # Load method settings from parameters.json
        try:
            with open("ctrl/parameters.json", "r") as f:
                params = json.load(f)
                net_cond_config = params.get("network_condition", {})
                # === New Schema Handling (String-based method) ===
                method_val = params.get("method")
                if isinstance(method_val, str):
                    self.active_protocol = method_val  # "tcp", "udp", "rudp", "dynamic"

                    # Set enabled flags based on active_protocol
                    self.tcp_enabled = self.active_protocol in ["tcp", "dynamic"]
                    self.udp_enabled = self.active_protocol in ["udp", "dynamic"]
                    self.rudp_enabled = self.active_protocol == "rudp"

                    # Protocol specific configs
                    # For simplify, we assume standard settings for now or load from root if desired.
                    # Actually implementation plan said we remove specific sub-configs.
                    # We just need to handle implicit settings.

                    if self.udp_enabled:
                        self.udp_fec_m = 16
                        self.udp_adaptive_fec = True
                        self.udp_max_packet_size = None

                    if self.rudp_enabled:
                        self.rudp_mode = "rudp"
                        self.rudp_max_retries = 20  # Standard default

                    # Compression
                    # For dynamic/udp, compression is usually off or auto?
                    # Plan said compression disabled fixed.
                    self.compression_enabled = False
                    self.compression_enabled = False
                    self.initial_comp_method = "zlib"

                    # Last reported stats for delta calculation (Control Server polling)
                    self._last_reported_tcp = {}
                    self._last_reported_udp = {}
                    self._last_reported_rudp = {}

                    # Protocol usage tracking for Dynamic mode
                    self.protocol_counts = {"TCP": 0, "UDP": 0, "RUDP": 0}

                else:
                    # === Backward Compatibility (Dict-based method) ===
                    method_config = params.get("method", {})
                    tcp_config = method_config.get("tcp", {})
                    udp_config = method_config.get("udp", {})
                    rudp_config = method_config.get("rudp", {})
                    comp_config = method_config.get("compression", {})

                    # TCP is enabled by default if no explicit config (backward compatibility)
                    self.tcp_enabled = tcp_config.get("enabled", True) if tcp_config else True

                    self.udp_enabled = udp_config.get("enabled", False)
                    fec_m_config = udp_config.get("fec_m", 16)
                    self.udp_max_packet_size = udp_config.get("max_packet_size", None)

                    if self.udp_enabled:
                        if fec_m_config != "auto":
                            # Warn instead of raise? No, strictly auto now.
                            self.logger.warning('Fixed UDP fec_m values are deprecated. Forcing "auto".')
                        self.udp_fec_m = 16
                        self.udp_adaptive_fec = True
                    else:
                        self.udp_fec_m = 16
                        self.udp_adaptive_fec = False

                    self.rudp_enabled = rudp_config.get("enabled", False)
                    self.rudp_mode = rudp_config.get("mode", "rudp")
                    self.rudp_max_retries = rudp_config.get("max_retries", 10)

                    self.compression_enabled = comp_config.get("enabled", False)
                    self.initial_comp_method = comp_config.get("initial_method", "zlib")

                    # Deduce active protocol for compat
                    if self.rudp_enabled:
                        self.active_protocol = "RUDP"  # normalize?
                    elif self.udp_enabled:
                        self.active_protocol = "UDP"
                    else:
                        self.active_protocol = "TCP"

                self.logger.info(f"🔌 Active transport protocol: {self.active_protocol} (TCP={self.tcp_enabled}, UDP={self.udp_enabled}, RUDP={self.rudp_enabled})")

                # Seed NetworkEstimator from static network condition (initialization only).
                # This avoids cold-start misconfiguration for UDP/RUDP adaptive parameters.
                try:
                    if isinstance(net_cond_config, dict) and net_cond_config.get("enabled", False):
                        rate = str(net_cond_config.get("rate", "")).strip().lower()
                        delay = str(net_cond_config.get("delay", "")).strip().lower()
                        loss = str(net_cond_config.get("loss", "")).strip().lower()

                        def _parse_rate_mbps(s: str) -> float:
                            m = re.match(r"^\s*([0-9.]+)\s*(kbit|mbit|gbit)\s*$", s)
                            if not m:
                                return 0.0
                            v = float(m.group(1))
                            unit = m.group(2)
                            if unit == "kbit":
                                return v / 1000.0
                            if unit == "mbit":
                                return v
                            if unit == "gbit":
                                return v * 1000.0
                            return 0.0

                        def _parse_delay_ms(s: str) -> float:
                            m = re.match(r"^\s*([0-9.]+)\s*ms\s*$", s)
                            if m:
                                return float(m.group(1))
                            m = re.match(r"^\s*([0-9.]+)\s*s\s*$", s)
                            if m:
                                return float(m.group(1)) * 1000.0
                            return 0.0

                        def _parse_loss_rate(s: str) -> float:
                            m = re.match(r"^\s*([0-9.]+)\s*%\s*$", s)
                            if m:
                                return float(m.group(1)) / 100.0
                            try:
                                v = float(s)
                                # allow either 0-1 or 0-100
                                return v / 100.0 if v > 1.0 else v
                            except Exception:
                                return 0.0

                        bw = _parse_rate_mbps(rate)
                        one_way_ms = _parse_delay_ms(delay)
                        # Assume delay is one-way; RTT ~ 2x
                        rtt_ms = max(1.0, one_way_ms * 2.0) if one_way_ms > 0 else 0.0
                        loss_rate = _parse_loss_rate(loss)

                        if bw > 0 and rtt_ms > 0:
                            get_network_estimator().seed_metrics(loss_rate=loss_rate, rtt_ms=rtt_ms, bandwidth_mbps=bw)
                            self.logger.info(f"🧪 Seeded estimator from network_condition: rate={rate}, delay={delay}, loss={loss}")
                except Exception as e:
                    self.logger.debug(f"Failed to seed NetworkEstimator: {e}")

        except ValueError as e:
            # Re-raise protocol validation errors
            raise RuntimeError(str(e))
        except Exception as e:
            self.logger.warning(f"Failed to load method settings from parameters.json: {e}. Using TCP defaults.")
            self.tcp_enabled = True
            self.udp_enabled = False
            self.rudp_enabled = False
            self.rudp_mode = "rudp"
            self.rudp_max_retries = 10
            self.udp_fec_m = 16
            self.udp_adaptive_fec = False
            self.compression_enabled = False
            self.initial_comp_method = "zlib"
            self.active_protocol = "TCP"

        # Load timeout configurations from config.json (set by ctrl server from execution_config.json)
        # Timeouts are REQUIRED - no defaults allowed
        try:
            with open("config.json", "r") as f:
                config = json.load(f)
                timeouts_config = config.get("timeouts")
                if timeouts_config is None:
                    raise ValueError("'timeouts' section is missing in config.json. Please configure timeouts in execution_config.json.")

                # Validate all required timeout values
                required_keys = ["model_fetch", "udp_initial_packet", "udp_inter_packet", "udp_model_completion"]
                missing_keys = [k for k in required_keys if k not in timeouts_config]
                if missing_keys:
                    raise ValueError(f"Missing required timeout keys in config.json: {missing_keys}")

                self.model_fetch_timeout = timeouts_config["model_fetch"]
                self.udp_initial_packet_timeout = timeouts_config["udp_initial_packet"]
                self.udp_inter_packet_timeout = timeouts_config["udp_inter_packet"]
                self.udp_model_completion_timeout = timeouts_config["udp_model_completion"]

        except FileNotFoundError:
            raise RuntimeError("config.json not found. Timeout configuration is required.")
        except ValueError as e:
            raise RuntimeError(str(e))
        except Exception as e:
            raise RuntimeError(f"Failed to load timeout settings from config.json: {e}")

        # Override the constructor timeout with configurable model_fetch_timeout
        # This applies to both TCP and UDP for fairness
        self.timeout = self.model_fetch_timeout

        # Log timeout configuration
        self.logger.info(f"⏱️ Model fetch timeout (TCP/UDP): {self.model_fetch_timeout}s")
        if self.udp_enabled:
            self.logger.info(f"⏱️ UDP timeouts: initial_packet={self.udp_initial_packet_timeout}s, inter_packet={self.udp_inter_packet_timeout}s, model_completion={self.udp_model_completion_timeout}s")

        # Initialize UDP and Compression
        if self.udp_enabled:
            self.logger.info(f"🚀 UDP Enabled (FEC M={self.udp_fec_m}, Mode=Auto)")
            self.udp_sharing = UDPModelSharing(
                self.addr,
                self.port,
                fec_m=self.udp_fec_m,
                timeout=self.udp_model_completion_timeout,
                inter_packet_timeout=self.udp_inter_packet_timeout,
                max_packet_size=self.udp_max_packet_size,
            )
            # Start listener callback for model data
            self.udp_sharing.start_listener(self._on_udp_model_received)
            # Register MDLREQ callback for pure UDP model requests
            self.udp_sharing.set_mdlreq_callback(self._on_udp_mdlreq)
        else:
            self.udp_sharing = None

        # Initialize RUDP/E-RUDP
        if self.rudp_enabled:
            # Load RUDP timeouts from config (use model_fetch timeout if not specified)
            # NOTE: WAFL parallel model fetch uses a strict ThreadPool wait() timeout.
            # If RUDP timeout is smaller than the model-completion budget, in-flight
            # transfers get cancelled mid-flight and inflate timeout_models.
            rudp_send_timeout = max(
                float(timeouts_config.get("rudp_send", self.model_fetch_timeout)),
                float(self.udp_model_completion_timeout),
            )

            mode_display = "E-RUDP" if self.rudp_mode == "erudp" else "RUDP"
            self.logger.info(f"🚀 {mode_display} Enabled (max_retries={self.rudp_max_retries}, dynamic params)")
            self.logger.info("   Aging/Window/FEC params are dynamically adjusted based on network conditions")

            self.rudp_sharing = RUDPModelSharing(
                ip=self.addr,
                port=self.port,
                mode=self.rudp_mode,
                timeout=rudp_send_timeout,
                max_retries=self.rudp_max_retries,
            )
            # Start listener callback for model data
            self.rudp_sharing.start_listener(self._on_rudp_model_received)
            # Register MDLREQ callback
            self.rudp_sharing.set_mdlreq_callback(self._on_rudp_mdlreq)
        else:
            self.rudp_sharing = None

            self.logger.info("UDP/RUDP Disabled (TCP mode)")

        # [OPTIMIZATION] Force-enable compression for Dynamic mode
        # Dynamic mode relies on compression to beat UDP benchmarks.
        if self.active_protocol == "dynamic":
            self.compression_enabled = True

        if self.compression_enabled:
            self.logger.info(f"🗜️ Compression Enabled (Initial: {self.initial_comp_method})")
            self.compression_manager = CompressionManager(initial_method=self.initial_comp_method)
        else:
            # Auto-enable compression (compression-only; no quantization) for low-quality networks
            # This is intentionally limited to UDP/RUDP where bandwidth dominates and where we
            # want UDP/RUDP to outperform TCP under fair/poor conditions.
            auto_enable = False
            try:
                measured_quality = _get_measured_network_quality()
                if (self.udp_enabled or self.rudp_enabled) and (measured_quality in ("fair", "poor")):
                    auto_enable = True
            except Exception:
                auto_enable = False

            if auto_enable:
                self.initial_comp_method = "adaptive" if self.initial_comp_method == "zlib" else self.initial_comp_method
                self.logger.info(f"🗜️ Compression Auto-Enabled for {self.active_protocol} (measured_quality={_get_measured_network_quality()}, initial={self.initial_comp_method})")
                self.compression_manager = CompressionManager(initial_method=self.initial_comp_method)
            else:
                self.logger.info("Compression Disabled")
                self.compression_manager = None

        # UDP Received models buffer
        self.received_models = {}  # {ip: data}
        self.received_models_lock = threading.Lock()
        self.received_models_cv = threading.Condition(self.received_models_lock)

        # Async Deserialization Executor
        self.executor = ProcessPoolExecutor(max_workers=2)

        # Smoothed Packet Loss (EWMA) for stable Adaptive FEC
        self.smoothed_packet_loss = 0.0
        self.packet_loss_alpha = 0.4  # EWMA weight - Higher for faster response to changing conditions
        self.previous_loss = 0.0  # For trend detection

        # Parity Ratchet with Decay: Prevents degradation but allows recovery
        # - max_observed_parity tracks highest needed Parity
        # - parity_decay_counter counts epochs since last increase
        # - After DECAY_EPOCHS, max_observed_parity decreases by 1
        self.max_observed_parity = 12  # Start with high Parity to maintain Survival
        self.parity_decay_counter = 0
        self.PARITY_DECAY_EPOCHS = 5  # Reset ratchet after N stable epochs (was 3, now more conservative)

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

        with self.received_models_cv:
            self.received_models[source_ip] = future
            self.received_models_cv.notify_all()

    def _on_udp_mdlreq(self, requester_ip: str):
        """
        Callback for handling UDP MDLREQ (pure UDP model request).
        Sends the current model via UDP to the requester.
        """
        try:
            self.logger.info(f"📨 Handling UDP MDLREQ from: {requester_ip}")

            # Get serialized model (use cache if available)
            model_data = self.vMODEL_INSTANCE
            if self.vMODEL_INSTANCE_CACHE is None:
                model_data = self._serialize_model(model_data)
                self.vMODEL_INSTANCE_CACHE = model_data
            else:
                model_data = self.vMODEL_INSTANCE_CACHE

            if model_data == b"ERROR" or model_data is None:
                self.logger.error(f"❌ Failed to serialize model for {requester_ip}")
                return

            # Send model via UDP
            if self.udp_sharing:
                success = self.udp_sharing.send_model(model_data, requester_ip, self.port)
                if success:
                    self.logger.debug(f"✅ Model sent via UDP to {requester_ip}")
                else:
                    self.logger.error(f"❌ Failed to send model via UDP to {requester_ip}")
            else:
                self.logger.error("❌ UDP sharing not available")

        except Exception as e:
            self.logger.error(f"Error handling UDP MDLREQ: {e}")

    def _on_rudp_model_received(self, data: bytes, source_ip: str):
        """Callback for RUDP/E-RUDP model reception."""
        self.logger.info(f"📦 Received RUDP model from {source_ip} ({len(data)} bytes) - Starting async deserialization")

        # Determine if compression is used (based on manager presence)
        use_compression = self.compression_manager is not None

        # Submit deserialization task to background process
        future = self.executor.submit(background_deserialize, data, use_compression)

        with self.received_models_cv:
            self.received_models[source_ip] = future
            self.received_models_cv.notify_all()

    def _on_rudp_mdlreq(self, requester_ip: str) -> bytes:
        """
        Callback for handling RUDP MDLREQ (RUDP model request).
        Returns the serialized model data to be sent back on the same connection.

        Returns:
            Serialized model data, or None if failed.
        """
        try:
            self.logger.info(f"📨 Handling RUDP MDLREQ from: {requester_ip}")

            # Get serialized model (use cache if available)
            model_data = self.vMODEL_INSTANCE
            if self.vMODEL_INSTANCE_CACHE is None:
                model_data = self._serialize_model(model_data)
                self.vMODEL_INSTANCE_CACHE = model_data
            else:
                model_data = self.vMODEL_INSTANCE_CACHE

            if model_data == b"ERROR" or model_data is None:
                self.logger.error(f"❌ Failed to serialize model for {requester_ip}")
                return None

            self.logger.debug(f"✅ Returning model data ({len(model_data)} bytes) for {requester_ip}")
            return model_data

        except Exception as e:
            self.logger.error(f"Error handling RUDP MDLREQ: {e}")
            return None

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

        In UDP mode, uses pure UDP for the entire exchange (no TCP handshake).
        In TCP mode, uses traditional TCP connection.
        """
        # === Dynamic Protocol Selection ===
        active_proto_for_call = str(self.active_protocol).upper()
        if self.active_protocol == "dynamic":
            try:
                metrics = get_network_estimator().get_metrics(peer_ip=peer_IP)
                quality = metrics.get_quality_level()
            except Exception:
                quality = "poor"

                active_proto_for_call = "UDP"
                self.logger.debug(f"⚖️ Dynamic: Using UDP for {peer_IP} (Quality: {quality})")

            # Count protocol choice for reporting (e.g. "TCP", "UDP")
            # Note: "E-RUDP" is counted as "RUDP" here unless we want separate?
            # active_proto_for_call is "UDP" or "TCP". RUDP check is below.
            # But wait, active_proto_for_call logic above is limiting.
            # If dynamic, we only set to UDP or TCP?
            # Ah, currently Dynamic Logic (lines 1046-1051) only chooses TCP or UDP?
            # RUDP is handled by use_rudp check below?
            # use_rudp = ("RUDP" in active_proto_for_call ...).
            # If active_proto_for_call is "UDP", use_rudp check sees no "RUDP".
            # So Dynamic logic currently only switches UDP/TCP?
            # "Poor" -> "UDP" -> use_rudp=False?
            # Wait, wafl_technical_spec says Poor -> RUDP.
            # I need to fix the Dynamic Logic to support RUDP selection too!

            # User Requirement: Excellent -> TCP, Good/Fair/Poor -> UDP
            # Note: "Good" quality can still have jitter causing TCP timeouts, so we use UDP for robustness.
            # RUDP is disabled in dynamic mode, so we fallback to UDP even for Poor.
            # User Requirement: Excellent -> TCP, Good -> TCP, Fair/Poor -> UDP
            # Note: Loss 2-5% (Good) is handled well by TCP retransmission.
            if quality in ["excellent", "good"]:
                active_proto_for_call = "TCP"
            else:
                active_proto_for_call = "UDP"

            self.logger.debug(f"⚖️ Dynamic: Using {active_proto_for_call} for {peer_IP} (Quality: {quality})")

            # Track count
            if active_proto_for_call in self.protocol_counts:
                self.protocol_counts[active_proto_for_call] += 1
            else:
                self.protocol_counts[active_proto_for_call] = 1  # Should not happen if init correct

        use_udp = active_proto_for_call == "UDP" and self.udp_sharing is not None
        use_rudp = ("RUDP" in active_proto_for_call or "E-RUDP" in active_proto_for_call) and self.rudp_sharing is not None

        # --- Pure UDP Mode: No TCP connection required ---
        if use_udp:
            try:
                self.logger.debug(f"📤 Sending pure UDP MDLREQ to peer: {peer_IP}")

                # Send UDP MDLREQ directly (no TCP connection)
                if not self.udp_sharing.send_mdlreq(peer_IP, self.port, self.addr):
                    self.logger.warning(f"❌ Failed to send UDP MDLREQ to {peer_IP}")
                    return False, b"ERROR"

                self.logger.debug(f"📡 UDP MDLREQ sent to {peer_IP}, waiting for UDP model...")

                # Wait for UDP data using configured timeout with Aggressive MDLREQ Retransmission
                start_wait = time.time()
                deadline = start_wait + float(self.udp_model_completion_timeout)
                last_req_time = start_wait
                RESEND_INTERVAL = 1.5  # Retransmit MDLREQ every 1.5s if no model received

                data_or_future = None
                with self.received_models_cv:
                    while peer_IP not in self.received_models:
                        now = time.time()
                        remaining = deadline - now
                        if remaining <= 0:
                            break

                        # Aggressive Retransmission: If waiting too long, resend MDLREQ
                        # But ONLY if we haven't received any data recently (streaming is not active)
                        if now - last_req_time > RESEND_INTERVAL:
                            last_activity = self.udp_sharing.get_last_peer_activity(peer_IP)
                            is_active = (now - last_activity) < 1.0

                            if not is_active:
                                self.logger.debug(f"🔄 Resending UDP MDLREQ to {peer_IP} (idle for {now - last_activity:.1f}s)")
                                self.udp_sharing.send_mdlreq(peer_IP, self.port, self.addr)
                                last_req_time = now
                            else:
                                # Transfer is active, just reset timer to avoid spamming
                                last_req_time = now

                        # Short wait to allow checking retransmission timer
                        wait_slice = min(0.2, remaining, max(0.1, RESEND_INTERVAL - (now - last_req_time)))
                        self.received_models_cv.wait(timeout=wait_slice)

                    if peer_IP in self.received_models:
                        data_or_future = self.received_models.pop(peer_IP)

                if data_or_future is not None:
                    elapsed = time.time() - start_wait
                    self.logger.debug(f"📥 UDP model received from {peer_IP} in {elapsed:.2f}s, deserializing...")

                    # Handle async deserialization (Future) or raw bytes
                    if isinstance(data_or_future, (bytes, bytearray)):
                        return True, self._deserialize_model(data_or_future)
                    try:
                        deserialized_output = data_or_future.result(timeout=10.0)
                        if isinstance(deserialized_output, bytes) and deserialized_output.startswith(b"ERROR"):
                            self.logger.error("Async deserialization returned ERROR")
                            return False, b"ERROR"
                        return True, deserialized_output
                    except Exception as e:
                        self.logger.error(f"Async deserialization failed or timed out: {e}")
                        return False, b"ERROR"

                # Timeout waiting for UDP model - NO TCP fallback
                self.logger.warning(f"⚠️ UDP MDLREQ timeout waiting for {peer_IP} (>{self.udp_model_completion_timeout}s)")
                return False, b"ERROR"

            except Exception as e:
                self.logger.error(f"Error in pure UDP MDLREQ: {e}")
                return False, b"ERROR"

        # --- RUDP/E-RUDP Mode: Reliable UDP with ARQ ---
        if use_rudp:
            try:
                mode_name = "E-RUDP" if self.rudp_mode == "erudp" else "RUDP"
                self.logger.debug(f"📤 Sending {mode_name} MDLREQ to peer: {peer_IP}")

                # Send RUDP MDLREQ
                if not self.rudp_sharing.send_mdlreq(peer_IP, self.port, self.addr):
                    self.logger.warning(f"❌ Failed to send {mode_name} MDLREQ to {peer_IP}")
                    return False, b"ERROR"

                self.logger.debug(f"📡 {mode_name} MDLREQ sent to {peer_IP}, waiting for model...")

                # Wait for RUDP data using configured timeout (no busy-wait)
                start_wait = time.time()
                # NOTE: RUDPModelSharing.send_mdlreq() performs the full request/response exchange
                # and uses its own completion timeout. Keep this wait aligned to avoid artificial
                # failures when model_fetch_timeout is shorter than the transport budget.
                transport_timeout = float(getattr(self.rudp_sharing, "timeout", 0.0) or 0.0)
                deadline = start_wait + max(float(self.model_fetch_timeout), transport_timeout)
                data_or_future = None
                with self.received_models_cv:
                    while peer_IP not in self.received_models:
                        remaining = deadline - time.time()
                        if remaining <= 0:
                            break
                        self.received_models_cv.wait(timeout=min(0.2, remaining))
                    if peer_IP in self.received_models:
                        data_or_future = self.received_models.pop(peer_IP)

                if data_or_future is not None:
                    elapsed = time.time() - start_wait
                    self.logger.debug(f"📥 {mode_name} model received from {peer_IP} in {elapsed:.2f}s, deserializing...")

                    # Handle async deserialization (Future) or raw bytes
                    if isinstance(data_or_future, (bytes, bytearray)):
                        return True, self._deserialize_model(data_or_future)
                    try:
                        deserialized_output = data_or_future.result(timeout=10.0)
                        if isinstance(deserialized_output, bytes) and deserialized_output.startswith(b"ERROR"):
                            self.logger.error("Async deserialization returned ERROR")
                            return False, b"ERROR"
                        return True, deserialized_output
                    except Exception as e:
                        self.logger.error(f"Async deserialization failed or timed out: {e}")
                        return False, b"ERROR"

                # Timeout waiting for RUDP model - NO TCP fallback
                self.logger.warning(f"⚠️ {mode_name} MDLREQ timeout waiting for {peer_IP} (>{max(float(self.model_fetch_timeout), transport_timeout):.1f}s)")
                return False, b"ERROR"

            except Exception as e:
                self.logger.error(f"Error in RUDP MDLREQ: {e}")
                return False, b"ERROR"

        # --- TCP Mode: Traditional TCP connection ---
        # Fallback if protocols above not selected
        # If RUDP is enabled but here? RUDP block ends with return.
        # So this corresponds to TCP (or fallback).
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                self.logger.debug(f"📥 Requesting WAFL model from peer (TCP): {str(peer_IP)}")

                command = f"{ModelSharingUtils.cMDLREQ}:src={self.addr}{other_options}\r\n"
                self.logger.debug(f"📤 Sending MDLREQ command: {command.strip()}")

                # Use configured timeout for connection
                s.settimeout(self.timeout)
                start_tcp_transfer = time.time()  # Start timing for metrics
                s.connect((peer_IP, self.port))
                self.logger.debug(f"🔌 Connected to {peer_IP}:{self.port}, timeout={self.timeout}s")
                s.sendall(command.encode("utf-8"))

                # Overall transfer deadline
                # Keep a single strict deadline for fairness across transports.
                transfer_deadline = time.time() + float(self.timeout)

                # Receive model data via TCP
                data = []
                bytes_received_in_loop = 0

                while True:
                    current_time = time.time()
                    if current_time > transfer_deadline:
                        self.logger.warning(f"⏱️ TCP strict deadline exceeded after receiving {bytes_received_in_loop} bytes")
                        raise TimeoutError(f"TCP transfer deadline exceeded ({self.timeout}s)")

                    remaining_time = transfer_deadline - current_time
                    s.settimeout(min(remaining_time, 0.5))

                    try:
                        packet = s.recv(4096)
                        if not packet:
                            self.logger.debug(f"📥 TCP transfer complete: {bytes_received_in_loop} bytes total")
                            break
                        data.append(packet)
                        bytes_received_in_loop += len(packet)
                    except socket.timeout:
                        if time.time() > transfer_deadline:
                            self.logger.warning(f"⏱️ TCP strict deadline exceeded during recv ({bytes_received_in_loop} bytes received)")
                            raise TimeoutError(f"TCP transfer deadline exceeded ({self.timeout}s)")
                        continue

                data = b"".join(data)
                self.tcp_stats["bytes_received"] += len(data)
                self.tcp_stats["models_received"] += 1
                self.logger.debug(f"📊 TCP stats updated: received={len(data)} bytes")

                # Record successful transfer metrics
                tcp_duration = time.time() - start_tcp_transfer
                get_network_estimator().record_transfer(len(data), tcp_duration, peer_IP)
                get_network_estimator().record_packet_result(True, peer_IP)

            data = self._deserialize_model(data)
            if data == b"ERROR" or data is None:
                raise Exception("FETCH ERROR")
            self.logger.debug(f"✅ Model fetched from {peer_IP} successfully")
            return True, data
        except TimeoutError as te:
            self.logger.warning(f"⏱️ TCP timeout in _fetch_model: {te}")
            get_network_estimator().record_packet_result(False, peer_IP)
            return False, b"ERROR"
        except Exception as exc:
            self.logger.error(f"The following error occurred in _fetch_model: {str(exc)[:100]}...")
            get_network_estimator().record_packet_result(False, peer_IP)
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

    def get_parallel_fetch_timeout(self) -> float:
        """Return the strict timeout used by the WAFL ThreadPool wait().

        This must be >= the transport's own completion/receive timeout; otherwise
        the WAFL loop cancels in-flight UDP/RUDP transfers and survival/timeout
        metrics become artificially worse.
        """
        base = float(self.timeout) if self.timeout is not None else 0.0

        if self.udp_enabled and self.udp_sharing is not None:
            udp_timeout = float(getattr(self.udp_sharing, "timeout", 0.0) or 0.0)
            udp_completion = float(getattr(self, "udp_model_completion_timeout", 0.0) or 0.0)
            return max(base, udp_timeout, udp_completion)

        if self.rudp_enabled and self.rudp_sharing is not None:
            rudp_timeout = float(getattr(self.rudp_sharing, "timeout", 0.0) or 0.0)
            return max(base, rudp_timeout)

        return base

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

        # Epoch hook: ensure RUDP reuses connections within the same epoch only.
        if self.rudp_enabled and self.rudp_sharing is not None:
            try:
                self.rudp_sharing.begin_epoch(epoch_number)
            except Exception as e:
                self.logger.warning(f"Failed to notify RUDP epoch boundary: {e}")
        self.logger.debug(f"Updated model instance (epoch={epoch_number})")

    def on_epoch_start(self, *, epoch: int, planned_peer_ips: list[str]) -> None:
        """Epoch開始時の検証用ログと、プロトコル層への通知。"""
        try:
            m = get_network_estimator().get_metrics()
            self.logger.info(
                f"📣 Epoch start: epoch={epoch}, protocol={self.active_protocol}, planned_peers={len(planned_peer_ips)}, measured(q={m.get_quality_level()}, loss={m.packet_loss_rate * 100:.2f}%, rtt={m.rtt_ms:.1f}ms, bw={m.bandwidth_mbps:.2f}Mbps)"
            )
        except Exception:
            self.logger.info(f"📣 Epoch start: epoch={epoch}, protocol={self.active_protocol}, planned_peers={len(planned_peer_ips)}")

        if self.udp_enabled and self.udp_sharing is not None:
            try:
                self.logger.info(f"📶 UDP runtime: k={self.udp_sharing.k}, payload={self.udp_sharing.PAYLOAD_SIZE}B (Dynamic Pacing/Parity)")
            except Exception:
                pass

        if self.rudp_enabled and self.rudp_sharing is not None:
            try:
                pool_stats = self.rudp_sharing.get_connection_pool_stats()
                addrs = pool_stats.get("addresses", []) or []
                addr_sample = [f"{a[0]}:{a[1]}" for a in addrs[:5]]
                self.logger.info(f"📶 RUDP pool (pre-epoch): active={pool_stats.get('active_connections', 0)}, sample={addr_sample}")
            except Exception:
                pass

        if self.compression_manager is not None:
            try:
                cstats = self.compression_manager.get_stats()
                self.logger.info(f"🗜️ Compression runtime: method={cstats.get('method')}, ratio={cstats.get('compression_ratio'):.3f}, bw_est={cstats.get('bandwidth_est_mbps'):.2f}MB/s")
            except Exception:
                pass

        # Reset epoch statistics for the new epoch
        self._reset_epoch_accumulators()

    def on_epoch_end(self, *, epoch: int, next_epoch_peer_ips: list[str]) -> None:
        """Epoch終了時の検証用ログと、次epochに向けた接続剪定（RUDP）。"""
        if self.rudp_enabled and self.rudp_sharing is not None:
            try:
                prune_stats = self.rudp_sharing.prune_for_next_epoch(next_epoch=epoch + 1, keep_peer_ips=next_epoch_peer_ips)
                pool_stats = self.rudp_sharing.get_connection_pool_stats()
                keep_sample = sorted({ip for ip in next_epoch_peer_ips if ip})[:5]
                self.logger.info(
                    f"📦 RUDP epoch end: epoch={epoch}, next_keep={len(next_epoch_peer_ips)} (sample={keep_sample}), "
                    f"before={prune_stats.get('before')}, after={prune_stats.get('after')}, kept={prune_stats.get('kept')}, "
                    f"closed_unhealthy={prune_stats.get('closed_unhealthy')} (sample={prune_stats.get('closed_unhealthy_sample')}), "
                    f"closed_unneeded={prune_stats.get('closed_unneeded')} (sample={prune_stats.get('closed_unneeded_sample')}), "
                    f"pool_after={pool_stats.get('active_connections', 0)} (kept_sample={prune_stats.get('kept_sample')})"
                )
            except Exception as e:
                self.logger.warning(f"RUDP prune_for_next_epoch failed: {e}")

        # Emit compact per-epoch protocol summary for verification
        try:
            if self.udp_enabled and self.udp_sharing is not None:
                s = self.udp_sharing.stats
                self.logger.info(
                    f"📊 UDP summary: sent={s.get('sent_models')}, failed={s.get('sent_failed')}, received={s.get('received_models')}, "
                    f"timeout_models={s.get('timeout_models')}, fec_ok={s.get('fec_recovery_success')}, fec_fail={s.get('fec_recovery_fail')}, chunks_decoded={s.get('chunks_decoded')}, bytes_sent={s.get('bytes_sent')}, bytes_recv={s.get('bytes_received')}"
                )
            elif self.rudp_enabled and self.rudp_sharing is not None:
                s = self.rudp_sharing.get_stats()
                self.logger.info(
                    f"📊 RUDP summary: sent={s.get('sent_models')}, failed={s.get('sent_failed')}, received={s.get('received_models')}, "
                    f"timeout_models={s.get('timeout_models')}, retrans={s.get('retransmissions')}, max_retries_reached={s.get('max_retries_reached')}, bytes_sent={s.get('bytes_sent')}, bytes_recv={s.get('bytes_received')}"
                )
            else:
                s = self.tcp_stats
                self.logger.info(f"📊 TCP summary: sent_models={s.get('models_sent')}, recv_models={s.get('models_received')}, fetch_failed={s.get('fetch_failed')}, bytes_sent={s.get('bytes_sent')}, bytes_recv={s.get('bytes_received')}")
        except Exception:
            pass

        if self.compression_manager is not None:
            try:
                estats = self.compression_manager.get_epoch_stats()
                cstats = self.compression_manager.get_stats()
                self.logger.info(
                    f"🗜️ Compression epoch summary: method={cstats.get('method')}, ratio={estats.get('ratio', 1.0):.3f}, "
                    f"count={estats.get('compression_count', 0)}, orig={estats.get('original_size', 0)}, comp={estats.get('compressed_size', 0)}, time_ms={estats.get('time_ms', 0):.1f}"
                )
            except Exception:
                pass

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
        # Note: In Dynamic/UDP mode, this captures high-level timeouts (e.g. MDLREQ failure)
        # that are not captured by UDPModelSharing (which only sees partial packet loss).
        if not FETCHED:
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

    def _read_tcp_retransmit_stats(self) -> int:
        """
        Read TCP retransmit segment count from /proc/net/snmp.
        Returns the total number of retransmitted TCP segments.
        This captures TCP's 'hidden' communication cost that affects epoch duration.
        """
        try:
            with open("/proc/net/snmp", "r") as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                if line.startswith("Tcp:") and i + 1 < len(lines):
                    # Header line followed by values line
                    headers = lines[i].split()
                    values = lines[i + 1].split()
                    if "RetransSegs" in headers:
                        idx = headers.index("RetransSegs")
                        return int(values[idx])
            return 0
        except Exception:
            return 0

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
        # --- Survival Rate and Model Counts (Application Layer) ---
        # --- Survival Rate and Model Counts (Application Layer) ---
        # Initialize accumulators for aggregated metrics
        agg_sent_models = 0
        agg_sent_failed = 0
        agg_received_models = 0
        agg_timeout_models = 0
        agg_app_bytes_sent = 0
        agg_app_bytes_received = 0

        # Protocol-specific metrics to merge later
        proto_specific_metrics = {}

        # 1. RUDP/E-RUDP Statistics
        if self.rudp_sharing is not None:
            current_rudp = self.rudp_sharing.get_stats()
            delta_rudp = self._get_delta_stats(current_rudp, self._last_reported_rudp)
            self._last_reported_rudp = current_rudp.copy()

            agg_sent_models += delta_rudp.get("sent_models", 0)
            agg_sent_failed += delta_rudp.get("sent_failed", 0)
            agg_received_models += delta_rudp.get("received_models", 0)
            agg_timeout_models += delta_rudp.get("timeout_models", 0)
            agg_app_bytes_sent += delta_rudp.get("bytes_sent", 0)
            agg_app_bytes_received += delta_rudp.get("bytes_received", 0)

            proto_specific_metrics.update(
                {
                    "fec_recovery_success": delta_rudp.get("fec_recoveries", 0),
                    "fec_recovery_fail": 0,
                    # RUDP-specific metrics
                    "rudp_retransmissions": delta_rudp.get("retransmissions", 0),
                    "rudp_acks_sent": delta_rudp.get("acks_sent", 0),
                    "rudp_acks_received": delta_rudp.get("acks_received", 0),
                    "rudp_eaks_sent": delta_rudp.get("eaks_sent", 0),
                    "rudp_eaks_received": delta_rudp.get("eaks_received", 0),
                    "rudp_aged_packets": delta_rudp.get("aged_packets", 0),
                    # Gauges
                    "rudp_connect_time_ms": current_rudp.get("connect_time_ms", 0),
                    "rudp_avg_rtt_ms": current_rudp.get("avg_rtt_ms", 0),
                    "rudp_max_retries_reached": delta_rudp.get("max_retries_reached", 0),
                    "rudp_nacks_sent": delta_rudp.get("nacks_sent", 0),
                }
            )

        # 2. UDP Statistics
        if self.udp_sharing is not None:
            current_udp = self.udp_sharing.stats
            delta_udp = self._get_delta_stats(current_udp, self._last_reported_udp)
            self._last_reported_udp = current_udp.copy()

            agg_sent_models += delta_udp.get("sent_models", 0)
            agg_sent_failed += delta_udp.get("sent_failed", 0)
            agg_received_models += delta_udp.get("received_models", 0)
            agg_timeout_models += delta_udp.get("timeout_models", 0)
            agg_app_bytes_sent += delta_udp.get("bytes_sent", 0)
            agg_app_bytes_received += delta_udp.get("bytes_received", 0)

            # For specific metrics, we might overwrite RUDP ones if both active (unlikely)
            # Use distinct keys or prioritizing UDP if Dynamic
            proto_specific_metrics.update(
                {
                    "fec_recovery_success": delta_udp.get("fec_recovery_success", 0),
                    "fec_recovery_fail": delta_udp.get("fec_recovery_fail", 0),
                    "fec_encode_time_ms": delta_udp.get("fec_encode_time_ms", 0),
                    "fec_decode_time_ms": delta_udp.get("fec_decode_time_ms", 0),
                    "udp_avg_parity": delta_udp.get("sum_parity", 0) / delta_udp.get("total_transfers", 1) if delta_udp.get("total_transfers", 0) > 0 else 0,
                    "udp_avg_pacing_ms": (delta_udp.get("sum_pacing_delay", 0.0) / delta_udp.get("total_transfers", 1) * 1000) if delta_udp.get("total_transfers", 0) > 0 else 0,
                }
            )

        # 3. TCP Statistics (Always processed as fallback or primary)
        current_tcp = self.tcp_stats
        delta_tcp = self._get_delta_stats(current_tcp, self._last_reported_tcp)
        self._last_reported_tcp = current_tcp.copy()

        agg_sent_models += delta_tcp.get("models_sent", 0)
        # TCP does not track sent_failed explicitly in this dict, assume 0
        agg_received_models += delta_tcp.get("models_received", 0)
        agg_timeout_models += delta_tcp.get("fetch_failed", 0)
        agg_app_bytes_sent += delta_tcp.get("bytes_sent", 0)
        agg_app_bytes_received += delta_tcp.get("bytes_received", 0)

        # Survival Rate Calculation (Aggregate)
        # Survival = Received / (Received + Timeouts)
        total_fetch_attempts = agg_received_models + agg_timeout_models
        survival_rate = agg_received_models / total_fetch_attempts if total_fetch_attempts > 0 else 1.0

        # Update Metrics
        metrics.update(
            {
                "survival_rate": survival_rate,
                "sent_models": agg_sent_models,
                "sent_failed": agg_sent_failed,
                "received_models": agg_received_models,
                "timeout_models": agg_timeout_models,
                "app_bytes_sent": agg_app_bytes_sent,
                "app_bytes_received": agg_app_bytes_received,
            }
        )

        # Merge protocol specific metrics
        metrics.update(proto_specific_metrics)

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

        # Log packet loss for observation (adaptive adjustment disabled)
        if self.udp_sharing is not None:
            packet_loss = self._calculate_packet_loss()
            if packet_loss > 0:
                self.logger.info(f"📊 Epoch packet loss: {packet_loss:.1%}")
            # Dynamic parameter adjustment is only enabled for UDP adaptive-FEC mode.
            if getattr(self, "udp_adaptive_fec", False):
                self._update_adaptive_fec(packet_loss)

        # --- Protocol Counts (Dynamic Mode) ---
        metrics.update(
            {
                "protocol_tcp_count": self.protocol_counts.get("TCP", 0),
                "protocol_udp_count": self.protocol_counts.get("UDP", 0),
                "protocol_rudp_count": self.protocol_counts.get("RUDP", 0),
            }
        )

        return metrics

    def _reset_epoch_accumulators(self):
        """Reset epoch-level statistics accumulators (called at start of epoch)."""
        # Reset TCP stats
        self.tcp_stats = {
            "bytes_sent": 0,
            "bytes_received": 0,
            "models_sent": 0,
            "models_received": 0,
            "fetch_failed": 0,
        }
        # Reset UDP stats if enabled
        if self.udp_sharing is not None:
            # Re-initialize stats dict to zero
            self.udp_sharing.stats = {k: 0 for k in self.udp_sharing.stats}
            # Keep float keys as float if needed, but 0 is fine for int/float add.
            self.udp_sharing.stats.update(
                {
                    "fec_encode_time_ms": 0.0,
                    "fec_decode_time_ms": 0.0,
                }
            )

        # Reset RUDP stats if enabled
        if self.rudp_sharing is not None:
            self.rudp_sharing.stats = {k: 0 for k in self.rudp_sharing.stats}
            self.rudp_sharing.stats.update(
                {
                    "connect_time_ms": 0.0,
                    "avg_rtt_ms": 0.0,
                }
            )

        # Reset compression stats if enabled
        if self.compression_manager is not None:
            self.compression_manager.reset_epoch_stats()

        # Reset last reported to 0 so deltas start from 0 for this epoch?
        # No, if we reset stats to 0, we must reset last_reported to 0 too.
        self._last_reported_tcp = {}
        self._last_reported_udp = {}
        self._last_reported_rudp = {}
        # Reset protocol counts for dynamic mode
        self.protocol_counts = {"TCP": 0, "UDP": 0, "RUDP": 0}

    def _get_delta_stats(self, current: dict, last: dict) -> dict:
        """Compute delta between current cumulative stats and last reported."""
        delta = {}
        for k, v in current.items():
            if isinstance(v, (int, float)):
                delta[k] = v - last.get(k, 0)
        return delta

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
        # NOTE: UDPModelSharing historically incremented total_chunks_received per packet.
        # Use chunk-level counter when available for a meaningful denominator.
        total_chunks = stats.get("chunks_decoded", 0) or stats.get("total_chunks_received", 0)
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

        # Feed epoch-level loss observation into the shared NetworkEstimator.
        # This keeps a single source of truth for parity/pacing/window recommendations.
        get_network_estimator().record_loss_rate_sample(packet_loss)

        # In auto mode we keep k fixed and allow parity/pacing to follow estimator recommendations.
        target_k = 16
        m = get_network_estimator().get_metrics()

        # DYNAMIC TIMEOUT ADJUSTMENT
        # Keep timeout consistent with estimator RTT to avoid premature cancellation.
        # Reduced base timeouts to improve Epoch Duration (fail fast & retry).
        # With aggressive MDLREQ retransmission, we don't need long waits for initial packet.
        if packet_loss < 0.05:
            dynamic_timeout = 5.0  # Was 10.0
        elif packet_loss < 0.15:
            dynamic_timeout = 6.0  # Was 8.0
        else:
            dynamic_timeout = 8.0  # Was 6.0 (High loss needs more time for FEC/Reconstruction)

        rtt_sec = (m.rtt_ms / 1000.0) if m.rtt_ms > 0 else 0.1
        # Ensure timeout allows for at least 15 RTTs (plenty for retransmissions if using ARQ, or just margin)
        dynamic_timeout = max(dynamic_timeout, rtt_sec * 15)

        # Bandwidth Safety Check: Ensure timeout is enough to transfer model at current BW
        # Assume approx 2MB model size (16Mb) + 50% margin
        if m.bandwidth_mbps > 0:
            estimated_transfer_time = (2.0 * 8) / m.bandwidth_mbps
            min_bw_timeout = estimated_transfer_time * 1.5
            dynamic_timeout = max(dynamic_timeout, min_bw_timeout)

        # Update UDPModelSharing timeout if changed significantly
        current_timeout = self.udp_sharing.timeout
        if abs(dynamic_timeout - current_timeout) > 1.0:
            self.udp_sharing.timeout = dynamic_timeout
            self.udp_model_completion_timeout = dynamic_timeout  # Also update local timeout
            self.logger.info(f"⏱️ Dynamic Timeout: {current_timeout:.1f}s -> {dynamic_timeout:.1f}s (loss={packet_loss:.1%})")

        # Sync parity to estimator recommendation (optional but makes changes visible immediately).
        # Sync parity to estimator recommendation
        # NOTE: UDPModelSharing now pulls directly from NetworkEstimator per-peer.
        # We don't need to push parity back to UDPModelSharing.
        # Just logging the recommendation for debug visibility.
        if self.udp_fec_m != target_k:
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
        file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(file_handler)
        self.logger.info(f"📝 Log file initialized: {log_file_path}")

        # Initialize Metrics Logger (Moved before setup_local_wafl_node so it can be passed/used)
        # User requested Project Name to simply be "WAFL-Testbed", grouping by Experiment ID.
        self.metrics_logger = MetricsLogger(self.experiment_id, self.name, self.start_timestamp, project_name="WAFL-Testbed")

        # Setup the WAFL node before starting the control thread
        self.setup_local_wafl_node(self.agent_index, self.name)

        self.ctrl_listener_thread = threading.Thread(target=self.wait_ctrl, daemon=False, name="CTRL_TCP_Listener")
        self.ctrl_listener_thread.start()

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

            self.experiment_name = experiment_info.get("experiment_name", "WAFL-Testbed")

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

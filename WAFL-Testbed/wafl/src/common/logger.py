"""Metrics logging utilities."""

import csv
import os
import time

import wandb


class MetricsLogger:
    """Extended metrics logger that writes to CSV with support for SSP, UDP, and compression metrics."""

    # Define all possible metric columns for consistent CSV structure
    ALL_COLUMNS = [
        "timestamp",
        "phase",
        "epoch",
        # Core training metrics
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
        # Timing metrics
        "epoch_duration_ms",
        "compute_time_ms",  # Pure computation time (training)
        "comm_time_ms",  # Communication time (model exchange)
        "waiting_time_ms",  # Waiting time (parallel fetch)
        # SSP metrics
        "wasted_ms",
        "wasted_norm",
        "batches_processed",
        "was_force_stopped",
        # UDP/FEC metrics
        "survival_rate",
        "sent_models",
        "sent_failed",
        "received_models",
        "fec_recovery_success",
        "fec_recovery_fail",
        "fec_encode_time_ms",  # FEC エンコード処理時間
        "fec_decode_time_ms",  # FEC デコード処理時間
        "bytes_sent",
        "bytes_received",
        "app_bytes_sent",  # Application payload sent (metric for useful data)
        "app_bytes_received",  # Application payload received (metric for goodput)
        "app_bytes_received",  # Application payload received (metric for goodput)
        "timeout_models",  # 追加
        "fast_mode_discarded",  # 追加: Fast modeで破棄されたモデル数
        # RUDP/E-RUDP metrics
        "rudp_retransmissions",
        "rudp_acks_sent",
        "rudp_acks_received",
        "rudp_eaks_sent",
        "rudp_eaks_received",
        "rudp_aged_packets",
        "rudp_connect_time_ms",
        "rudp_avg_rtt_ms",
        "rudp_max_retries_reached",
        "rudp_nacks_sent",  # 追加
        "fec_recoveries",  # 追加
        # Compression metrics
        "compression_method",
        "compression_ratio",
        "compression_time_ms",
        "original_size",
        "compressed_size",
        # Dynamic Protocol Counts
        "protocol_tcp_count",
        "protocol_udp_count",
        "protocol_rudp_count",
        # UDP Dynamic Params
        "udp_avg_parity",
        "udp_avg_pacing_ms",
    ]

    def __init__(self, experiment_id: str, node_name: str, start_timestamp: float = None, project_name: str = "WAFL-Testbed"):
        """Initialize metrics logger.

        Args:
            experiment_id: The experiment ID
            node_name: Node name
            start_timestamp: Experiment start timestamp (epoch)
            project_name: WandB project name (Experiment Name)
        """
        self.experiment_id = experiment_id
        self.node_name = node_name

        if start_timestamp is not None:
            self.start_time = start_timestamp
        else:
            # Fallback: Parse start time from experiment ID
            try:
                ts_str = experiment_id.split("-")[-1]
                struct_time = time.strptime(ts_str, "%Y%m%dT%H%M%S")
                self.start_time = time.mktime(struct_time)
            except Exception:
                self.start_time = time.time()

        # Create results directory
        results_dir = f"./results/{experiment_id}"
        os.makedirs(results_dir, exist_ok=True)

        # CSV file
        self.csv_path = os.path.join(results_dir, "metrics.csv")
        self.header_written = False

        # Initialize header with all columns
        self._init_csv()

        # Initialize WandB
        try:
            wandb.init(
                project=project_name,
                name=f"{node_name}",
                group=experiment_id,
                config={
                    "experiment_id": experiment_id,
                    "node_name": node_name,
                    "start_time": self.start_time,
                },
                reinit=True,
            )
        except Exception as e:
            print(f"Failed to initialize wandb: {e}")

    def _init_csv(self):
        """Initialize CSV file with all possible columns."""
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            print(f"Creating new metrics.csv at {self.csv_path}")
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.ALL_COLUMNS)
                writer.writeheader()
        else:
            print(f"Appending to existing metrics.csv at {self.csv_path}")
        self.header_written = True

    def _write_row(self, entry: dict):
        """Write a row to CSV, filling missing columns with empty values."""
        # Ensure all columns are present
        full_entry = {col: entry.get(col, "") for col in self.ALL_COLUMNS}
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.ALL_COLUMNS)
            writer.writerow(full_entry)
            f.flush()
            os.fsync(f.fileno())

        # Log to WandB
        try:
            # Convert empty strings to None or 0 where appropriate, or let wandb handle it.
            # Wandb handles dicts well. We should filter out internal CSV artifacts if any.
            # But here `entry` is cleaner than `full_entry` because it doesn't have empty strings for missing values yet.
            wandb.log(entry)
        except Exception:
            pass

    def log_epoch(self, phase: str, epoch: int, metrics: dict):
        """Log all metrics for an epoch to CSV.

        Args:
            phase: Training phase (SELF or WAFL)
            epoch: Epoch number
            metrics: Dictionary of metrics. All possible fields:
                Core training:
                    - train_loss, train_accuracy, test_loss, test_accuracy
                Timing:
                    - epoch_duration_ms
                SSP:
                    - wasted_ms, wasted_norm, batches_processed, was_force_stopped
                UDP/FEC:
                    - survival_rate, sent_models, sent_failed, received_models,
                      fec_recovery_success, fec_recovery_fail, bytes_sent, bytes_received
                Compression:
                    - compression_method, compression_ratio, compression_time_ms,
                      original_size, compressed_size

                Missing fields will be recorded as 0 for numeric, "" for string.
        """
        # Calculate relative time from experiment start
        relative_time = time.time() - self.start_time

        # Build entry with defaults for all columns
        entry = {
            "timestamp": relative_time,
            "phase": phase,
            "epoch": epoch,
            # Core training (provided or empty)
            "train_loss": metrics.get("train_loss", ""),
            "train_accuracy": metrics.get("train_accuracy", ""),
            "test_loss": metrics.get("test_loss", ""),
            "test_accuracy": metrics.get("test_accuracy", ""),
            # Timing
            "epoch_duration_ms": metrics.get("epoch_duration_ms", ""),
            "compute_time_ms": metrics.get("compute_time_ms", ""),
            "comm_time_ms": metrics.get("comm_time_ms", 0),
            "waiting_time_ms": metrics.get("waiting_time_ms", 0),
            # SSP metrics - default to 0 so graphs can show "no waste"
            "wasted_ms": metrics.get("wasted_ms", 0),
            "wasted_norm": metrics.get("wasted_norm", 0),
            "batches_processed": metrics.get("batches_processed", 0),
            "was_force_stopped": metrics.get("was_force_stopped", 0),
            # UDP/FEC metrics - default to 0 for numeric
            "survival_rate": metrics.get("survival_rate", 1.0),  # Default 100% if no UDP
            "sent_models": metrics.get("sent_models", 0),
            "sent_failed": metrics.get("sent_failed", 0),
            "received_models": metrics.get("received_models", 0),
            "fec_recovery_success": metrics.get("fec_recovery_success", 0),
            "fec_recovery_fail": metrics.get("fec_recovery_fail", 0),
            "fec_encode_time_ms": metrics.get("fec_encode_time_ms", 0),
            "fec_decode_time_ms": metrics.get("fec_decode_time_ms", 0),
            "bytes_sent": metrics.get("bytes_sent", 0),
            "bytes_received": metrics.get("bytes_received", 0),
            "app_bytes_sent": metrics.get("app_bytes_sent", 0),
            "app_bytes_received": metrics.get("app_bytes_received", 0),
            "timeout_models": metrics.get("timeout_models", 0),
            "fast_mode_discarded": metrics.get("fast_mode_discarded", 0),
            # UDP Dynamic Params
            "udp_avg_parity": metrics.get("udp_avg_parity", 0),
            "udp_avg_pacing_ms": metrics.get("udp_avg_pacing_ms", 0),
            # RUDP specific
            "rudp_nacks_sent": metrics.get("rudp_nacks_sent", 0),
            "fec_recoveries": metrics.get("fec_recoveries", 0),
            "rudp_retransmissions": metrics.get("rudp_retransmissions", 0),
            "rudp_acks_sent": metrics.get("rudp_acks_sent", 0),
            "rudp_acks_received": metrics.get("rudp_acks_received", 0),
            "rudp_eaks_sent": metrics.get("rudp_eaks_sent", 0),
            "rudp_eaks_received": metrics.get("rudp_eaks_received", 0),
            "rudp_aged_packets": metrics.get("rudp_aged_packets", 0),
            "rudp_connect_time_ms": metrics.get("rudp_connect_time_ms", 0),
            "rudp_avg_rtt_ms": metrics.get("rudp_avg_rtt_ms", 0),
            "rudp_max_retries_reached": metrics.get("rudp_max_retries_reached", 0),
            # Compression metrics
            "compression_method": metrics.get("compression_method", "none"),
            "compression_ratio": metrics.get("compression_ratio", 1.0),
            "compression_time_ms": metrics.get("compression_time_ms", 0),
            "original_size": metrics.get("original_size", 0),
            "compressed_size": metrics.get("compressed_size", 0),
            # Protocol Counts
            "protocol_tcp_count": metrics.get("protocol_tcp_count", 0),
            "protocol_udp_count": metrics.get("protocol_udp_count", 0),
            "protocol_rudp_count": metrics.get("protocol_rudp_count", 0),
        }
        self._write_row(entry)

    def log_udp_stats(self, phase: str, epoch: int, stats: dict):
        """Log UDP/FEC communication statistics.

        Args:
            phase: Training phase
            epoch: Epoch number
            stats: Dictionary containing UDP stats (survival_rate, sent_models, etc.)
        """
        relative_time = time.time() - self.start_time
        entry = {
            "timestamp": relative_time,
            "phase": phase,
            "epoch": epoch,
            "survival_rate": stats.get("survival_rate", ""),
            "sent_models": stats.get("sent_models", ""),
            "sent_failed": stats.get("sent_failed", ""),
            "received_models": stats.get("received_models", ""),
            "fec_recovery_success": stats.get("fec_recovery_success", ""),
            "fec_recovery_fail": stats.get("fec_recovery_fail", ""),
            "bytes_sent": stats.get("bytes_sent", ""),
            "bytes_received": stats.get("bytes_received", ""),
        }
        self._write_row(entry)

    def log_compression_stats(
        self,
        phase: str,
        epoch: int,
        method: str,
        original_size: int,
        compressed_size: int,
        compression_time_ms: float,
    ):
        """Log compression statistics.

        Args:
            phase: Training phase
            epoch: Epoch number
            method: Compression method used (none, lz4, zlib, adaptive)
            original_size: Original data size in bytes
            compressed_size: Compressed data size in bytes
            compression_time_ms: Time spent compressing in milliseconds
        """
        relative_time = time.time() - self.start_time
        compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
        entry = {
            "timestamp": relative_time,
            "phase": phase,
            "epoch": epoch,
            "compression_method": method,
            "compression_ratio": compression_ratio,
            "compression_time_ms": compression_time_ms,
            "original_size": original_size,
            "compressed_size": compressed_size,
        }
        self._write_row(entry)

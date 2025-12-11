"""Metrics logging utilities."""

import csv
import os
import time


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
        "bytes_sent",
        "bytes_received",
        # Compression metrics
        "compression_method",
        "compression_ratio",
        "compression_time_ms",
        "original_size",
        "compressed_size",
    ]

    def __init__(self, experiment_id: str, node_name: str, start_timestamp: float = None):
        """Initialize metrics logger.

        Args:
            experiment_id: The experiment ID
            node_name: Node name
            start_timestamp: Experiment start timestamp (epoch)
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

    def _init_csv(self):
        """Initialize CSV file with all possible columns."""
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.ALL_COLUMNS)
                writer.writeheader()
        self.header_written = True

    def _write_row(self, entry: dict):
        """Write a row to CSV, filling missing columns with empty values."""
        # Ensure all columns are present
        full_entry = {col: entry.get(col, "") for col in self.ALL_COLUMNS}
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.ALL_COLUMNS)
            writer.writerow(full_entry)

    def log_epoch(self, phase: str, epoch: int, metrics: dict):
        """Log all metrics for an epoch to CSV.

        Args:
            phase: Training phase (SELF or WAFL)
            epoch: Epoch number
            metrics: Dictionary of metrics (e.g., {'train_loss': 0.5, 'train_accuracy': 0.8})
        """
        # Calculate relative time from experiment start
        relative_time = time.time() - self.start_time

        entry = {
            "timestamp": relative_time,
            "phase": phase,
            "epoch": epoch,
        }
        entry.update(metrics)
        self._write_row(entry)

    def log_ssp_metrics(
        self,
        phase: str,
        epoch: int,
        wasted_ms: float,
        wasted_norm: float,
        batches_processed: int,
        was_force_stopped: bool = True,
    ):
        """Log SSP (Semi-Synchronous Protocol) metrics when a node is force-stopped.

        Args:
            phase: Training phase (SELF or WAFL)
            epoch: Epoch number
            wasted_ms: Wasted computation time in milliseconds
            wasted_norm: Wasted gradient norm
            batches_processed: Number of batches processed before force stop
            was_force_stopped: Whether this epoch was force-stopped
        """
        relative_time = time.time() - self.start_time
        entry = {
            "timestamp": relative_time,
            "phase": phase,
            "epoch": epoch,
            "wasted_ms": wasted_ms,
            "wasted_norm": wasted_norm,
            "batches_processed": batches_processed,
            "was_force_stopped": 1 if was_force_stopped else 0,
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

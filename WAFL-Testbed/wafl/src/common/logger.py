"""Metrics logging utilities."""

import csv
import os
import time


class MetricsLogger:
    """Simplified metrics logger that writes to CSV."""

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

        # Write header if not written
        if not self.header_written:
            if os.path.exists(self.csv_path):
                # Check if file is empty or has content
                if os.path.getsize(self.csv_path) > 0:
                    self.header_written = True

            if not self.header_written:
                with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(entry.keys()))
                    writer.writeheader()
                self.header_written = True

        # Write row
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(entry.keys()))
            writer.writerow(entry)

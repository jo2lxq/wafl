import argparse
import csv
import os
import time

import psutil


def monitor(output_file, interval=1.0):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Initialize CSV
    file_exists = os.path.isfile(output_file)
    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "timestamp",
                    "cpu_percent",
                    "memory_percent",
                    "net_sent_bytes",
                    "net_recv_bytes",
                ]
            )

    print(f"Monitoring resources to {output_file} (interval={interval}s)...")

    net_io_start = psutil.net_io_counters()
    last_sent = net_io_start.bytes_sent
    last_recv = net_io_start.bytes_recv

    # Get current process for container-specific CPU measurement
    current_process = psutil.Process()
    # Initial call to cpu_percent to start measurement
    current_process.cpu_percent(interval=None)

    try:
        while True:
            time.sleep(interval)

            timestamp = time.time()
            # Use process-level CPU to measure only this container's workload
            cpu = current_process.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent

            net_io = psutil.net_io_counters()
            sent = net_io.bytes_sent
            recv = net_io.bytes_recv

            # Calculate delta
            delta_sent = sent - last_sent
            delta_recv = recv - last_recv

            last_sent = sent
            last_recv = recv

            with open(output_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, cpu, mem, delta_sent, delta_recv])

    except KeyboardInterrupt:
        print("Monitoring stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--interval", type=float, default=1.0, help="Monitoring interval in seconds")
    args = parser.parse_args()

    monitor(args.output, args.interval)

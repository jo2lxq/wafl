import argparse
import json
import multiprocessing
import os
import re
import subprocess
import sys
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import paramiko
import seaborn as sns

# Use non-interactive backend for parallel processing
matplotlib.use("Agg")

# Japanese font configuration for matplotlib
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

# Global seaborn theme for consistent styling across all plots
sns.set_theme()

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CTRL_DIR = PROJECT_ROOT / "ctrl"
CONFIG_FILE = CTRL_DIR / "execution_config.json"
RESULTS_DIR = PROJECT_ROOT / "results"

# Target accuracy for Time-to-Accuracy plot
TARGET_ACCURACY = 0.8

# =============================================================================
# Color Palette - Balanced contrast, readable and pleasant colors
# =============================================================================
# Primary palette for multi-node plots (moderate saturation, good distinction)
NODE_PALETTE = [
    "#5b9bd5",  # Soft Blue
    "#ed7d31",  # Soft Orange
    "#70ad47",  # Soft Green
    "#9e7cc3",  # Soft Purple
    "#ffc000",  # Soft Gold
    "#44a5a1",  # Soft Teal
    "#c45b9a",  # Soft Pink
    "#4ecdc4",  # Soft Cyan
    "#95c471",  # Light Green
    "#e89b5c",  # Light Orange
    "#8e7cc3",  # Light Purple
    "#a5826d",  # Light Brown
    "#8097a4",  # Blue Grey
    "#d4a539",  # Mustard
    "#5ba89f",  # Teal
    "#e67c73",  # Coral
    "#6aa84f",  # Green
    "#6fa8dc",  # Light Blue
    "#a64d79",  # Mauve
    "#b4c74a",  # Lime
    "#d9a128",  # Amber
    "#4a86c7",  # Medium Blue
    "#cc6666",  # Soft Red
    "#76a398",  # Sage
    "#9373b0",  # Medium Purple
    "#d98757",  # Terracotta
    "#6b7db3",  # Soft Indigo
    "#5aA99A",  # Medium Teal
]

# Single-purpose colors (balanced contrast, readable)
COLORS = {
    "mean_line": "#2c3e50",  # Dark Slate - clearly visible but not harsh
    "phase_line": "#c0392b",  # Muted Red - for phase switch
    "target_line": "#27ae60",  # Soft Green - for target accuracy
    "accuracy_fill": "#5b9bd5",  # Soft Blue - for accuracy charts
    "loss_fill": "#e67c73",  # Soft Coral - for loss charts
    "bar_primary": "#6b7db3",  # Soft Indigo - primary bar color
    "bar_secondary": "#e89b5c",  # Soft Orange - secondary bar
    "bar_tertiary": "#5aA99A",  # Medium Teal - tertiary bar
    "goodput": "#70ad47",  # Soft Green - for goodput
    "traffic": "#44a5a1",  # Soft Teal - for traffic
    "idle": "#9e9ec8",  # Soft Lavender - for idle time
    "wasted_bar": "#e89b5c",  # Soft Orange - wasted computation
    "wasted_line": "#2c3e50",  # Dark Slate - batches line
    "control_server": "#E63946",  # Vivid Red - for control server
    "no_data": "#7f8c8d",  # Cool Gray - for "no data" messages (avoid pure gray)
}


def load_config():
    if not CONFIG_FILE.exists():
        print(f"❌ Error: Configuration file not found: {CONFIG_FILE}")
        sys.exit(1)
    with open(CONFIG_FILE) as f:
        return json.load(f)


def _run_plot_task(args):
    """Worker function for multiprocessing."""
    func, func_args = args
    if func is None:
        return None
    try:
        # Ensure Agg backend in worker process
        matplotlib.use("Agg")
        return func(*func_args)
    except Exception as e:
        # Print a concise error message
        print(f"  ❌ Error running {func.__name__ if hasattr(func, '__name__') else 'plot task'}: {e}")
        return None


def _add_phase_line(ax, data, x_col="epoch", text_position="top"):
    """Add a vertical line indicating the start of the WAFL phase."""
    # Filter out SSP interrupted rows for accurate phase detection
    clean_data = data.copy()
    if "is_ssp_interrupted" in clean_data.columns:
        clean_data = clean_data[~clean_data["is_ssp_interrupted"]]

    wafl_data = clean_data[clean_data["phase"] == "WAFL"]
    if wafl_data.empty:
        return

    wafl_start = wafl_data[x_col].min()
    if pd.isna(wafl_start):
        return

    # Offset by 0.5 to draw between epochs (e.g. between 2 and 3)
    if x_col == "epoch":
        line_pos = wafl_start - 0.5
    else:
        line_pos = wafl_start

    ax.axvline(
        x=line_pos,
        color=COLORS["phase_line"],
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
    )
    y_min, y_max = ax.get_ylim()
    if text_position == "top":
        y_pos = y_max * 0.95
        va = "top"
    else:
        y_pos = y_min + (y_max - y_min) * 0.05
        va = "bottom"
    ax.text(
        line_pos,
        y_pos,
        " WAFL Start",
        color=COLORS["phase_line"],
        va=va,
        fontsize=10,
        fontweight="bold",
    )


def get_latest_experiment_id():
    """Find the latest experiment directory in results/ based on timestamp in name."""
    if not RESULTS_DIR.exists():
        return None
    # List all directories in results/ excluding hidden ones like .deploy and .comparison
    dirs = [d for d in RESULTS_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not dirs:
        return None

    # Helper to extract timestamp from folder name (e.g., normal-20251127T175937)
    def extract_timestamp(d):
        try:
            return d.name.split("-")[-1]
        except IndexError:
            return ""

    # Return the one with the latest timestamp string (lexicographical sort works for ISO-like format)
    return max(dirs, key=extract_timestamp).name


def get_experiments_without_analysis():
    """Find all experiment directories that don't have an 'analysis' subdirectory."""
    if not RESULTS_DIR.exists():
        return []

    experiments = []
    for d in RESULTS_DIR.iterdir():
        if d.is_dir() and not d.name.startswith("."):
            analysis_dir = d / "analysis"
            if not analysis_dir.exists():
                experiments.append(d.name)

    # Sort by timestamp (newest first)
    def extract_timestamp(name):
        try:
            return name.split("-")[-1]
        except IndexError:
            return ""

    experiments.sort(key=extract_timestamp, reverse=True)
    return experiments


def get_all_experiments():
    """Get all experiment directories."""
    if not RESULTS_DIR.exists():
        return []

    experiments = []
    for d in RESULTS_DIR.iterdir():
        if d.is_dir() and not d.name.startswith("."):
            experiments.append(d.name)

    # Sort by timestamp (newest first)
    def extract_timestamp(name):
        try:
            return name.split("-")[-1]
        except IndexError:
            return ""

    experiments.sort(key=extract_timestamp, reverse=True)
    return experiments


def collect_results(experiment_id, config):
    """Collect results from all nodes for the given experiment ID."""
    print(f"📥 Collecting results for experiment: {experiment_id}")

    local_exp_dir = RESULTS_DIR / experiment_id
    local_exp_dir.mkdir(parents=True, exist_ok=True)

    nodes = config.get("nodes", [])
    user = config.get("user", "denjo")
    deployment_location = config.get("deployment_location", "/home/denjo")
    key_path = os.path.expanduser("~/.ssh/id_ed25519")

    def collect_node(node):
        node_name = str(node["name"])
        ip = node["physical_ip"]

        # Remote path: <deploy_loc>/WAFL-Testbed/results/<exp_id>
        remote_path = f"{deployment_location}/WAFL-Testbed/results/{experiment_id}"
        local_node_dir = local_exp_dir / node_name
        local_node_dir.mkdir(exist_ok=True)

        files_to_collect = ["metrics.csv", "resources.csv", "model.pth", "output.log"]
        collected = []

        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(ip, username=user, key_filename=key_path, timeout=10)
            sftp = ssh.open_sftp()

            for fname in files_to_collect:
                r_file = f"{remote_path}/{fname}"
                l_file = local_node_dir / fname
                try:
                    sftp.get(r_file, str(l_file))
                    collected.append(fname)
                except FileNotFoundError:
                    pass
                except Exception as e:
                    print(f"  ⚠️  {node_name}: Failed to collect {fname}: {e}")

            sftp.close()
            ssh.close()
            return True, node_name, collected
        except Exception as e:
            return False, node_name, str(e)

    success_count = 0
    with ThreadPoolExecutor(max_workers=len(nodes)) as executor:
        futures = {executor.submit(collect_node, node): node for node in nodes}
        for future in as_completed(futures):
            success, name, res = future.result()
            if success:
                print(f"  ✅ Node {name}: Collected {len(res)} files")
                success_count += 1
            else:
                print(f"  ❌ Node {name}: Connection failed - {res}")

    print(f"🏁 Collection complete: {success_count}/{len(nodes)} nodes")
    return success_count > 0


def collect_results_via_rsync(experiment_id, config, remote_host=None):
    """Collect results from management server via rsync.

    Args:
        experiment_id: Experiment ID to collect
        config: Configuration dictionary
        remote_host: Remote host address (default: uses config)
    """
    print(f"📥 Collecting results via rsync for experiment: {experiment_id}")

    # Get remote host from config if not specified
    if remote_host is None:
        # Try to get from management_server or use first node's IP
        remote_host = config.get("management_server")
        if not remote_host:
            print("❌ No remote host specified and no management_server in config")
            return False

    user = config.get("user", "denjo")
    deployment_location = config.get("deployment_location", "/home/denjo")
    key_path = os.path.expanduser("~/.ssh/id_ed25519")

    # Remote path
    remote_path = f"{user}@{remote_host}:{deployment_location}/WAFL-Testbed/results/{experiment_id}/"
    local_path = RESULTS_DIR / experiment_id
    local_path.mkdir(parents=True, exist_ok=True)

    # rsync command with SSH key
    rsync_cmd = [
        "rsync",
        "-avz",
        "--progress",
        "-e",
        f"ssh -i {key_path} -o StrictHostKeyChecking=no",
        remote_path,
        str(local_path) + "/",
    ]

    print(f"  🔄 Running: rsync from {remote_host}...")
    try:
        result = subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("  ✅ rsync completed successfully")
            # Count collected files
            files = list(local_path.rglob("*"))
            file_count = sum(1 for f in files if f.is_file())
            print(f"  📁 {file_count} files collected to: {local_path}")
            return True
        else:
            print(f"  ❌ rsync failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("  ❌ rsync timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"  ❌ rsync error: {e}")
        return False


def collect_all_experiments_via_rsync(config, remote_host=None):
    """Collect all experiment results from management server via rsync."""
    print("📥 Collecting all experiments via rsync...")

    if remote_host is None:
        remote_host = config.get("management_server")
        if not remote_host:
            print("❌ No remote host specified and no management_server in config")
            return False

    user = config.get("user", "denjo")
    deployment_location = config.get("deployment_location", "/home/denjo")
    key_path = os.path.expanduser("~/.ssh/id_ed25519")

    # rsync entire results directory (excluding hidden files)
    remote_path = f"{user}@{remote_host}:{deployment_location}/WAFL-Testbed/results/"
    local_path = RESULTS_DIR

    rsync_cmd = [
        "rsync",
        "-avz",
        "--progress",
        "--exclude",
        ".*",
        "-e",
        f"ssh -i {key_path} -o StrictHostKeyChecking=no",
        remote_path,
        str(local_path) + "/",
    ]

    print(f"  🔄 Running: rsync all experiments from {remote_host}...")
    try:
        result = subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print("  ✅ rsync completed successfully")
            return True
        else:
            print(f"  ❌ rsync failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("  ❌ rsync timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"  ❌ rsync error: {e}")
        return False


def _load_metrics_and_resources(exp_dir):
    """Load metrics and resources dataframes from experiment directory."""
    # Load Metrics
    metrics_dfs = []
    for node_dir in exp_dir.iterdir():
        if node_dir.is_dir() and node_dir.name != "analysis" and node_dir.name != "ctrl":
            metrics_file = node_dir / "metrics.csv"
            if metrics_file.exists():
                try:
                    df = pd.read_csv(metrics_file)
                    # Force numeric conversion for key columns to avoid object type issues (empty strings)
                    numeric_cols = [
                        "epoch",
                        "train_loss",
                        "train_accuracy",
                        "test_loss",
                        "test_accuracy",
                        "epoch_duration_ms",
                        "compute_time_ms",
                        "comm_time_ms",
                        "waiting_time_ms",
                        "wasted_ms",
                        "wasted_norm",
                        "batches_processed",
                        "was_force_stopped",
                        "survival_rate",
                        "sent_models",
                        "sent_failed",
                        "received_models",
                        "fec_recovery_success",
                        "fec_recovery_fail",
                        "fec_encode_time_ms",
                        "fec_decode_time_ms",
                        "bytes_sent",
                        "bytes_received",
                        "timeout_models",
                        "rudp_nacks_sent",
                        "fec_recoveries",
                        "rudp_retransmissions",
                        "rudp_acks_sent",
                        "rudp_acks_received",
                        "rudp_eaks_sent",
                        "rudp_eaks_received",
                        "rudp_aged_packets",
                        "rudp_connect_time_ms",
                        "rudp_avg_rtt_ms",
                        "rudp_max_retries_reached",
                        # New Metrics
                        "app_bytes_sent",
                        "app_bytes_received",
                        "protocol_tcp_count",
                        "protocol_udp_count",
                        "protocol_rudp_count",
                        "udp_avg_parity",
                        "udp_avg_pacing_ms",
                        "compression_ratio",
                        "compression_time_ms",
                        "original_size",
                        "compressed_size",
                    ]
                    for col in numeric_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors="coerce")

                    df["node"] = node_dir.name
                    metrics_dfs.append(df)
                except Exception:
                    pass

    df = pd.concat(metrics_dfs) if metrics_dfs else pd.DataFrame()

    # Handle SSP FORCE_NEXT rows - these have was_force_stopped=1 and may create duplicate epochs
    # Keep SSP rows for wasted computation analysis, but for other plots, prioritize completed epochs
    if not df.empty and "was_force_stopped" in df.columns:
        # Create a marker for SSP-interrupted epochs (these have wasted_ms > 0 or was_force_stopped = 1)
        df["is_ssp_interrupted"] = df["was_force_stopped"].fillna(0).astype(int) == 1

        # For epoch-based plots, we need to handle duplicates
        # Sort by timestamp to get proper ordering, then for duplicates, keep the complete epoch
        df = df.sort_values(["node", "timestamp"]).reset_index(drop=True)

    # Load Resources
    resources_dfs = []
    for node_dir in exp_dir.iterdir():
        if node_dir.is_dir() and node_dir.name != "analysis" and node_dir.name != "ctrl":
            res_file = node_dir / "resources.csv"
            if res_file.exists():
                try:
                    rdf = pd.read_csv(res_file)
                    rdf["node"] = node_dir.name
                    resources_dfs.append(rdf)
                except Exception:
                    pass

    resources_df = pd.concat(resources_dfs) if resources_dfs else pd.DataFrame()

    return df, resources_df


# =============================================================================
# Individual Plot Generation Functions (for parallel execution)
# =============================================================================


def _generate_traffic_overhead_plot(df, experiment_name, analysis_dir):
    """Generate Traffic Overhead Ratio plot (Physical Bytes / App Bytes)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    has_meaningful_data = False

    if not df.empty and "bytes_sent" in df.columns and "app_bytes_sent" in df.columns:
        wafl_df = df[df["phase"] == "WAFL"].copy()
        valid = wafl_df[(wafl_df["bytes_sent"].notna()) & (wafl_df["app_bytes_sent"].notna())].copy()

        if not valid.empty and valid["bytes_sent"].sum() > 0:
            has_meaningful_data = True

            # Calculate ratio per epoch
            agg = valid.groupby("epoch").agg({"bytes_sent": "sum", "app_bytes_sent": "sum"}).reset_index()
            # Avoid division by zero
            agg["overhead_ratio"] = agg["bytes_sent"] / agg["app_bytes_sent"].replace(0, np.nan)

            # Plot Ratio
            sns.lineplot(data=agg, x="epoch", y="overhead_ratio", color=COLORS["phase_line"], linewidth=2.5, marker="o", label="Overhead Ratio (Physical/App)", ax=ax)

            # Add reference line at 1.0 (No Overhead)
            ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Ideal (1.0)")

            # Add mean ratio to title
            mean_ratio = agg["overhead_ratio"].mean()
            ax.set_title(f"Traffic Overhead Ratio - {experiment_name}\n(Mean Ratio: {mean_ratio:.2f}x)")
            ax.set_ylabel("Overhead Ratio (Physical / App)")
            ax.set_xlabel("Epoch")
            ax.legend()
            ax.grid(True, alpha=0.3)

    if not has_meaningful_data:
        ax.text(0.5, 0.5, "Traffic Data N/A", ha="center", va="center", color=COLORS["no_data"], transform=ax.transAxes)
        ax.axis("off")
        ax.set_title(f"Traffic Overhead Ratio - {experiment_name}")

    fig.tight_layout()
    fig.savefig(analysis_dir / "traffic_overhead.png", dpi=150)
    plt.close(fig)
    return "traffic_overhead.png"


def _generate_throughput_overhead_plot(df, experiment_name, analysis_dir):
    """Generate Throughput Overhead plot (Physical Throughput vs Goodput)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    has_meaningful_data = False

    cols = ["bytes_sent", "app_bytes_sent", "comm_time_ms"]
    if not df.empty and all(c in df.columns for c in cols):
        wafl_df = df[df["phase"] == "WAFL"].copy()
        valid = wafl_df[(wafl_df["comm_time_ms"] > 50)].copy()  # Min 50ms filter

        if not valid.empty:
            has_meaningful_data = True

            # Calculate stats per epoch
            # (agg variable calculation removed as it was unused)

            # Convert to Mbps
            # Comm time is average per node, bytes are SUM.
            # To get accurate throughput comparison, we should calculate per node then average, OR
            # if we sum bytes, we should sum comm_time? No.
            # Let's use the average throughput per node approach for consistency.

            # Option 2: Calculate per-node throughput first
            valid["phy_mbps"] = (valid["bytes_sent"] * 8 / 1e6) / (valid["comm_time_ms"] / 1000)
            valid["app_mbps"] = (valid["app_bytes_sent"] * 8 / 1e6) / (valid["comm_time_ms"] / 1000)

            node_agg = valid.groupby("epoch").agg({"phy_mbps": "mean", "app_mbps": "mean"}).reset_index()

            sns.lineplot(data=node_agg, x="epoch", y="phy_mbps", label="Physical Throughput (Bandwidth Used)", color=COLORS["traffic"], linewidth=2, ax=ax)
            sns.lineplot(data=node_agg, x="epoch", y="app_mbps", label="Goodput (Data Delivered)", color=COLORS["goodput"], linewidth=2, ax=ax)

            ax.fill_between(node_agg["epoch"], node_agg["app_mbps"], node_agg["phy_mbps"], color=COLORS["traffic"], alpha=0.1, label="Overhead Gap")

            ax.set_title(f"Throughput Overhead (Physical vs Goodput) - {experiment_name}")
            ax.set_ylabel("Throughput [Mbps]")
            ax.set_xlabel("Epoch")
            ax.legend()
            ax.grid(True, alpha=0.3)

    if not has_meaningful_data:
        ax.text(0.5, 0.5, "Throughput Data N/A", ha="center", va="center", color=COLORS["no_data"], transform=ax.transAxes)
        ax.axis("off")
        ax.set_title(f"Throughput Overhead - {experiment_name}")

    fig.tight_layout()
    fig.savefig(analysis_dir / "throughput_overhead.png", dpi=150)
    plt.close(fig)
    return "throughput_overhead.png"


def _generate_accuracy_time_plot(df, experiment_name, analysis_dir, add_phase_line_func):
    """Generate Accuracy vs Wall-clock Time plot (from WAFL Start)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    if not df.empty and "test_accuracy" in df.columns and "wafl_relative_timestamp" in df.columns:
        # Use WAFL phase only for clean time-series from 0
        wafl_df = df[df["phase"] == "WAFL"].copy()
        if "is_ssp_interrupted" in wafl_df.columns:
            wafl_df = wafl_df[~wafl_df["is_ssp_interrupted"]]

        if not wafl_df.empty:
            # Stats per epoch to get mean time and accuracy
            agg = wafl_df.groupby("epoch").agg({"wafl_relative_timestamp": "mean", "test_accuracy": ["mean", "std"]})
            agg.columns = ["time", "mean", "std"]
            agg = agg.reset_index()
            agg["std"] = agg["std"].fillna(0)

            ax.plot(agg["time"], agg["mean"], color=COLORS["mean_line"], linewidth=2.5, label="Test Accuracy", marker="o", markersize=3)
            ax.fill_between(agg["time"], agg["mean"] - agg["std"], agg["mean"] + agg["std"], color=COLORS["accuracy_fill"], alpha=0.3)

            # Target line
            ax.axhline(y=TARGET_ACCURACY, color=COLORS["target_line"], linestyle="--", label=f"Target ({TARGET_ACCURACY:.0%})")

            ax.set_title(f"Test Accuracy vs Wall-clock Time - {experiment_name}")
            ax.set_ylabel("Test Accuracy")
            ax.set_xlabel("Time from WAFL Start [sec]")
            ax.set_ylim(0, 1.05)
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)

            fig.tight_layout()
            fig.savefig(analysis_dir / "accuracy_time.png", dpi=150)
            plt.close(fig)
            return "accuracy_time.png"

    # No data
    plt.close(fig)
    return None


def _generate_asymmetry_plot(df, experiment_name, analysis_dir, add_phase_line_func):
    """Generate model exchange asymmetry plot (Received Models Distribution)."""
    if df.empty or "received_models" not in df.columns:
        return None

    wafl_df = df[df["phase"] == "WAFL"].copy()
    if wafl_df.empty:
        return None

    # Filter out SSP interrupted rows
    if "is_ssp_interrupted" in wafl_df.columns:
        wafl_df = wafl_df[~wafl_df["is_ssp_interrupted"]]

    if wafl_df.empty:
        return None

    plt.figure(figsize=(12, 6))

    # User Request: "Make symmetry easier to verify"
    # Total Received Models per Node (Bar Chart)
    # This clearly shows if some nodes are starved.
    total_received = wafl_df.groupby("node")["received_models"].sum().reset_index()

    # Convert node names to int for proper sorting if possible
    try:
        total_received["node_int"] = total_received["node"].astype(int)
        total_received = total_received.sort_values("node_int")
    except ValueError:
        # Fallback to string sort
        total_received = total_received.sort_values("node")

    sns.barplot(data=total_received, x="node", y="received_models", palette=NODE_PALETTE[: wafl_df["node"].nunique()] if wafl_df["node"].nunique() <= len(NODE_PALETTE) else "husl", hue="node", dodge=False, legend=False, edgecolor="black", linewidth=0.5)

    plt.title(f"Asymmetry Check (Total Received Models per Node) - {experiment_name}")
    plt.xlabel("Node")
    plt.ylabel("Total Received Models")
    plt.tight_layout()
    plt.savefig(analysis_dir / "asymmetry_distribution.png", dpi=150)
    plt.close()
    plt.close()

    # User Request: "Case B: Lorenz Curve" (Gini Coefficient)
    # Sort data by value
    sorted_data = np.sort(total_received["received_models"].values)

    # Cumulative sum of received models
    cumulative_received = np.cumsum(sorted_data)
    total_sum = cumulative_received[-1]

    if total_sum > 0:
        # Normalize
        lorenz_y = np.insert(cumulative_received / total_sum, 0, 0)
        lorenz_x = np.linspace(0, 1, len(lorenz_y))

        # Ideal Equality Line (y=x)
        ideal_x = [0, 1]
        ideal_y = [0, 1]

        # Gini Coefficient Calculation
        # G = 1 - 2 * (Area under Lorenz Curve)
        # Area under Lorenz approx trapezoidal rule
        area_lorenz = np.trapezoid(lorenz_y, lorenz_x)
        gini = 1 - 2 * area_lorenz

        plt.figure(figsize=(8, 8))
        plt.plot(lorenz_x, lorenz_y, label=f"Observed (Gini={gini:.3f})", color=COLORS["wasted_bar"], linewidth=2.5)
        plt.plot(ideal_x, ideal_y, label="Perfect Equality", color="gray", linestyle="--", alpha=0.7)
        plt.fill_between(lorenz_x, lorenz_y, lorenz_x, color=COLORS["wasted_bar"], alpha=0.1)

        plt.title(f"Asymmetry Check (Lorenz Curve) - {experiment_name}")
        plt.xlabel("Cumulative Share of Nodes (Ranked by Reception)")
        plt.ylabel("Cumulative Share of Received Models")
        plt.legend()
        plt.tight_layout()
        plt.savefig(analysis_dir / "asymmetry_lorenz.png", dpi=150)
        plt.close()

    return "asymmetry_distribution.png"


def _generate_survivor_quality_plot(df, experiment_name, analysis_dir, add_phase_line_func):
    """Generate Survivor Quality plot (Test Accuracy vs Received Models)."""
    if df.empty or "received_models" not in df.columns or "test_accuracy" not in df.columns:
        return None

    wafl_df = df[df["phase"] == "WAFL"].copy()
    if wafl_df.empty:
        return None

    # Filter out SSP interrupted rows
    if "is_ssp_interrupted" in wafl_df.columns:
        wafl_df = wafl_df[~wafl_df["is_ssp_interrupted"]]

    if wafl_df.empty:
        return None

    # 1. Survivor Quality Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Box Plot of Accuracy vs Received Models Count
    sns.boxplot(
        data=wafl_df,
        x="received_models",
        y="test_accuracy",
        color=COLORS["accuracy_fill"],
        width=0.6,
        showfliers=False,
        ax=ax,
    )

    means = wafl_df.groupby("received_models")["test_accuracy"].mean().reset_index()
    sns.lineplot(
        data=means,
        x="received_models",
        y="test_accuracy",
        color=COLORS["mean_line"],
        linewidth=2,
        marker="o",
        label="Mean Accuracy",
        ax=ax,
    )

    ax.set_title(f"Survivor Quality (Accuracy vs Connectivity) - {experiment_name}")
    ax.set_xlabel("Received Models (Count per Epoch)")
    ax.set_ylabel("Test Accuracy distributions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(analysis_dir / "survivor_quality.png", dpi=150)
    plt.close(fig)

    # 2. Survivor Trajectory Groups Plot
    # Classify nodes based on TOTAL received models
    node_totals = wafl_df.groupby("node")["received_models"].sum().reset_index()
    median_val = node_totals["received_models"].median()

    # Handle edge case where all are same (median == min == max)
    if node_totals["received_models"].nunique() > 1:
        high_nodes = set(node_totals[node_totals["received_models"] > median_val]["node"].unique())

        # Vectorized assignment for speed
        wafl_df["connectivity_group"] = "Low (<= Median)"
        wafl_df.loc[wafl_df["node"].isin(high_nodes), "connectivity_group"] = "High (> Median)"

        # Plot trajectory (Accuracy vs Epoch) per group
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        sns.lineplot(
            data=wafl_df,
            x="epoch",
            y="test_accuracy",
            hue="connectivity_group",
            style="connectivity_group",
            linewidth=2.5,
            markers=True,
            dashes=False,
            palette={"High (> Median)": COLORS["goodput"], "Low (<= Median)": COLORS["traffic"]},
            ax=ax2,
        )

        ax2.set_title(f"Survivor Trajectory (High vs Low Connectivity) - {experiment_name}")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Test Accuracy (Group Mean)")
        ax2.legend(title="Connectivity Group")
        fig2.tight_layout()
        fig2.savefig(analysis_dir / "survivor_trajectory_groups.png", dpi=150)
        plt.close(fig2)

    return "survivor_quality.png"


def _generate_accuracy_plot(df, experiment_name, analysis_dir, add_phase_line_func):
    """Generate test accuracy plot."""
    if df.empty or "test_accuracy" not in df.columns:
        return None

    # Filter out SSP interrupted rows (they don't have valid test_accuracy) and NaN values
    plot_df = df[df["test_accuracy"].notna()].copy()
    if "is_ssp_interrupted" in plot_df.columns:
        plot_df = plot_df[~plot_df["is_ssp_interrupted"]]
    if plot_df.empty:
        return None

    plt.figure(figsize=(12, 6))
    num_nodes = plot_df["node"].nunique()
    palette = NODE_PALETTE[:num_nodes] if num_nodes <= len(NODE_PALETTE) else "husl"

    ax = sns.lineplot(
        data=plot_df,
        x="epoch",
        y="test_accuracy",
        hue="node",
        alpha=0.4,
        legend=False,
        palette=palette,
        estimator=None,
        linewidth=1.2,
    )
    sns.lineplot(
        data=plot_df,
        x="epoch",
        y="test_accuracy",
        color=COLORS["mean_line"],
        linewidth=2.5,
        label="Mean",
        errorbar=None,
        ax=ax,
    )
    add_phase_line_func(ax, plot_df, "epoch", text_position="bottom")
    plt.title(f"Test Accuracy - {experiment_name}")
    plt.ylabel("Test Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(analysis_dir / "accuracy.png", dpi=150)
    plt.close()
    return "accuracy.png"


def _generate_epoch_duration_plot(df, experiment_name, analysis_dir, add_phase_line_func, exp_dir=None):
    """Generate wall-clock time per epoch plot with control server time overlay."""
    if df.empty or "epoch_duration_ms" not in df.columns:
        return None

    epoch_dur = df[df["epoch_duration_ms"].notna()].copy()
    if epoch_dur.empty:
        return None

    epoch_dur["epoch_duration_s"] = epoch_dur["epoch_duration_ms"] / 1000
    num_nodes = epoch_dur["node"].nunique()
    palette = NODE_PALETTE[:num_nodes] if num_nodes <= len(NODE_PALETTE) else "husl"

    fig, ax = plt.subplots(figsize=(12, 6))
    # Plot individual node times with light lines
    sns.lineplot(
        data=epoch_dur,
        x="epoch",
        y="epoch_duration_s",
        hue="node",
        alpha=0.4,
        legend=False,
        palette=palette,
        estimator=None,
        linewidth=1.2,
        ax=ax,
    )

    # Plot mean of node times
    sns.lineplot(
        data=epoch_dur,
        x="epoch",
        y="epoch_duration_s",
        color=COLORS["mean_line"],
        linewidth=2.0,
        label="Node Mean",
        errorbar=None,
        ax=ax,
        alpha=0.7,
    )

    # Load and plot control server epoch durations if available
    if exp_dir is not None:
        # Load metadata.json (unified format)
        metadata_path = exp_dir / "metadata.json"
        if not metadata_path.exists():
            metadata_path = exp_dir / "ctrl" / "metadata.json"  # Try ctrl subdirectory
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    ctrl_metadata = json.load(f)
                ctrl_durations = ctrl_metadata.get("epoch_durations", [])
                if ctrl_durations:
                    ctrl_df = pd.DataFrame(ctrl_durations)

                    # Remove duplicates if any (keep last occurrence)
                    ctrl_df = ctrl_df.drop_duplicates(subset=["epoch"], keep="last")

                    ctrl_df["duration_s"] = ctrl_df["duration_ms"] / 1000

                    # Sort by epoch for proper line plotting
                    ctrl_df = ctrl_df.sort_values("epoch")

                    # Plot control server times with prominent colored line
                    # Epoch numbers are now global (continuous across SELF and WAFL phases)
                    ax.plot(
                        ctrl_df["epoch"],
                        ctrl_df["duration_s"],
                        color=COLORS["control_server"],  # Vivid red for visibility
                        linewidth=3.0,
                        label="Control Server",
                        marker="o",
                        markersize=4,
                        zorder=10,  # Ensure it's on top
                    )
            except Exception as e:
                print(f"   ⚠️ Could not load control server epoch durations: {e}")

    add_phase_line_func(ax, df, "epoch")
    ax.set_title(f"Wall-clock Time per Epoch - {experiment_name}")
    ax.set_ylabel("Duration [sec]")
    ax.set_xlabel("Epoch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(analysis_dir / "epoch_duration.png", dpi=150)
    plt.close(fig)
    return "epoch_duration.png"


def _generate_idle_time_plot(df, experiment_name, analysis_dir, add_phase_line_func):
    """Generate idle time ratio plot."""
    if df.empty or "epoch_duration_ms" not in df.columns:
        return None

    epoch_dur = df[df["epoch_duration_ms"].notna()].copy()
    if epoch_dur.empty:
        return None

    max_dur_per_epoch = epoch_dur.groupby("epoch")["epoch_duration_ms"].max().reset_index()
    max_dur_per_epoch.columns = ["epoch", "max_duration_ms"]
    epoch_dur = epoch_dur.merge(max_dur_per_epoch, on="epoch")
    epoch_dur["idle_time_ms"] = epoch_dur["max_duration_ms"] - epoch_dur["epoch_duration_ms"]
    epoch_dur["idle_time_ratio"] = epoch_dur["idle_time_ms"] / epoch_dur["max_duration_ms"]
    idle_agg = epoch_dur.groupby("epoch").agg({"idle_time_ratio": "mean", "idle_time_ms": "sum"}).reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create a percentage column for seaborn
    idle_agg["idle_time_pct"] = idle_agg["idle_time_ratio"] * 100
    # Use matplotlib bar for correct numeric x-axis alignment
    ax.bar(
        idle_agg["epoch"],
        idle_agg["idle_time_pct"],
        color=COLORS["idle"],
        alpha=0.8,
        edgecolor=COLORS["idle"],
        linewidth=0.5,
        width=0.6,
        label="Idle Time (%)",
    )

    # Add phase switch line
    add_phase_line_func(ax, df, x_col="epoch", text_position="top")

    ax.set_title(f"Idle Time Ratio (Sync Wait Time) - {experiment_name}")
    fig.tight_layout()
    fig.savefig(analysis_dir / "idle_time_ratio.png", dpi=150)
    plt.close(fig)
    return "idle_time_ratio.png"


def _generate_wasted_computation_plot(df, experiment_name, analysis_dir):
    """Generate wasted computation plot (WAFL phase only)."""
    has_wasted_data = False

    # Always try to generate if column exists, even if values are 0 (to show 0 waste)
    if not df.empty and "wasted_ms" in df.columns:
        # Filter to WAFL phase only
        wafl_df = df[df["phase"] == "WAFL"].copy()

        # Determine if we should plot.
        # If the column exists, we plot. If all 0, it just shows 0.
        if not wafl_df.empty:
            has_wasted_data = True

            # Ensure 0s are treated as 0s
            wafl_df["wasted_ms"] = wafl_df["wasted_ms"].fillna(0)

            wasted_per_epoch = wafl_df.groupby("epoch").agg({"wasted_ms": "sum", "batches_processed": "sum"}).reset_index()
            wasted_per_epoch["wasted_s"] = wasted_per_epoch["wasted_ms"] / 1000

            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax1.bar(
                wasted_per_epoch["epoch"],
                wasted_per_epoch["wasted_s"],
                color=COLORS["wasted_bar"],
                alpha=0.8,
                label="Wasted Time",
                edgecolor=COLORS["wasted_bar"],
                linewidth=0.5,
                width=0.6,
            )
            ax1.set_ylabel("Wasted Time [sec]", color=COLORS["wasted_bar"])
            ax1.tick_params(axis="y", labelcolor=COLORS["wasted_bar"])
            ax1.set_xlabel("Epoch")

            # Use ax.plot for the line to avoid Seaborn/TwinX issues
            ax2 = ax1.twinx()
            ax2.plot(
                wasted_per_epoch["epoch"],
                wasted_per_epoch["batches_processed"],
                color=COLORS["wasted_line"],
                linewidth=2.5,
                marker="o",
                markersize=4,
                label="Incomplete Batches",
            )
            ax2.set_ylabel("Incomplete Batches (before force-skip)", color=COLORS["wasted_line"])
            ax2.tick_params(axis="y", labelcolor=COLORS["wasted_line"])

            # Manually add legend for ax2 since it doesn't auto-integrate with ax1
            # Or just rely on colors/labels.
            # We can create a combined legend.
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

            ax1.set_title(f"Wasted Computation (SSP Force-Skip) - {experiment_name}")
            fig.tight_layout()
            fig.savefig(analysis_dir / "wasted_computation.png", dpi=150)
            plt.close(fig)
            return "wasted_computation.png"

    if not has_wasted_data:
        # This fallback usually only triggers if WAFL phase is empty or column missing
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(
            0.5,
            0.5,
            "No wasted computation data column found",
            ha="center",
            va="center",
            fontsize=14,
            color=COLORS["no_data"],
            transform=ax.transAxes,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        plt.title(f"Wasted Computation (SSP) - {experiment_name}")
        plt.tight_layout()
        plt.savefig(analysis_dir / "wasted_computation.png", dpi=150)
        plt.close()
        return "wasted_computation.png (no SSP data)"


def _generate_survival_rate_plot(df, experiment_name, analysis_dir, add_phase_line_func):
    """Generate survival rate plot (WAFL phase only)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    has_meaningful_survival_data = False

    if not df.empty and "survival_rate" in df.columns:
        # Filter to WAFL phase only
        wafl_df = df[df["phase"] == "WAFL"].copy()
        udp_data = wafl_df[wafl_df["survival_rate"].notna()].copy()
        udp_used = False
        if "bytes_sent" in wafl_df.columns and "bytes_received" in wafl_df.columns:
            udp_used = (wafl_df["bytes_sent"].sum() > 0) or (wafl_df["bytes_received"].sum() > 0)

        has_variation = udp_data["survival_rate"].std() > 0.001 if not udp_data.empty else False

        if not udp_data.empty and (has_variation or udp_used):
            has_meaningful_survival_data = True
            num_nodes = udp_data["node"].nunique()
            palette = NODE_PALETTE[:num_nodes] if num_nodes <= len(NODE_PALETTE) else "husl"

            sns.lineplot(
                data=udp_data,
                x="epoch",
                y="survival_rate",
                hue="node",
                alpha=0.4,
                legend=False,
                palette=palette,
                estimator=None,
                linewidth=1.2,
                ax=ax,
            )
            sns.lineplot(
                data=udp_data,
                x="epoch",
                y="survival_rate",
                color=COLORS["mean_line"],
                linewidth=2.5,
                label="Mean",
                errorbar=None,
                ax=ax,
            )
            ax.set_ylim(0, 1.05)

            if not has_variation:
                ax.set_title(f"Survival Rate (UDP/FEC) - {experiment_name}\n(100% survival - no packet loss or FEC recovery successful)")
            else:
                ax.set_title(f"Survival Rate (UDP/FEC) - {experiment_name}")
            ax.set_ylabel("Survival Rate")
            ax.set_xlabel("Epoch")
            ax.legend()

    if not has_meaningful_survival_data:
        ax.text(
            0.5,
            0.5,
            "UDP/FEC not enabled\n(survival_rate = 1.0 by default)",
            ha="center",
            va="center",
            fontsize=14,
            color=COLORS["no_data"],
            transform=ax.transAxes,
        )
        ax.axis("off")
        ax.set_title(f"Survival Rate (UDP/FEC) - {experiment_name}")

    fig.tight_layout()
    fig.savefig(analysis_dir / "survival_rate.png", dpi=150)
    plt.close(fig)
    return "survival_rate.png"


def _generate_goodput_plot(df, experiment_name, analysis_dir):
    """Generate goodput plot showing both sent and received throughput (WAFL phase only)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    has_meaningful_goodput_data = False

    if not df.empty and "bytes_received" in df.columns and "bytes_sent" in df.columns and "comm_time_ms" in df.columns:
        # Filter to WAFL phase only
        wafl_df = df[df["phase"] == "WAFL"].copy()
        goodput_data = wafl_df[
            (wafl_df["bytes_received"].notna()) & (wafl_df["bytes_sent"].notna()) & (wafl_df["comm_time_ms"].notna()) & (wafl_df["comm_time_ms"] > 0)  # Only where communication occurred
        ].copy()
        if not goodput_data.empty and (goodput_data["bytes_received"].sum() > 0 or goodput_data["bytes_sent"].sum() > 0):
            has_meaningful_goodput_data = True
            # Determine meaningful bytes column for Goodput (prefer app_bytes_received)
            if "app_bytes_received" in goodput_data.columns and goodput_data["app_bytes_received"].sum() > 0:
                rx_col = "app_bytes_received"
            else:
                rx_col = "bytes_received"

            # Calculate throughput in Mbps using comm_time_ms (pure communication time)
            goodput_data["goodput_mbps"] = (goodput_data[rx_col] * 8 / 1e6) / (goodput_data["comm_time_ms"] / 1000)

            # bytes_sent is always Physical, so sent_mbps represents "Network Bandwidth Usage"
            goodput_data["sent_mbps"] = (goodput_data["bytes_sent"] * 8 / 1e6) / (goodput_data["comm_time_ms"] / 1000)

            # Calculate retransmission estimation based on survival_rate
            # Loss rate = 1 - survival_rate
            # Retransmission factor = 1 / (1 - loss_rate) = 1 / survival_rate
            if "survival_rate" in goodput_data.columns:
                goodput_data["survival_rate"] = goodput_data["survival_rate"].fillna(1.0)
                # Clamp survival_rate to avoid division by zero (min 0.1 = max 10x retrans)
                goodput_data["survival_rate_clamped"] = goodput_data["survival_rate"].clip(lower=0.1)
                goodput_data["retrans_factor"] = 1.0 / goodput_data["survival_rate_clamped"]
                goodput_data["estimated_physical_mbps"] = goodput_data["sent_mbps"] * goodput_data["retrans_factor"]
            else:
                goodput_data["estimated_physical_mbps"] = goodput_data["sent_mbps"]
                goodput_data["retrans_factor"] = 1.0

            goodput_agg = goodput_data.groupby("epoch").agg({"goodput_mbps": "mean", "sent_mbps": "mean", "estimated_physical_mbps": "mean", "retrans_factor": "mean"}).reset_index()

            # Plot estimated physical throughput (includes retransmissions)
            sns.lineplot(
                data=goodput_agg,
                x="epoch",
                y="estimated_physical_mbps",
                color=COLORS["phase_line"],  # Red to highlight overhead
                linewidth=2,
                marker="^",
                markersize=8,
                label="Est. Physical (w/ Retrans)",
                alpha=0.8,
                linestyle="--",
                ax=ax,
            )

            # Plot sent throughput (application layer)
            sns.lineplot(
                data=goodput_agg,
                x="epoch",
                y="sent_mbps",
                color=COLORS["bar_secondary"],
                linewidth=2,
                marker="s",
                markersize=8,
                label="Sent (App Layer)",
                alpha=0.7,
                ax=ax,
            )
            # Plot received (goodput)
            sns.lineplot(
                data=goodput_agg,
                x="epoch",
                y="goodput_mbps",
                color=COLORS["goodput"],
                linewidth=2.5,
                marker="o",
                markersize=8,
                label="Goodput (Received)",
                ax=ax,
            )
            ax.fill_between(
                goodput_agg["epoch"],
                goodput_agg["goodput_mbps"],
                alpha=0.3,
                color=COLORS["goodput"],
            )
            ax.set_ylabel("Throughput [Mbps]")
            ax.set_xlabel("Epoch")
            ax.legend()

            # Add retrans factor info to title
            avg_retrans = goodput_agg["retrans_factor"].mean()
            if avg_retrans > 1.01:
                ax.set_title(f"Throughput (Sent vs Goodput) - {experiment_name}\n(Avg Retrans Factor: {avg_retrans:.2f}x)")
            else:
                ax.set_title(f"Throughput (Sent vs Goodput) - {experiment_name}")

    if not has_meaningful_goodput_data:
        ax.text(
            0.5,
            0.5,
            "No data transfer recorded\n(UDP not enabled or no model sharing)",
            ha="center",
            va="center",
            fontsize=14,
            color=COLORS["no_data"],
            transform=ax.transAxes,
        )
        ax.axis("off")
        ax.set_title(f"Goodput (Effective Throughput) - {experiment_name}")
    fig.tight_layout()
    fig.savefig(analysis_dir / "goodput.png", dpi=150)
    plt.close(fig)
    return "goodput.png"


def _generate_traffic_volume_plot(df, experiment_name, analysis_dir):
    """Generate traffic volume plot with cumulative line (WAFL phase only)."""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    has_meaningful_traffic_data = False

    if not df.empty and "bytes_sent" in df.columns:
        # Filter to WAFL phase only
        wafl_df = df[df["phase"] == "WAFL"].copy()
        traffic_data = wafl_df[wafl_df["bytes_sent"].notna()].copy()
        if not traffic_data.empty and traffic_data["bytes_sent"].sum() > 0:
            has_meaningful_traffic_data = True
            traffic_agg = traffic_data.groupby("epoch").agg({"bytes_sent": "sum"}).reset_index()
            traffic_agg["sent_mb"] = traffic_agg["bytes_sent"] / (1024 * 1024)
            traffic_agg["cumulative_mb"] = traffic_agg["sent_mb"].cumsum()

            # Bar chart for per-epoch traffic
            ax1.bar(
                traffic_agg["epoch"],
                traffic_agg["sent_mb"],
                color=COLORS["traffic"],
                alpha=0.8,
                edgecolor=COLORS["traffic"],
                linewidth=0.5,
                label="Per-epoch Traffic",
                width=0.6,
            )
            ax1.set_ylabel("Sent Data per Epoch [MB]", color=COLORS["traffic"])
            ax1.set_xlabel("Epoch")
            ax1.tick_params(axis="y", labelcolor=COLORS["traffic"])

            # Line chart for cumulative traffic (secondary y-axis)
            ax2 = ax1.twinx()
            sns.lineplot(
                data=traffic_agg,
                x="epoch",
                y="cumulative_mb",
                color=COLORS["mean_line"],
                linewidth=2.5,
                marker="",
                label="Cumulative",
                ax=ax2,
            )
            ax2.set_ylabel("Cumulative Sent Data [MB]", color=COLORS["mean_line"])
            ax2.tick_params(axis="y", labelcolor=COLORS["mean_line"])

            # Add legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

            # Explicitly remove ax2 legend if it exists to prevent overlap
            if ax2.get_legend():
                ax2.get_legend().remove()

            ax1.set_title(f"Traffic Volume (WAFL Phase) - {experiment_name}")
            fig.tight_layout()
            fig.savefig(analysis_dir / "traffic_volume.png", dpi=150)
            plt.close(fig)
            return "traffic_volume.png"

    if not has_meaningful_traffic_data:
        ax1.text(
            0.5,
            0.5,
            "No data transfer recorded\n(UDP not enabled or no model sharing)",
            ha="center",
            va="center",
            fontsize=14,
            color=COLORS["no_data"],
            transform=ax1.transAxes,
        )
        ax1.axis("off")
        ax1.set_title(f"Traffic Volume - {experiment_name}")

    fig.tight_layout()
    fig.savefig(analysis_dir / "traffic_volume.png", dpi=150)
    plt.close(fig)
    return "traffic_volume.png"


def _generate_transfer_time_plot(df, experiment_name, analysis_dir):
    """Generate total transfer time plot with compression time visibility."""

    if not df.empty and "compression_time_ms" in df.columns and "epoch_duration_ms" in df.columns:
        # User Request: Show only after WAFL Start
        transfer_data = df[(df["phase"] == "WAFL") & (df["compression_time_ms"].notna()) & (df["epoch_duration_ms"].notna())].copy()
        if not transfer_data.empty and transfer_data["compression_time_ms"].sum() > 0:
            transfer_data["epoch_duration_s"] = transfer_data["epoch_duration_ms"] / 1000
            transfer_data["compression_ms"] = transfer_data["compression_time_ms"]

            transfer_agg = transfer_data.groupby("epoch").agg({"epoch_duration_s": "mean", "compression_ms": "mean"}).reset_index()

            # Create 2 subplots: epoch duration on top, compression time on bottom
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            # Top: Epoch Duration
            # Top: Epoch Duration
            ax1.bar(
                transfer_agg["epoch"],
                transfer_agg["epoch_duration_s"],
                color=COLORS["bar_primary"],
                alpha=0.8,
                label="Epoch Duration",
                edgecolor=COLORS["bar_primary"],
                linewidth=0.5,
                width=0.6,
            )
            ax1.set_ylabel("Epoch Duration [sec]")
            ax1.legend(loc="upper right")
            ax1.set_title(f"Total Transfer Time Breakdown - {experiment_name}")

            # Bottom: Compression Time (in ms for visibility)
            # Bottom: Compression Time (in ms for visibility)
            ax2.bar(
                transfer_agg["epoch"],
                transfer_agg["compression_ms"],
                color=COLORS["bar_secondary"],
                alpha=0.9,
                label="Compression Time (T_comp)",
                edgecolor=COLORS["bar_secondary"],
                linewidth=0.5,
                width=0.6,
            )
            ax2.set_ylabel("Compression Time [ms]")
            ax2.set_xlabel("Epoch")
            ax2.legend(loc="upper right")

            fig.tight_layout()
            fig.savefig(analysis_dir / "total_transfer_time.png", dpi=150)
            plt.close(fig)
            return "total_transfer_time.png"

    # No meaningful data case
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.text(
        0.5,
        0.5,
        "Compression not enabled\n(compression_time_ms = 0)",
        ha="center",
        va="center",
        fontsize=14,
        color=COLORS["no_data"],
        transform=ax.transAxes,
    )
    ax.axis("off")
    ax.set_title(f"Total Transfer Time (T_comm + T_comp) - {experiment_name}")
    fig.tight_layout()
    fig.savefig(analysis_dir / "total_transfer_time.png", dpi=150)
    plt.close(fig)
    return "total_transfer_time.png"


def _generate_compute_comm_breakdown_plot(df, experiment_name, analysis_dir):
    """Generate compute time vs communication time breakdown plot."""
    if df.empty or "compute_time_ms" not in df.columns or "comm_time_ms" not in df.columns:
        return None

    # Only use WAFL phase data (where communication time is meaningful)
    wafl_data = df[(df["phase"] == "WAFL") & (df["compute_time_ms"].notna())].copy()
    if wafl_data.empty:
        return None

    # Convert NaN to 0 for comm_time_ms
    wafl_data["comm_time_ms"] = wafl_data["comm_time_ms"].fillna(0)

    # Handle waiting_time_ms if present
    has_waiting_time = "waiting_time_ms" in wafl_data.columns
    if has_waiting_time:
        wafl_data["waiting_time_ms"] = wafl_data["waiting_time_ms"].fillna(0)
        wafl_data["net_comm_ms"] = wafl_data["comm_time_ms"] - wafl_data["waiting_time_ms"]
        # Ensure non-negative (just in case)
        wafl_data["net_comm_ms"] = wafl_data["net_comm_ms"].clip(lower=0)
    else:
        wafl_data["waiting_time_ms"] = 0
        wafl_data["net_comm_ms"] = wafl_data["comm_time_ms"]

    # Aggregate by epoch
    agg_dict = {
        "compute_time_ms": "mean",
        "comm_time_ms": "mean",  # Total comm time
        "net_comm_ms": "mean",  # Net comm time
        "waiting_time_ms": "mean",  # Waiting time
        "epoch_duration_ms": "mean",
    }

    epoch_agg = wafl_data.groupby("epoch").agg(agg_dict).reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Stacked bar chart
    bar_width = 0.8
    x = epoch_agg["epoch"]

    # Convert to seconds for better readability
    compute_sec = epoch_agg["compute_time_ms"] / 1000
    net_comm_sec = epoch_agg["net_comm_ms"] / 1000
    wait_sec = epoch_agg["waiting_time_ms"] / 1000

    # 1. Compute Time (Bottom)
    ax.bar(
        x,
        compute_sec,
        bar_width,
        label="Compute Time",
        color=COLORS["bar_primary"],
        alpha=0.9,
        edgecolor=COLORS["bar_primary"],
        linewidth=0.5,
    )

    # 2. Net Communication Time (Middle)
    ax.bar(
        x,
        net_comm_sec,
        bar_width,
        bottom=compute_sec,
        label="Net Communication",
        color=COLORS["bar_secondary"],
        alpha=0.9,
        edgecolor=COLORS["bar_secondary"],
        linewidth=0.5,
    )

    # 3. Waiting Time (Top)
    # Only plot if we actually have waiting time data
    if has_waiting_time and wait_sec.sum() > 0:
        ax.bar(
            x,
            wait_sec,
            bar_width,
            bottom=compute_sec + net_comm_sec,
            label="Waiting Time",
            color=COLORS["idle"],  # Reuse idle color
            alpha=0.9,
            edgecolor=COLORS["idle"],
            linewidth=0.5,
            hatch="//",  # Add hatch to distinguish
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Time [sec]")
    ax.legend(loc="upper right")
    ax.set_title(f"Time Breakdown (Compute vs Comm vs Wait) - {experiment_name}")
    fig.tight_layout()
    fig.savefig(analysis_dir / "compute_comm_breakdown.png", dpi=150)
    plt.close(fig)
    return "compute_comm_breakdown.png"


def _generate_time_to_accuracy_plot(df, experiment_name, analysis_dir):
    """Generate time-to-accuracy plot with WAFL phase start as time 0."""
    if df.empty or "test_accuracy" not in df.columns or "wafl_relative_timestamp" not in df.columns:
        return None

    # Only use WAFL phase data for time-to-accuracy (WAFL phase start = 0)
    wafl_data = df[(df["phase"] == "WAFL") & (df["test_accuracy"].notna())].copy()
    if wafl_data.empty:
        # Fallback to all data if no WAFL data
        wafl_data = df[df["test_accuracy"].notna()].copy()
        if wafl_data.empty:
            return None

    fig, ax = plt.subplots(figsize=(12, 6))
    epoch_stats = wafl_data.groupby("epoch").agg(
        {
            "wafl_relative_timestamp": "mean",
            "test_accuracy": ["mean", "std", "min", "max"],
        }
    )
    epoch_stats.columns = ["timestamp", "mean", "std", "min", "max"]
    epoch_stats = epoch_stats.reset_index()
    epoch_stats["std"] = epoch_stats["std"].fillna(0)

    ax.fill_between(
        epoch_stats["timestamp"],
        epoch_stats["mean"] - epoch_stats["std"],
        epoch_stats["mean"] + epoch_stats["std"],
        alpha=0.3,
        color=COLORS["accuracy_fill"],
        label="Mean ± SD",
    )

    ax.plot(
        epoch_stats["timestamp"],
        epoch_stats["mean"],
        color=COLORS["mean_line"],
        linewidth=2.5,
        label="Mean",
        marker="o",
        markersize=3,
    )

    ax.axhline(
        y=TARGET_ACCURACY,
        color=COLORS["target_line"],
        linestyle="--",
        linewidth=2,
        label=f"Target ({TARGET_ACCURACY:.0%})",
    )

    reached_epochs = epoch_stats[epoch_stats["mean"] >= TARGET_ACCURACY]
    if not reached_epochs.empty:
        first_reach = reached_epochs["timestamp"].iloc[0]
        ax.axvline(
            x=first_reach,
            color=COLORS["phase_line"],
            linestyle=":",
            linewidth=2,
            label=f"Target reached: {first_reach:.1f}s",
        )

    ax.set_title(f"Time-to-Accuracy (WAFL Phase, Target: {TARGET_ACCURACY:.0%}) - {experiment_name}")
    ax.set_ylabel("Test Accuracy")
    ax.set_xlabel("Elapsed Time from WAFL Start [sec]")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(analysis_dir / "time_to_accuracy.png", dpi=150)
    plt.close(fig)
    return "time_to_accuracy.png"


def _generate_accuracy_mean_plot(df, experiment_name, analysis_dir, add_phase_line_func):
    """Generate accuracy mean plot with test accuracy emphasized."""
    acc_df = df.melt(
        id_vars=["epoch", "phase", "node"],
        value_vars=["train_accuracy", "test_accuracy"],
        var_name="metric",
        value_name="value",
    )

    if acc_df.empty:
        return None

    # Filter out SSP interrupted rows
    if "is_ssp_interrupted" in df.columns:
        valid_epochs = df[~df["is_ssp_interrupted"]]["epoch"].unique()
        acc_df = acc_df[acc_df["epoch"].isin(valid_epochs)]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot train accuracy first (background, thinner, more transparent)
    train_data = acc_df[acc_df["metric"] == "train_accuracy"]
    test_data = acc_df[acc_df["metric"] == "test_accuracy"]

    sns.lineplot(
        data=train_data,
        x="epoch",
        y="value",
        color=COLORS["accuracy_fill"],
        linewidth=1.5,
        alpha=0.5,
        label="Train Accuracy",
        errorbar="sd",
        ax=ax,
    )
    # Plot test accuracy (foreground, thicker, more prominent)
    sns.lineplot(
        data=test_data,
        x="epoch",
        y="value",
        color=COLORS["mean_line"],
        linewidth=2.5,
        label="Test Accuracy",
        errorbar="sd",
        ax=ax,
    )
    add_phase_line_func(ax, df, "epoch", text_position="bottom")
    ax.set_title(f"Accuracy over Epochs (Mean +/- SD) - {experiment_name}")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Epoch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(analysis_dir / "accuracy_mean.png", dpi=150)
    plt.close(fig)
    return "accuracy_mean.png"


def _generate_loss_mean_plot(df, experiment_name, analysis_dir, add_phase_line_func):
    """Generate loss mean plot with test loss emphasized."""
    loss_df = df.melt(
        id_vars=["epoch", "phase", "node"],
        value_vars=["train_loss", "test_loss"],
        var_name="metric",
        value_name="value",
    )

    if loss_df.empty:
        return None

    # Filter out SSP interrupted rows
    if "is_ssp_interrupted" in df.columns:
        valid_epochs = df[~df["is_ssp_interrupted"]]["epoch"].unique()
        loss_df = loss_df[loss_df["epoch"].isin(valid_epochs)]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot train loss first (background, thinner, more transparent)
    train_data = loss_df[loss_df["metric"] == "train_loss"]
    test_data = loss_df[loss_df["metric"] == "test_loss"]

    sns.lineplot(
        data=train_data,
        x="epoch",
        y="value",
        color=COLORS["loss_fill"],
        linewidth=1.5,
        alpha=0.5,
        label="Train Loss",
        errorbar="sd",
        ax=ax,
    )
    # Plot test loss (foreground, thicker, more prominent)
    sns.lineplot(
        data=test_data,
        x="epoch",
        y="value",
        color=COLORS["mean_line"],
        linewidth=2.5,
        label="Test Loss",
        errorbar="sd",
        ax=ax,
    )
    add_phase_line_func(ax, df, "epoch")
    ax.set_title(f"Loss over Epochs (Mean +/- SD) - {experiment_name}")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(analysis_dir / "loss_mean.png", dpi=150)
    plt.close(fig)
    return "loss_mean.png"


def _generate_accuracy_nodes_plot(df, experiment_name, analysis_dir, add_phase_line_func):
    """Generate node-wise accuracy plot."""
    if df.empty:
        return None

    # Filter out SSP interrupted rows
    plot_df = df[df["test_accuracy"].notna()].copy()
    if "is_ssp_interrupted" in plot_df.columns:
        plot_df = plot_df[~plot_df["is_ssp_interrupted"]]
    if plot_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    num_nodes = plot_df["node"].nunique()
    palette = NODE_PALETTE[:num_nodes] if num_nodes <= len(NODE_PALETTE) else "husl"

    sns.lineplot(
        data=plot_df,
        x="epoch",
        y="test_accuracy",
        hue="node",
        alpha=0.4,
        legend=False,
        palette=palette,
        estimator=None,
        linewidth=1.2,
        ax=ax,
    )
    sns.lineplot(
        data=plot_df,
        x="epoch",
        y="test_accuracy",
        color=COLORS["mean_line"],
        linewidth=2.5,
        label="Mean",
        errorbar=None,
        ax=ax,
    )
    add_phase_line_func(ax, plot_df, "epoch", text_position="bottom")
    ax.set_title(f"Node-wise Test Accuracy - {experiment_name}")
    ax.set_ylabel("Test Accuracy")
    ax.set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(analysis_dir / "accuracy_nodes.png", dpi=150)
    plt.close(fig)
    return "accuracy_nodes.png"


def _generate_loss_nodes_plot(df, experiment_name, analysis_dir, add_phase_line_func):
    """Generate node-wise loss plot."""
    if df.empty:
        return None

    # Filter out SSP interrupted rows
    plot_df = df[df["test_loss"].notna()].copy()
    if "is_ssp_interrupted" in plot_df.columns:
        plot_df = plot_df[~plot_df["is_ssp_interrupted"]]
    if plot_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    num_nodes = plot_df["node"].nunique()
    palette = NODE_PALETTE[:num_nodes] if num_nodes <= len(NODE_PALETTE) else "husl"

    sns.lineplot(
        data=plot_df,
        x="epoch",
        y="test_loss",
        hue="node",
        alpha=0.4,
        legend=False,
        palette=palette,
        estimator=None,
        linewidth=1.2,
        ax=ax,
    )
    sns.lineplot(
        data=plot_df,
        x="epoch",
        y="test_loss",
        color=COLORS["mean_line"],
        linewidth=2.5,
        label="Mean",
        errorbar=None,
        ax=ax,
    )
    add_phase_line_func(ax, plot_df, "epoch")
    ax.set_title(f"Node-wise Test Loss - {experiment_name}")
    ax.set_ylabel("Test Loss")
    ax.set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(analysis_dir / "loss_nodes.png", dpi=150)
    plt.close(fig)
    return "loss_nodes.png"


def _generate_accuracy_vs_time_plot(df, experiment_name, analysis_dir, add_phase_line_func):
    """Generate test accuracy vs elapsed time plot (all phases)."""
    if df.empty or "test_accuracy" not in df.columns or "timestamp" not in df.columns:
        return None

    # Use all data with valid test_accuracy
    plot_data = df[df["test_accuracy"].notna()].copy()
    if "is_ssp_interrupted" in plot_data.columns:
        plot_data = plot_data[~plot_data["is_ssp_interrupted"]]
    if plot_data.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate epoch stats with mean time and accuracy
    epoch_stats = plot_data.groupby("epoch").agg({"timestamp": "mean", "test_accuracy": ["mean", "std"]})
    epoch_stats.columns = ["timestamp", "mean", "std"]
    epoch_stats = epoch_stats.reset_index()
    epoch_stats["std"] = epoch_stats["std"].fillna(0)

    # Plot shaded area for std deviation
    ax.fill_between(
        epoch_stats["timestamp"],
        epoch_stats["mean"] - epoch_stats["std"],
        epoch_stats["mean"] + epoch_stats["std"],
        alpha=0.3,
        color=COLORS["accuracy_fill"],
        label="Mean ± SD",
    )

    # Plot mean line
    ax.plot(
        epoch_stats["timestamp"],
        epoch_stats["mean"],
        color=COLORS["mean_line"],
        linewidth=2.5,
        label="Mean",
        marker="o",
        markersize=3,
    )

    # Add target accuracy line
    ax.axhline(
        y=TARGET_ACCURACY,
        color=COLORS["target_line"],
        linestyle="--",
        linewidth=2,
        label=f"Target ({TARGET_ACCURACY:.0%})",
    )

    # Add phase switch line (using helper)
    add_phase_line_func(ax, plot_data, x_col="wafl_relative_timestamp", text_position="bottom")

    ax.set_title(f"Test Accuracy vs Time - {experiment_name}")
    ax.set_ylabel("Test Accuracy")
    ax.set_xlabel("Elapsed Time [sec]")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(analysis_dir / "accuracy_vs_time.png", dpi=150)
    plt.close(fig)
    return "accuracy_vs_time.png"


def _generate_fec_overhead_plot(df, experiment_name, analysis_dir):
    """Generate FEC processing overhead plot (encode + decode time)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    has_meaningful_data = False

    if not df.empty and "fec_encode_time_ms" in df.columns and "fec_decode_time_ms" in df.columns:
        # Filter to WAFL phase only
        wafl_df = df[df["phase"] == "WAFL"].copy()
        fec_data = wafl_df[(wafl_df["fec_encode_time_ms"].notna()) | (wafl_df["fec_decode_time_ms"].notna())].copy()

        if not fec_data.empty:
            # Check if there's any meaningful FEC data
            total_encode = fec_data["fec_encode_time_ms"].fillna(0).sum()
            total_decode = fec_data["fec_decode_time_ms"].fillna(0).sum()

            if total_encode > 0 or total_decode > 0:
                has_meaningful_data = True

                # Aggregate by epoch (mean across nodes)
                fec_agg = fec_data.groupby("epoch").agg({"fec_encode_time_ms": "mean", "fec_decode_time_ms": "mean"}).reset_index()
                fec_agg["fec_encode_time_ms"] = fec_agg["fec_encode_time_ms"].fillna(0)
                fec_agg["fec_decode_time_ms"] = fec_agg["fec_decode_time_ms"].fillna(0)

                bar_width = 0.8

                # Stacked bar chart: Encode (bottom) + Decode (top)
                ax.bar(
                    fec_agg["epoch"],
                    fec_agg["fec_encode_time_ms"],
                    bar_width,
                    label="FEC Encode",
                    color=COLORS["bar_primary"],
                    alpha=0.9,
                    edgecolor=COLORS["bar_primary"],
                    linewidth=0.5,
                )
                ax.bar(
                    fec_agg["epoch"],
                    fec_agg["fec_decode_time_ms"],
                    bar_width,
                    bottom=fec_agg["fec_encode_time_ms"],
                    label="FEC Decode",
                    color=COLORS["bar_secondary"],
                    alpha=0.9,
                    edgecolor=COLORS["bar_secondary"],
                    linewidth=0.5,
                )

                ax.set_xlabel("Epoch")
                ax.set_ylabel("Time [ms]")
                ax.legend(loc="upper right")

                # Calculate average overhead for title
                avg_encode = fec_agg["fec_encode_time_ms"].mean()
                avg_decode = fec_agg["fec_decode_time_ms"].mean()
                avg_total = avg_encode + avg_decode

                ax.set_title(f"FEC Processing Overhead - {experiment_name}\n(Avg: Encode={avg_encode:.1f}ms, Decode={avg_decode:.1f}ms, Total={avg_total:.1f}ms)")
                fig.tight_layout()
                fig.savefig(analysis_dir / "fec_overhead.png", dpi=150)
                plt.close(fig)
                return "fec_overhead.png"

    if not has_meaningful_data:
        ax.text(
            0.5,
            0.5,
            "FEC not enabled or no processing data\n(UDP with FEC disabled or TCP mode)",
            ha="center",
            va="center",
            fontsize=14,
            color=COLORS["no_data"],
            transform=ax.transAxes,
        )
        ax.axis("off")
        ax.set_title(f"FEC Processing Overhead - {experiment_name}")

    fig.tight_layout()
    fig.savefig(analysis_dir / "fec_overhead.png", dpi=150)
    plt.close(fig)
    return "fec_overhead.png"


def _generate_cumulative_sent_data_plot(df, experiment_name, analysis_dir):
    """Generate cumulative sent data (bytes_sent) bar plot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    has_meaningful_data = False

    if not df.empty and "bytes_sent" in df.columns:
        traffic_data = df[df["bytes_sent"].notna()].copy()
        if not traffic_data.empty and traffic_data["bytes_sent"].sum() > 0:
            has_meaningful_data = True

            # Aggregate bytes_sent per epoch (sum across all nodes)
            # Use 'bytes_sent' which is Physical Bytes (Total Network Load: Payload + Headers + Retrans)
            traffic_agg = traffic_data.groupby("epoch").agg({"bytes_sent": "sum"}).reset_index()
            traffic_agg["sent_mb"] = traffic_agg["bytes_sent"] / (1024 * 1024)

            # Calculate cumulative sum
            traffic_agg["cumulative_mb"] = traffic_agg["sent_mb"].cumsum()

            ax.bar(
                traffic_agg["epoch"],
                traffic_agg["cumulative_mb"],
                color=COLORS["traffic"],
                alpha=0.8,
                edgecolor=COLORS["traffic"],
                linewidth=0.5,
                width=0.6,
            )
            ax.set_ylabel("Cumulative Sent Data [MB]")
            ax.set_xlabel("Epoch")
            ax.set_title(f"Cumulative Sent Data - {experiment_name}")

    if not has_meaningful_data:
        ax.text(
            0.5,
            0.5,
            "No data transfer recorded\n(UDP not enabled or no model sharing)",
            ha="center",
            va="center",
            fontsize=14,
            color=COLORS["no_data"],
            transform=ax.transAxes,
        )
        ax.axis("off")
        ax.set_title(f"Cumulative Sent Data - {experiment_name}")

    fig.tight_layout()
    fig.savefig(analysis_dir / "cumulative_sent_data.png", dpi=150)
    plt.close(fig)
    return "cumulative_sent_data.png"


def _generate_udp_dynamic_params_plot(df, experiment_name, analysis_dir, add_phase_line_func):
    """Generate UDP dynamic parameters plot (Parity & Pacing)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    if not df.empty and ("udp_avg_parity" in df.columns or "udp_avg_pacing_ms" in df.columns):
        wafl_df = df[df["phase"] == "WAFL"].copy()

        # Check if there is any non-zero data
        has_parity = "udp_avg_parity" in wafl_df.columns and wafl_df["udp_avg_parity"].sum() > 0
        has_pacing = "udp_avg_pacing_ms" in wafl_df.columns and wafl_df["udp_avg_pacing_ms"].sum() > 0

        if has_parity or has_pacing:
            # Aggregate per epoch (mean across nodes)
            agg_funcs = {}
            if "udp_avg_parity" in wafl_df.columns:
                agg_funcs["udp_avg_parity"] = "mean"
            if "udp_avg_pacing_ms" in wafl_df.columns:
                agg_funcs["udp_avg_pacing_ms"] = "mean"

            param_agg = wafl_df.groupby("epoch").agg(agg_funcs).reset_index()

            # Subplot 1: Average Parity
            if has_parity:
                sns.lineplot(data=param_agg, x="epoch", y="udp_avg_parity", color=COLORS["wasted_bar"], linewidth=2, label="Avg Parity (m-k)", ax=ax1)
            ax1.set_title(f"Dynamic UDP Parameters - {experiment_name}")
            ax1.set_ylabel("Avg Parity")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Subplot 2: Average Pacing
            if has_pacing:
                sns.lineplot(data=param_agg, x="epoch", y="udp_avg_pacing_ms", color=COLORS["goodput"], linewidth=2, label="Avg Pacing Delay [ms]", ax=ax2)
            ax2.set_ylabel("Pacing [ms]")
            ax2.set_xlabel("Epoch")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            fig.tight_layout()
            fig.savefig(analysis_dir / "udp_dynamic_params.png", dpi=150)
            plt.close(fig)
            return "udp_dynamic_params.png"

    # Fallback/No Data
    plt.close(fig)  # Close the unused subplot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.text(
        0.5,
        0.5,
        "UDP Adaptive Parameters not relevant\n(Protocol not enabled or static)",
        ha="center",
        va="center",
        fontsize=14,
        color=COLORS["no_data"],
        transform=ax.transAxes,
    )
    ax.axis("off")
    ax.set_title(f"Dynamic UDP Parameters - {experiment_name}")

    fig.tight_layout()
    fig.savefig(analysis_dir / "udp_dynamic_params.png", dpi=150)
    plt.close(fig)
    return "udp_dynamic_params.png"


def _generate_protocol_distribution_plot(df, experiment_name, analysis_dir, add_phase_line_func):
    """Generate protocol distribution plot (Dynamic Mode)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    has_meaningful_data = False

    # Check columns existence
    cols = ["protocol_tcp_count", "protocol_udp_count", "protocol_rudp_count"]
    avail_cols = [c for c in cols if c in df.columns]

    if not df.empty and avail_cols:
        wafl_df = df[df["phase"] == "WAFL"].copy()

        # Check if total sum > 0
        total_counts = wafl_df[avail_cols].sum().sum()
        if total_counts > 0:
            has_meaningful_data = True

            # Aggregate sums per epoch
            agg_funcs = {c: "sum" for c in avail_cols}
            proto_agg = wafl_df.groupby("epoch").agg(agg_funcs).reset_index()

            bottom = None
            labels = {"protocol_tcp_count": "TCP", "protocol_udp_count": "UDP", "protocol_rudp_count": "RUDP"}
            colors = {"protocol_tcp_count": "#3498db", "protocol_udp_count": "#e67e22", "protocol_rudp_count": "#9b59b6"}  # Blue, Orange, Purple

            for col in avail_cols:
                if col not in proto_agg.columns:
                    continue

                # Check if this protocol was actually used
                if proto_agg[col].sum() == 0:
                    continue

                ax.bar(
                    proto_agg["epoch"],
                    proto_agg[col],
                    label=labels[col],
                    bottom=bottom,
                    color=colors.get(col, "gray"),
                    alpha=0.8,
                    width=0.8,
                    edgecolor=colors.get(col, "gray"),
                    linewidth=0.5,
                )
                if bottom is None:
                    bottom = proto_agg[col]
                else:
                    bottom += proto_agg[col]

            ax.set_ylabel("Total Transfers (Count)")
            ax.set_xlabel("Epoch")
            ax.legend()
            ax.set_title(f"Dynamic Protocol Distribution - {experiment_name}")
            ax.grid(True, axis="y", alpha=0.3)

    if not has_meaningful_data:
        ax.text(
            0.5,
            0.5,
            "Dynamic Protocol Distribution N/A\n(Stats not collected or Single Protocol)",
            ha="center",
            va="center",
            fontsize=14,
            color=COLORS["no_data"],
            transform=ax.transAxes,
        )
        ax.axis("off")
        ax.set_title(f"Dynamic Protocol Distribution - {experiment_name}")

    fig.tight_layout()
    fig.savefig(analysis_dir / "protocol_distribution.png", dpi=150)
    plt.close(fig)
    return "protocol_distribution.png"


def _generate_udp_recovery_plot(df, experiment_name, analysis_dir, add_phase_line_func):
    """Generate UDP FEC recovery stats (Success vs Fail) plot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    has_meaningful_data = False

    if not df.empty and ("fec_recovery_success" in df.columns or "fec_recovery_fail" in df.columns):
        wafl_df = df[df["phase"] == "WAFL"].copy()

        has_success = "fec_recovery_success" in wafl_df.columns and wafl_df["fec_recovery_success"].sum() > 0
        has_fail = "fec_recovery_fail" in wafl_df.columns and wafl_df["fec_recovery_fail"].sum() > 0

        if has_success or has_fail:
            has_meaningful_data = True

            # Aggregate per epoch (sum across nodes)
            agg_funcs = {}
            if "fec_recovery_success" in wafl_df.columns:
                agg_funcs["fec_recovery_success"] = "sum"
            if "fec_recovery_fail" in wafl_df.columns:
                agg_funcs["fec_recovery_fail"] = "sum"

            rec_agg = wafl_df.groupby("epoch").agg(agg_funcs).reset_index()

            bar_width = 0.8

            s_vals = rec_agg["fec_recovery_success"] if has_success else pd.Series([0] * len(rec_agg))
            f_vals = rec_agg["fec_recovery_fail"] if has_fail else pd.Series([0] * len(rec_agg))

            ax.bar(
                rec_agg["epoch"],
                s_vals,
                label="FEC Recovered",
                color=COLORS["bar_primary"],
                alpha=0.9,
                width=bar_width,
                edgecolor=COLORS["bar_primary"],
                linewidth=0.5,
            )
            ax.bar(
                rec_agg["epoch"],
                f_vals,
                bottom=s_vals,
                label="Recovery Failed",
                color=COLORS["control_server"],
                alpha=0.9,
                width=bar_width,
                edgecolor=COLORS["control_server"],
                linewidth=0.5,
            )

            ax.set_ylabel("Models (Count)")
            ax.set_xlabel("Epoch")
            ax.legend()
            ax.set_title(f"UDP FEC Recovery Status - {experiment_name}")
            ax.grid(True, axis="y", alpha=0.3)

    if not has_meaningful_data:
        ax.text(0.5, 0.5, "UDP FEC stats N/A", ha="center", va="center", color=COLORS["no_data"], transform=ax.transAxes)
        ax.axis("off")
        ax.set_title(f"UDP FEC Recovery Status - {experiment_name}")

    fig.tight_layout()
    fig.savefig(analysis_dir / "udp_recovery.png", dpi=150)
    plt.close(fig)
    return "udp_recovery.png"


def _generate_rudp_failure_plot(df, experiment_name, analysis_dir, add_phase_line_func):
    """Generate RUDP failure (Max Retries Reached) plot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    has_meaningful_data = False

    if not df.empty and "rudp_max_retries_reached" in df.columns:
        wafl_df = df[df["phase"] == "WAFL"].copy()
        if wafl_df["rudp_max_retries_reached"].sum() > 0:
            has_meaningful_data = True

            agg = wafl_df.groupby("epoch").agg({"rudp_max_retries_reached": "sum"}).reset_index()

            ax.bar(
                agg["epoch"],
                agg["rudp_max_retries_reached"],
                color=COLORS["control_server"],
                alpha=0.8,
                edgecolor=COLORS["control_server"],
                linewidth=0.5,
                width=0.6,
            )

            add_phase_line_func(ax, df, "epoch")
            ax.set_ylabel("Max Retries Reached (Count)")
            ax.set_xlabel("Epoch")
            ax.set_title(f"RUDP Max Retry Failures - {experiment_name}")
            ax.grid(True, axis="y", alpha=0.3)

    if not has_meaningful_data:
        ax.text(0.5, 0.5, "RUDP Failures N/A", ha="center", va="center", color=COLORS["no_data"], transform=ax.transAxes)
        ax.axis("off")
        ax.set_title(f"RUDP Max Retry Failures - {experiment_name}")

    fig.tight_layout()
    fig.savefig(analysis_dir / "rudp_failures.png", dpi=150)
    plt.close(fig)
    return "rudp_failures.png"


def _generate_rudp_control_overhead_plot(df, experiment_name, analysis_dir, add_phase_line_func):
    """Generate RUDP Control Packet Overhead plot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    has_meaningful_data = False

    cols = ["rudp_acks_sent", "rudp_eaks_sent", "rudp_nacks_sent"]
    avail_cols = [c for c in cols if c in df.columns]

    if not df.empty and avail_cols:
        wafl_df = df[df["phase"] == "WAFL"].copy()
        if wafl_df[avail_cols].sum().sum() > 0:
            has_meaningful_data = True

            sns.lineplot(data=wafl_df, x="epoch", y="rudp_acks_sent", label="ACKs", errorbar=None, ax=ax)
            if "rudp_eaks_sent" in wafl_df.columns:
                sns.lineplot(data=wafl_df, x="epoch", y="rudp_eaks_sent", label="EAKs", errorbar=None, ax=ax)
            if "rudp_nacks_sent" in wafl_df.columns:
                sns.lineplot(data=wafl_df, x="epoch", y="rudp_nacks_sent", label="NACKs", errorbar=None, ax=ax)

            add_phase_line_func(ax, df, "epoch")
            ax.set_ylabel("Control Packets (Avg per Node)")
            ax.set_xlabel("Epoch")
            ax.set_title(f"RUDP Control Overhead - {experiment_name}")
            ax.legend()
            ax.grid(True, alpha=0.3)

    if not has_meaningful_data:
        ax.text(0.5, 0.5, "RUDP Overhead N/A", ha="center", va="center", color=COLORS["no_data"], transform=ax.transAxes)
        ax.axis("off")
        ax.set_title(f"RUDP Control Overhead - {experiment_name}")

    fig.tight_layout()
    fig.savefig(analysis_dir / "rudp_control_overhead.png", dpi=150)
    plt.close(fig)
    return "rudp_control_overhead.png"


def _generate_compression_stats_plot(df, experiment_name, analysis_dir, add_phase_line_func):
    """Generate Compression Ratio and Time plot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    if not df.empty and "compression_ratio" in df.columns:
        wafl_df = df[df["phase"] == "WAFL"].copy()

        # Check if compression was active (ratio != 1.0 or time > 0)
        # Checking time > 0 is safer.
        if wafl_df["compression_time_ms"].mean() > 0.001:
            agg = wafl_df.groupby("epoch").agg({"compression_ratio": "mean", "compression_time_ms": "mean"}).reset_index()

            # Subplot 1: Ratio
            sns.lineplot(data=agg, x="epoch", y="compression_ratio", color=COLORS["goodput"], linewidth=2, ax=ax1)
            ax1.set_title(f"Compression Statistics - {experiment_name}")
            ax1.set_ylabel("Compression Ratio")
            ax1.grid(True, alpha=0.3)

            # Subplot 2: Time
            sns.lineplot(data=agg, x="epoch", y="compression_time_ms", color=COLORS["wasted_bar"], linewidth=2, ax=ax2)
            ax2.set_ylabel("Compression Time [ms]")
            ax2.set_xlabel("Epoch")
            ax2.grid(True, alpha=0.3)

            fig.tight_layout()
            fig.savefig(analysis_dir / "compression_stats.png", dpi=150)
            plt.close(fig)
            return "compression_stats.png"

    # Fallback/No Data
    # Since we created subplots, we need to clear/reuse them or just make a simple no-data plot
    plt.close(fig)  # Close the unused subplot figure

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.text(0.5, 0.5, "Compression not used", ha="center", va="center", color=COLORS["no_data"], transform=ax.transAxes)
    ax.axis("off")
    ax.set_title(f"Compression Statistics - {experiment_name}")

    fig.tight_layout()
    fig.savefig(analysis_dir / "compression_stats.png", dpi=150)
    plt.close(fig)
    return "compression_stats.png"


def _calculate_metrics_summary(df, wafl_phase_start_relative=None, group_name=""):
    """Calculate summary metrics for reporting."""
    metrics = {}

    if df.empty:
        return metrics

    # Filter to WAFL phase for most metrics
    wafl_df = df[df["phase"] == "WAFL"].copy() if "phase" in df.columns else df.copy()

    # Filter out SSP interrupted rows for accuracy-related metrics
    clean_df = df.copy()
    if "is_ssp_interrupted" in clean_df.columns:
        clean_df = clean_df[~clean_df["is_ssp_interrupted"]]

    wafl_clean = wafl_df.copy()
    if "is_ssp_interrupted" in wafl_clean.columns:
        wafl_clean = wafl_clean[~wafl_clean["is_ssp_interrupted"]]

    # 1. Final Test Accuracy (last epoch mean)
    if "test_accuracy" in clean_df.columns and not clean_df[clean_df["test_accuracy"].notna()].empty:
        final_epoch = clean_df["epoch"].max()
        final_acc = clean_df[clean_df["epoch"] == final_epoch]["test_accuracy"].mean()
        metrics["final_accuracy"] = final_acc

    # 2. Max Test Accuracy (WAFL phase)
    if "test_accuracy" in wafl_clean.columns and not wafl_clean[wafl_clean["test_accuracy"].notna()].empty:
        metrics["max_accuracy"] = wafl_clean["test_accuracy"].max()

    # 3. Time to Target Accuracy
    if "wafl_relative_timestamp" in wafl_clean.columns and "test_accuracy" in wafl_clean.columns:
        # Determine target based on group name
        target_val = 0.90 if "Experiment 4" in group_name else TARGET_ACCURACY

        epoch_acc = wafl_clean.groupby("epoch").agg({"test_accuracy": "mean", "wafl_relative_timestamp": "mean"}).reset_index()
        reached = epoch_acc[epoch_acc["test_accuracy"] >= target_val]
        if not reached.empty:
            metrics["time_to_target"] = reached["wafl_relative_timestamp"].iloc[0]

    # 4. Average Epoch Duration (WAFL phase)
    if "epoch_duration_ms" in wafl_df.columns:
        avg_dur = wafl_df["epoch_duration_ms"].mean() / 1000
        metrics["avg_epoch_duration_s"] = avg_dur

    # 5. Total Training Time
    if "timestamp" in df.columns:
        total_time = df["timestamp"].max() - df["timestamp"].min()
        metrics["total_time_s"] = total_time

    # 6. Total Traffic (bytes_sent)
    if "bytes_sent" in wafl_df.columns:
        total_bytes = wafl_df["bytes_sent"].sum()
        metrics["total_traffic_mb"] = total_bytes / (1024 * 1024)

    # 7. Average Survival Rate
    if "survival_rate" in wafl_df.columns:
        avg_survival = wafl_df["survival_rate"].mean()
        metrics["avg_survival_rate"] = avg_survival

    # 8. Average Goodput (Mbps)
    if "bytes_received" in wafl_df.columns and "comm_time_ms" in wafl_df.columns:
        valid = wafl_df[(wafl_df["comm_time_ms"].notna()) & (wafl_df["comm_time_ms"] > 50)]  # Min 50ms
        if not valid.empty and valid["bytes_received"].sum() > 0:
            total_bytes = valid["bytes_received"].sum()
            total_time_s = valid["comm_time_ms"].sum() / 1000
            avg_goodput = (total_bytes * 8 / 1e6) / total_time_s
            avg_goodput = (total_bytes * 8 / 1e6) / total_time_s
            metrics["avg_goodput_mbps"] = avg_goodput

    # 8.5 Average Physical Throughput (Mbps)
    if "bytes_sent" in wafl_df.columns and "comm_time_ms" in wafl_df.columns:
        valid = wafl_df[(wafl_df["comm_time_ms"].notna()) & (wafl_df["comm_time_ms"] > 50)]
        if not valid.empty and valid["bytes_sent"].sum() > 0:
            total_bytes = valid["bytes_sent"].sum()
            total_time_s = valid["comm_time_ms"].sum() / 1000
            avg_phy = (total_bytes * 8 / 1e6) / total_time_s
            metrics["avg_phy_throughput_mbps"] = avg_phy

    # 9. Number of Nodes
    if "node" in df.columns:
        metrics["num_nodes"] = df["node"].nunique()

    # 10. Total Epochs
    if "epoch" in df.columns:
        metrics["total_epochs"] = int(df["epoch"].max())

    # 11. WAFL Epochs
    if "epoch" in wafl_df.columns and not wafl_df.empty:
        metrics["wafl_epochs"] = int(wafl_df["epoch"].max() - wafl_df["epoch"].min() + 1)

    # 12. Traffic Overhead Ratio
    if "bytes_sent" in wafl_df.columns and "app_bytes_sent" in wafl_df.columns:
        phy_bytes = wafl_df["bytes_sent"].sum()
        app_bytes = wafl_df["app_bytes_sent"].sum()
        if app_bytes > 0:
            metrics["traffic_overhead_ratio"] = phy_bytes / app_bytes

    # 13. FEC Processing Time (encode + decode)
    if "fec_encode_time_ms" in wafl_df.columns:
        avg_encode = wafl_df["fec_encode_time_ms"].mean()
        if pd.notna(avg_encode) and avg_encode > 0:
            metrics["avg_fec_encode_ms"] = avg_encode
    if "fec_decode_time_ms" in wafl_df.columns:
        avg_decode = wafl_df["fec_decode_time_ms"].mean()
        if pd.notna(avg_decode) and avg_decode > 0:
            metrics["avg_fec_decode_ms"] = avg_decode

    # 14. Compression Stats
    if "compression_ratio" in wafl_df.columns:
        metrics["compression_ratio_avg"] = wafl_df["compression_ratio"].mean()
    if "compression_time_ms" in wafl_df.columns:
        metrics["compression_time_avg"] = wafl_df["compression_time_ms"].mean()

    return metrics


def _generate_experiment_report(df, experiment_id, experiment_name, analysis_dir, wafl_phase_start_relative=None):
    """Generate Markdown report for a single experiment."""
    metrics = _calculate_metrics_summary(df, wafl_phase_start_relative)

    report_path = analysis_dir / "report.md"

    lines = []
    lines.append(f"# {experiment_name}")
    lines.append("")
    lines.append(f"**Experiment ID**: `{experiment_id}`")
    lines.append("")

    # Summary Table
    lines.append("## Summary Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")

    if "num_nodes" in metrics:
        lines.append(f"| Number of Nodes | {metrics['num_nodes']} |")
    if "total_epochs" in metrics:
        lines.append(f"| Total Epochs | {metrics['total_epochs']} |")
    if "wafl_epochs" in metrics:
        lines.append(f"| WAFL Epochs | {metrics['wafl_epochs']} |")
    if "final_accuracy" in metrics:
        lines.append(f"| Final Test Accuracy | {metrics['final_accuracy']:.4f} ({metrics['final_accuracy'] * 100:.2f}%) |")
    if "max_accuracy" in metrics:
        lines.append(f"| Max Test Accuracy | {metrics['max_accuracy']:.4f} ({metrics['max_accuracy'] * 100:.2f}%) |")
    if "time_to_target" in metrics:
        lines.append(f"| Time to {TARGET_ACCURACY * 100:.0f}% Target | {metrics['time_to_target']:.2f} sec |")
    if "total_time_s" in metrics:
        lines.append(f"| Total Training Time | {metrics['total_time_s']:.2f} sec |")
    if "avg_epoch_duration_s" in metrics:
        lines.append(f"| Avg Epoch Duration (WAFL) | {metrics['avg_epoch_duration_s']:.2f} sec |")
    if "total_traffic_mb" in metrics:
        lines.append(f"| Total Traffic (WAFL) | {metrics['total_traffic_mb']:.2f} MB |")
    if "avg_survival_rate" in metrics:
        lines.append(f"| Avg Survival Rate | {metrics['avg_survival_rate']:.4f} ({metrics['avg_survival_rate'] * 100:.2f}%) |")
    if "avg_goodput_mbps" in metrics:
        lines.append(f"| Avg Goodput | {metrics['avg_goodput_mbps']:.2f} Mbps |")
    if "traffic_overhead_ratio" in metrics:
        lines.append(f"| Traffic Overhead Ratio | {metrics['traffic_overhead_ratio']:.2f}x |")
    if "avg_fec_encode_ms" in metrics:
        lines.append(f"| Avg FEC Encode Time | {metrics['avg_fec_encode_ms']:.2f} ms |")
    if "avg_fec_decode_ms" in metrics:
        lines.append(f"| Avg FEC Decode Time | {metrics['avg_fec_decode_ms']:.2f} ms |")
    if "avg_fec_encode_ms" in metrics and "avg_fec_decode_ms" in metrics:
        total_fec = metrics["avg_fec_encode_ms"] + metrics["avg_fec_decode_ms"]
        lines.append(f"| Avg Total FEC Overhead | {total_fec:.2f} ms |")
    if "compression_ratio_avg" in metrics:
        lines.append(f"| Avg Compression Ratio | {metrics['compression_ratio_avg']:.2f}x |")
    if "compression_time_avg" in metrics:
        lines.append(f"| Avg Compression Time | {metrics['compression_time_avg']:.2f} ms |")
    lines.append("")
    lines.append("")

    # Generated Graphs Section
    lines.append("## Generated Graphs")
    lines.append("")

    graph_descriptions = {
        "accuracy_mean.png": "Train and Test Accuracy (Mean ± SD)",
        "accuracy_time.png": "Test Accuracy vs Wall-clock Time",
        "accuracy_nodes.png": "Node-wise Test Accuracy",
        "loss_mean.png": "Train and Test Loss (Mean ± SD)",
        "loss_nodes.png": "Node-wise Test Loss",
        "time_to_accuracy.png": "Time to Target Accuracy",
        "epoch_duration.png": "Wall-clock Time per Epoch",
        "compute_comm_breakdown.png": "Compute vs Communication Time",
        "idle_time_ratio.png": "Idle Time Ratio",
        "survival_rate.png": "Survival Rate (UDP/FEC)",
        "goodput.png": "Goodput (Payload Throughput)",
        "effective_goodput.png": "Effective Goodput (Original Size / Time)",
        "traffic_volume.png": "Traffic Volume per Epoch",
        "total_transfer_time.png": "Total Transfer Time",
        "traffic_overhead.png": "Traffic Overhead Ratio (Physical / App)",
        "throughput_overhead.png": "Throughput Overhead (Physical vs Goodput)",
        "wasted_computation.png": "Wasted Computation (SSP)",
        "fec_overhead.png": "FEC Processing Overhead (Encode + Decode)",
        "udp_dynamic_params.png": "UDP Dynamic Parameters (Parity & Pacing)",
        "udp_recovery.png": "UDP FEC Recovery Status",
        "protocol_distribution.png": "Protocol Distribution (Dynamic Mode)",
        "asymmetry_distribution.png": "Asymmetry Check (Received Models Distribution)",
        "asymmetry_lorenz.png": "Asymmetry Check (Lorenz Curve & Gini)",
        "survivor_quality.png": "Survivor Quality (Accuracy vs Connectivity)",
        "survivor_trajectory_groups.png": "Survivor Trajectory Groups",
        "compression_stats.png": "Compression Statistics (Ratio & Time)",
    }

    for graph_file, description in graph_descriptions.items():
        if (analysis_dir / graph_file).exists():
            lines.append(f"### {description}")
            lines.append("")
            lines.append(f"![{description}]({graph_file})")
            lines.append("")

    # Write report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_path


def analyze_results(experiment_id):
    """Analyze collected results and generate plots in parallel."""
    print(f"📊 Analyzing results for: {experiment_id}")

    exp_dir = RESULTS_DIR / experiment_id
    analysis_dir = exp_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    # Extract experiment name without timestamp
    timestamp_pattern = r"-\d{8}T\d{6}$"
    experiment_name = re.sub(timestamp_pattern, "", experiment_id)

    # Load metadata for WAFL phase start timestamp
    wafl_phase_start_relative = None
    metadata_path = exp_dir / "ctrl" / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
                wafl_phase_start_relative = metadata.get("wafl_phase_start_relative")
                print(f"   WAFL phase start: {wafl_phase_start_relative:.1f}s from experiment start")
        except Exception as e:
            print(f"   ⚠️ Could not load metadata: {e}")

    # Load data
    df, resources_df = _load_metrics_and_resources(exp_dir)

    if df.empty:
        print("⚠️  No metrics data found.")
        # If no data found, remove the analysis directory so it's not marked as "analyzed"
        try:
            analysis_dir.rmdir()
            print("  🗑️ Removed empty analysis folder")
        except OSError:
            pass  # Directory not empty or other error
        return

    # Add WAFL-relative timestamp (time since WAFL phase start)
    # For time-based plots, this makes comparisons fairer by removing SELF phase time
    if wafl_phase_start_relative is not None:
        df["wafl_relative_timestamp"] = df["timestamp"] - wafl_phase_start_relative
        # For SELF phase data, wafl_relative_timestamp will be negative
    else:
        # Fallback: use timestamp as-is (backward compatibility)
        df["wafl_relative_timestamp"] = df["timestamp"]

    # Define all plot generation tasks using picklable (func, args) tuples
    plot_tasks = []

    # helper for constructing args
    def add_task(func, *args):
        plot_tasks.append((func, args))

    add_task(_generate_epoch_duration_plot, df, experiment_name, analysis_dir, _add_phase_line, exp_dir)
    add_task(_generate_wasted_computation_plot, df, experiment_name, analysis_dir)
    add_task(_generate_survival_rate_plot, df, experiment_name, analysis_dir, _add_phase_line)
    add_task(_generate_goodput_plot, df, experiment_name, analysis_dir, _add_phase_line)
    add_task(_generate_effective_goodput_plot, df, experiment_name, analysis_dir, _add_phase_line)
    add_task(_generate_traffic_volume_plot, df, experiment_name, analysis_dir)

    # New / Refined Plots
    add_task(_generate_traffic_overhead_plot, df, experiment_name, analysis_dir, _add_phase_line)
    add_task(_generate_throughput_overhead_plot, df, experiment_name, analysis_dir)
    add_task(_generate_accuracy_time_plot, df, experiment_name, analysis_dir, _add_phase_line)

    add_task(_generate_accuracy_mean_plot, df, experiment_name, analysis_dir, _add_phase_line)
    add_task(_generate_loss_mean_plot, df, experiment_name, analysis_dir, _add_phase_line)
    add_task(_generate_accuracy_nodes_plot, df, experiment_name, analysis_dir, _add_phase_line)
    add_task(_generate_loss_nodes_plot, df, experiment_name, analysis_dir, _add_phase_line)
    add_task(_generate_compute_comm_breakdown_plot, df, experiment_name, analysis_dir)
    add_task(_generate_fec_overhead_plot, df, experiment_name, analysis_dir)
    add_task(_generate_udp_dynamic_params_plot, df, experiment_name, analysis_dir, _add_phase_line)
    add_task(_generate_protocol_distribution_plot, df, experiment_name, analysis_dir, _add_phase_line)
    add_task(_generate_asymmetry_plot, df, experiment_name, analysis_dir, _add_phase_line)
    add_task(_generate_survivor_quality_plot, df, experiment_name, analysis_dir, _add_phase_line)
    add_task(_generate_compression_stats_plot, df, experiment_name, analysis_dir, _add_phase_line)

    # Execute plot generation in parallel using multiprocessing
    generated_plots = []
    print(f"  🔧 Generating {len(plot_tasks)} plots in parallel...")

    # Use max CPUs but leave one or two free for system stability if possible,
    # or just use cpu_count(). For now, cpu_count() is usually safe for non-IO heavy tasks.
    # However, matplotlib uses memory. Let's cap at cpu_count().
    # Note: On shared servers, too many processes might be rude. But for analysis, speed is key.
    # We will use min(cpu_count(), len(plot_tasks))
    num_processes = min(multiprocessing.cpu_count(), len(plot_tasks))

    with multiprocessing.Pool(processes=num_processes) as pool:
        # returns an iterator that yields results as soon as they are ready
        for res in pool.imap_unordered(_run_plot_task, plot_tasks):
            if res:
                generated_plots.append(res)
                print(f"  ✅ Generated {res}")

    # Ensure all figures are closed (redundant if checking individual functions, but safe for any missed ones in main process)
    plt.close("all")

    # Generate Markdown report
    print("  📝 Generating Markdown report...")
    _generate_experiment_report(df, experiment_id, experiment_name, analysis_dir, wafl_phase_start_relative)

    print(f"✨ Analysis complete. {len(generated_plots)} plots + report generated in: {analysis_dir}")


def _generate_goodput_plot(df, experiment_name, analysis_dir, add_phase_line_func):
    """Generate per-node goodput plot using Application Payload (app_bytes_received)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Use 'app_bytes_received' for Goodput (Payload Rate) if available
    target_col = "app_bytes_received"
    if target_col not in df.columns or df[target_col].sum() == 0:
        target_col = "bytes_received"  # Fallback (should not happen with new logic)

    plot_df = df[(df[target_col].notna()) & (df["comm_time_ms"] > 50)].copy()  # Filter noise < 50ms

    if plot_df.empty:
        return None

    # Filter SSP interrupted if present (omitted for brevity, assume standard filter)
    if "is_ssp_interrupted" in plot_df.columns:
        plot_df = plot_df[~plot_df["is_ssp_interrupted"]]

    # Goodput = Payload Bits / Communication Time
    plot_df["goodput_mbps"] = (plot_df[target_col] * 8 / 1e6) / (plot_df["comm_time_ms"] / 1000)

    num_nodes = plot_df["node"].nunique()
    palette = NODE_PALETTE[:num_nodes] if num_nodes <= len(NODE_PALETTE) else "husl"

    sns.lineplot(
        data=plot_df,
        x="epoch",
        y="goodput_mbps",
        hue="node",
        alpha=0.4,
        legend=False,
        palette=palette,
        linewidth=1.2,
        ax=ax,
    )
    sns.lineplot(
        data=plot_df,
        x="epoch",
        y="goodput_mbps",
        color=COLORS["goodput"],
        linewidth=2.5,
        label="Mean Goodput",
        errorbar=None,
        ax=ax,
    )

    add_phase_line_func(ax, plot_df, "epoch")
    ax.set_title(f"Goodput (Payload) - {experiment_name}")
    ax.set_ylabel("Goodput [Mbps]")
    ax.set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(analysis_dir / "goodput.png", dpi=150)
    plt.close(fig)
    return "goodput.png"


def _generate_effective_goodput_plot(df, experiment_name, analysis_dir, add_phase_line_func):
    """Generate per-node Effective Goodput plot using Original Size (Pre-compression)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    plot_df = None

    # Priority 1: Calculate from original_size and comm_time_ms (Most accurate)
    if "original_size" in df.columns and "comm_time_ms" in df.columns:
        temp_df = df[(df["original_size"].notna()) & (df["comm_time_ms"] > 50)].copy()  # Min 50ms noise filter
        if not temp_df.empty:
            # Effective Goodput = Original Bits / Time
            temp_df["effective_goodput_mbps"] = (temp_df["original_size"] * 8 / 1e6) / (temp_df["comm_time_ms"] / 1000)
            plot_df = temp_df

    # Priority 2: Use pre-calculated effective_goodput_mbps (Fallback)
    if plot_df is None and "effective_goodput_mbps" in df.columns:
        temp_df = df[(df["effective_goodput_mbps"].notna()) & (df["effective_goodput_mbps"] > 0)].copy()
        if "comm_time_ms" in temp_df.columns:
            temp_df = temp_df[temp_df["comm_time_ms"] > 50]

        if not temp_df.empty:
            plot_df = temp_df

    if plot_df is None:
        plt.close(fig)
        return None

    if "is_ssp_interrupted" in plot_df.columns:
        plot_df = plot_df[~plot_df["is_ssp_interrupted"]]

    num_nodes = plot_df["node"].nunique()
    palette = NODE_PALETTE[:num_nodes] if num_nodes <= len(NODE_PALETTE) else "husl"

    sns.lineplot(
        data=plot_df,
        x="epoch",
        y="effective_goodput_mbps",
        hue="node",
        alpha=0.4,
        legend=False,
        palette=palette,
        linewidth=1.2,
        ax=ax,
    )
    sns.lineplot(
        data=plot_df,
        x="epoch",
        y="effective_goodput_mbps",
        color=COLORS["mean_line"],
        linewidth=2.5,
        label="Mean Effective Goodput",
        errorbar=None,
        ax=ax,
    )

    add_phase_line_func(ax, plot_df, "epoch")
    ax.set_title(f"Effective Goodput (Original Size / Time) - {experiment_name}")
    ax.set_ylabel("Effective Goodput [Mbps]")
    ax.set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(analysis_dir / "effective_goodput.png", dpi=150)
    plt.close(fig)
    return "effective_goodput.png"


def _generate_traffic_overhead_plot(df, experiment_name, analysis_dir, add_phase_line_func):
    """Generate Traffic Overhead Ratio (Physical / App Payload)."""
    if df.empty or "bytes_sent" not in df.columns or "app_bytes_sent" not in df.columns:
        return None

    plot_df = df[(df["bytes_sent"].notna()) & (df["app_bytes_sent"].notna())].copy()
    if plot_df.empty:
        return None

    if "is_ssp_interrupted" in plot_df.columns:
        plot_df = plot_df[~plot_df["is_ssp_interrupted"]]

    # Using 'app_bytes_sent' which is purely payload
    # Add small epsilon to avoid division by zero
    plot_df["overhead_ratio"] = plot_df["bytes_sent"] / (plot_df["app_bytes_sent"] + 1e-9)
    # Clip extreme values for visualization stability (e.g. initial empty packets)
    plot_df = plot_df[plot_df["app_bytes_sent"] > 100]  # Ignore irrelevant small packets

    if plot_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.lineplot(
        data=plot_df,
        x="epoch",
        y="overhead_ratio",
        color=COLORS["wasted_bar"],
        linewidth=2.5,
        label="Overhead Ratio (Physical/App)",
        errorbar=None,
        ax=ax,
    )

    ax.axhline(y=1.0, color="gray", linestyle="--", label="Ideal (1.0)")
    add_phase_line_func(ax, plot_df, "epoch")

    ax.set_title(f"Traffic Overhead Ratio - {experiment_name}")
    ax.set_ylabel("Overhead Ratio")
    ax.set_xlabel("Epoch")
    ax.legend()

    fig.tight_layout()
    fig.savefig(analysis_dir / "traffic_overhead.png", dpi=150)
    plt.close(fig)
    return "traffic_overhead.png"


# =============================================================================
# Cross-Experiment Comparison Functions
# =============================================================================


def _get_experiment_groups():
    """Group experiments by 'Experiment X:' or 'Experiment X (Name):' pattern."""
    if not RESULTS_DIR.exists():
        return {}

    groups = {}
    for d in RESULTS_DIR.iterdir():
        if d.is_dir() and not d.name.startswith("."):
            # Extract experiment group pattern like "Experiment 1:" or "Experiment 1 (RWP):"
            # Pattern matches: "Experiment N:" or "Experiment N (something):"
            match = re.match(r"^(Experiment \d+(?:\s*\([^)]+\))?:)", d.name)
            if match:
                group_name = match.group(1).strip(":")

                if group_name not in groups:
                    groups[group_name] = []
                groups[group_name].append(d.name)

    # Sort experiments within each group using custom sort key
    for group in groups:
        groups[group].sort(key=_get_experiment_sort_key)

    return groups


def _load_experiment_data(experiment_id):
    """Load metrics data for an experiment with WAFL-relative timestamp."""
    exp_dir = RESULTS_DIR / experiment_id
    df, _ = _load_metrics_and_resources(exp_dir)

    if df.empty:
        return df

    # Load metadata for WAFL phase start timestamp
    wafl_phase_start_relative = None
    metadata_path = exp_dir / "ctrl" / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
                wafl_phase_start_relative = metadata.get("wafl_phase_start_relative")
        except Exception:
            pass

    # Add WAFL-relative timestamp (time since WAFL phase start)
    if wafl_phase_start_relative is not None:
        df["wafl_relative_timestamp"] = df["timestamp"] - wafl_phase_start_relative
    else:
        # Fallback: use timestamp as-is
        df["wafl_relative_timestamp"] = df["timestamp"]

    return df


def _get_short_name(experiment_id):
    """Extract short name from experiment ID for legend.

    Removes the 'Experiment X:' or 'Experiment X (Name):' prefix and timestamp suffix.
    Examples:
    - "Experiment 1 (RWP): TCP-20251216T114459" → "TCP"
    - "Experiment 2: SUMO - Fast-20251217T031947" → "SUMO - Fast"
    """
    # First, remove timestamp suffix
    name = re.sub(r"-\d{8}T\d{6}$", "", experiment_id)

    # Remove "Experiment X:" or "Experiment X (Name):" prefix
    name = re.sub(r"^Experiment \d+(?:\s*\([^)]+\))?: ", "", name)

    return name.strip()


def _get_experiment_sort_key(experiment_id):
    """Generate sort key for experiment ordering.

    Priority order:
    1. Network condition: Excellent < Good < Fair < Poor
    2. Protocol: TCP < Fast
    3. Alphabetical fallback
    """
    short_name = _get_short_name(experiment_id).lower()

    # Network condition priority
    condition_order = {"excellent": 0, "good": 1, "fair": 2, "poor": 3}
    condition_key = 99
    for cond, order in condition_order.items():
        if cond in short_name:
            condition_key = order
            break

    # Protocol priority
    protocol_order = {"tcp": 0, "fast": 1}
    protocol_key = 99
    for proto, order in protocol_order.items():
        if proto in short_name:
            protocol_key = order
            break

    return (condition_key, protocol_key, short_name)


def _generate_accuracy_comparison(experiments_data, group_name, output_dir):
    """Generate accuracy comparison plot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    phase_switch_epoch = None

    for i, (exp_id, df) in enumerate(experiments_data.items()):
        if df.empty or "test_accuracy" not in df.columns:
            continue

        plot_df = df[df["test_accuracy"].notna()].copy()
        if "is_ssp_interrupted" in plot_df.columns:
            plot_df = plot_df[~plot_df["is_ssp_interrupted"]]

        if plot_df.empty:
            continue

        # Get phase switch epoch from first experiment
        if phase_switch_epoch is None and "phase" in df.columns:
            wafl_epochs = df[df["phase"] == "WAFL"]["epoch"]
            if not wafl_epochs.empty:
                phase_switch_epoch = wafl_epochs.min()

        acc_mean = plot_df.groupby("epoch")["test_accuracy"].mean().reset_index()

        color = NODE_PALETTE[i % len(NODE_PALETTE)]
        ax.plot(
            acc_mean["epoch"],
            acc_mean["test_accuracy"],
            color=color,
            linewidth=2,
            label=_get_short_name(exp_id),
        )

    # Add phase switch line
    if phase_switch_epoch is not None:
        ax.axvline(
            x=phase_switch_epoch,
            color=COLORS["phase_line"],
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
        )
        y_min, y_max = ax.get_ylim()
        ax.text(
            phase_switch_epoch,
            y_min + (y_max - y_min) * 0.05,
            " WAFL Start",
            color=COLORS["phase_line"],
            fontsize=10,
            fontweight="bold",
            va="bottom",
        )

    # Add target accuracy line
    ax.axhline(
        y=TARGET_ACCURACY,
        color=COLORS["target_line"],
        linestyle="--",
        linewidth=2,
        label=f"Target ({TARGET_ACCURACY:.0%})",
    )

    ax.set_title(f"Test Accuracy Comparison - {group_name}")
    ax.set_ylabel("Test Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_comparison.png", dpi=150)
    plt.close(fig)
    return "accuracy_comparison.png"


def _generate_loss_comparison(experiments_data, group_name, output_dir):
    """Generate loss comparison plot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    phase_switch_epoch = None

    for i, (exp_id, df) in enumerate(experiments_data.items()):
        if df.empty or "test_loss" not in df.columns:
            continue

        plot_df = df[df["test_loss"].notna()].copy()
        if "is_ssp_interrupted" in plot_df.columns:
            plot_df = plot_df[~plot_df["is_ssp_interrupted"]]

        if plot_df.empty:
            continue

        # Get phase switch epoch from first experiment
        if phase_switch_epoch is None and "phase" in df.columns:
            wafl_epochs = df[df["phase"] == "WAFL"]["epoch"]
            if not wafl_epochs.empty:
                phase_switch_epoch = wafl_epochs.min()

        loss_mean = plot_df.groupby("epoch")["test_loss"].mean().reset_index()

        color = NODE_PALETTE[i % len(NODE_PALETTE)]
        ax.plot(
            loss_mean["epoch"],
            loss_mean["test_loss"],
            color=color,
            linewidth=2,
            label=_get_short_name(exp_id),
        )

    # Add phase switch line
    if phase_switch_epoch is not None:
        ax.axvline(
            x=phase_switch_epoch,
            color=COLORS["phase_line"],
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
        )
        y_min, y_max = ax.get_ylim()
        ax.text(
            phase_switch_epoch,
            y_max * 0.95,
            " WAFL Start",
            color=COLORS["phase_line"],
            fontsize=10,
            fontweight="bold",
            va="top",
        )

    ax.set_title(f"Test Loss Comparison - {group_name}")
    ax.set_ylabel("Test Loss")
    ax.set_xlabel("Epoch")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "loss_comparison.png", dpi=150)
    plt.close(fig)
    return "loss_comparison.png"


def _generate_duration_comparison(experiments_data, group_name, output_dir):
    """Generate epoch duration comparison plot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    phase_switch_epoch = None

    for i, (exp_id, df) in enumerate(experiments_data.items()):
        if df.empty or "epoch_duration_ms" not in df.columns:
            continue

        epoch_dur = df[df["epoch_duration_ms"].notna()].copy()
        if epoch_dur.empty:
            continue

        # Get phase switch epoch from first experiment
        if phase_switch_epoch is None and "phase" in df.columns:
            wafl_epochs = df[df["phase"] == "WAFL"]["epoch"]
            if not wafl_epochs.empty:
                phase_switch_epoch = wafl_epochs.min()

        dur_mean = epoch_dur.groupby("epoch")["epoch_duration_ms"].mean().reset_index()
        dur_mean["duration_s"] = dur_mean["epoch_duration_ms"] / 1000

        color = NODE_PALETTE[i % len(NODE_PALETTE)]
        ax.plot(
            dur_mean["epoch"],
            dur_mean["duration_s"],
            color=color,
            linewidth=2,
            label=_get_short_name(exp_id),
        )

    # Add phase switch line
    if phase_switch_epoch is not None:
        ax.axvline(
            x=phase_switch_epoch,
            color=COLORS["phase_line"],
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
        )
        y_min, y_max = ax.get_ylim()
        ax.text(
            phase_switch_epoch,
            y_max * 0.95,
            " WAFL Start",
            color=COLORS["phase_line"],
            fontsize=10,
            fontweight="bold",
            va="top",
        )

    ax.set_title(f"Epoch Duration Comparison - {group_name}")
    ax.set_ylabel("Duration [sec]")
    ax.set_xlabel("Epoch")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "epoch_duration_comparison.png", dpi=150)
    plt.close(fig)
    return "epoch_duration_comparison.png"


def _generate_time_to_accuracy_comparison(experiments_data, group_name, output_dir):
    """Generate time-to-accuracy bar chart comparison (WAFL phase only, from WAFL start)."""
    tta_data = []

    for exp_id, df in experiments_data.items():
        if df.empty or "test_accuracy" not in df.columns or "wafl_relative_timestamp" not in df.columns:
            continue

        # Only use WAFL phase data (WAFL phase start = 0)
        wafl_df = df[(df["phase"] == "WAFL") & (df["test_accuracy"].notna())].copy()
        if "is_ssp_interrupted" in wafl_df.columns:
            wafl_df = wafl_df[~wafl_df["is_ssp_interrupted"]]

        if wafl_df.empty:
            continue

        epoch_stats = wafl_df.groupby("epoch").agg({"test_accuracy": "mean", "wafl_relative_timestamp": "max"}).reset_index()
        epoch_stats.columns = ["epoch", "mean", "timestamp"]

        reached = epoch_stats[epoch_stats["mean"] >= TARGET_ACCURACY]
        if not reached.empty:
            tta = reached["timestamp"].iloc[0]
        else:
            tta = epoch_stats["timestamp"].max()

        tta_data.append(
            {
                "experiment": _get_short_name(exp_id),
                "time_to_accuracy": tta,
                "reached": not reached.empty,
            }
        )

    if not tta_data:
        return None

    tta_df = pd.DataFrame(tta_data)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [COLORS["goodput"] if r else COLORS["loss_fill"] for r in tta_df["reached"]]
    bars = ax.bar(tta_df["experiment"], tta_df["time_to_accuracy"], color=colors, alpha=0.8)

    for bar, reached in zip(bars, tta_df["reached"]):
        height = bar.get_height()
        suffix = "" if reached else " (not reached)"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.1f}s{suffix}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title(f"Time to {TARGET_ACCURACY * 100:.0%} Accuracy (from WAFL Start) - {group_name}")
    ax.set_ylabel("Time from WAFL Start [sec]")
    ax.set_xlabel("Experiment")
    # Explicitly set tick labels to ensure they are correct when rotated
    ax.set_xticks(range(len(tta_df["experiment"])))
    ax.set_xticklabels(tta_df["experiment"])
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")

    fig.tight_layout()
    fig.savefig(output_dir / "time_to_accuracy_comparison.png", dpi=150)
    plt.close(fig)
    return "time_to_accuracy_comparison.png"


def _generate_survival_rate_comparison(experiments_data, group_name, output_dir):
    """Generate survival rate comparison plot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    has_data = False

    for i, (exp_id, df) in enumerate(experiments_data.items()):
        if df.empty or "survival_rate" not in df.columns:
            continue

        # User Request: Show only after WAFL Start
        wafl_df = df[df["phase"] == "WAFL"].copy() if "phase" in df.columns else df
        plot_df = wafl_df[wafl_df["survival_rate"].notna()].copy()
        if plot_df.empty or (plot_df["survival_rate"] == 1.0).all():
            continue

        has_data = True
        sr_mean = plot_df.groupby("epoch")["survival_rate"].mean().reset_index()

        color = NODE_PALETTE[i % len(NODE_PALETTE)]
        ax.plot(
            sr_mean["epoch"],
            sr_mean["survival_rate"],
            color=color,
            linewidth=2,
            label=_get_short_name(exp_id),
        )

    if not has_data:
        plt.close(fig)
        return None

    ax.set_title(f"Survival Rate Comparison (UDP/FEC) - {group_name}")
    ax.set_ylabel("Survival Rate")
    ax.set_xlabel("Epoch")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "survival_rate_comparison.png", dpi=150)
    plt.close(fig)
    return "survival_rate_comparison.png"


def _generate_throughput_comparison(experiments_data, group_name, output_dir):
    """Generate throughput comparison plot (WAFL phase only)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    has_data = False

    for i, (exp_id, df) in enumerate(experiments_data.items()):
        if df.empty or "bytes_received" not in df.columns or "comm_time_ms" not in df.columns:
            continue

        # Filter to WAFL phase only
        wafl_df = df[df["phase"] == "WAFL"].copy() if "phase" in df.columns else df
        plot_df = wafl_df[(wafl_df["bytes_received"].notna()) & (wafl_df["comm_time_ms"].notna()) & (wafl_df["comm_time_ms"] > 0)].copy()
        if plot_df.empty or plot_df["bytes_received"].sum() == 0:
            continue

        has_data = True
        plot_df["throughput_mbps"] = (plot_df["bytes_received"] * 8 / 1e6) / (plot_df["comm_time_ms"] / 1000)
        tp_mean = plot_df.groupby("epoch")["throughput_mbps"].mean().reset_index()

        color = NODE_PALETTE[i % len(NODE_PALETTE)]
        ax.plot(
            tp_mean["epoch"],
            tp_mean["throughput_mbps"],
            color=color,
            linewidth=2,
            label=_get_short_name(exp_id),
        )

    if not has_data:
        plt.close(fig)
        return None

    ax.set_title(f"Throughput Comparison - {group_name}")
    ax.set_ylabel("Throughput [Mbps]")
    ax.set_xlabel("Epoch")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "throughput_comparison.png", dpi=150)
    plt.close(fig)
    return "throughput_comparison.png"


def _generate_accuracy_time_comparison(experiments_data, group_name, output_dir):
    """Generate test accuracy vs time comparison plot (WAFL phase, aligned at WAFL start)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (exp_id, df) in enumerate(experiments_data.items()):
        if df.empty or "test_accuracy" not in df.columns or "wafl_relative_timestamp" not in df.columns:
            continue

        # Only use WAFL phase data (WAFL phase start = 0)
        wafl_df = df[(df["phase"] == "WAFL") & (df["test_accuracy"].notna())].copy()
        if "is_ssp_interrupted" in wafl_df.columns:
            wafl_df = wafl_df[~wafl_df["is_ssp_interrupted"]]

        if wafl_df.empty:
            continue

        # Calculate mean accuracy and time per epoch (using WAFL-relative timestamp)
        epoch_stats = wafl_df.groupby("epoch").agg({"wafl_relative_timestamp": "mean", "test_accuracy": "mean"}).reset_index()

        color = NODE_PALETTE[i % len(NODE_PALETTE)]
        ax.plot(
            epoch_stats["wafl_relative_timestamp"],
            epoch_stats["test_accuracy"],
            color=color,
            linewidth=2,
            label=_get_short_name(exp_id),
        )

    # Add target accuracy line
    ax.axhline(
        y=TARGET_ACCURACY,
        color=COLORS["target_line"],
        linestyle="--",
        linewidth=2,
        label=f"Target ({TARGET_ACCURACY:.0%})",
    )

    ax.set_title(f"Test Accuracy vs Time (from WAFL Start) - {group_name}")
    ax.set_ylabel("Test Accuracy")
    ax.set_xlabel("Time from WAFL Start [sec]")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_time_comparison.png", dpi=150)
    plt.close(fig)
    return "accuracy_vs_time_comparison.png"


def _generate_cumulative_sent_data_comparison(experiments_data, group_name, output_dir):
    """Generate cumulative sent data comparison bar chart."""
    cumulative_data = []

    for exp_id, df in experiments_data.items():
        if df.empty or "bytes_sent" not in df.columns:
            continue

        traffic_data = df[df["bytes_sent"].notna()].copy()
        if traffic_data.empty or traffic_data["bytes_sent"].sum() == 0:
            continue

        # Calculate total bytes sent across all epochs
        total_bytes = traffic_data["bytes_sent"].sum()
        total_mb = total_bytes / (1024 * 1024)

        cumulative_data.append(
            {
                "experiment": _get_short_name(exp_id),
                "cumulative_mb": total_mb,
            }
        )

    if not cumulative_data:
        return None

    cumulative_df = pd.DataFrame(cumulative_data)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(
        cumulative_df["experiment"],
        cumulative_df["cumulative_mb"],
        color=COLORS["traffic"],
        alpha=0.8,
        edgecolor=COLORS["traffic"],
    )

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.1f} MB",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title(f"Total Sent Data Comparison - {group_name}")
    ax.set_ylabel("Cumulative Sent Data [MB]")
    ax.set_xlabel("Experiment")

    # Explicitly set ticks and labels
    ax.set_xticks(range(len(cumulative_df["experiment"])))
    ax.set_xticklabels(cumulative_df["experiment"])
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")

    fig.tight_layout()
    fig.savefig(output_dir / "cumulative_sent_data_comparison.png", dpi=150)
    plt.close(fig)
    return "cumulative_sent_data_comparison.png"


def _generate_idle_time_comparison(experiments_data, group_name, output_dir):
    """Generate idle time ratio comparison plot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    has_data = False
    phase_switch_epoch = None

    for i, (exp_id, df) in enumerate(experiments_data.items()):
        if df.empty or "epoch_duration_ms" not in df.columns:
            continue

        epoch_dur = df[df["epoch_duration_ms"].notna()].copy()
        if epoch_dur.empty:
            continue

        max_dur_per_epoch = epoch_dur.groupby("epoch")["epoch_duration_ms"].max().reset_index()
        max_dur_per_epoch.columns = ["epoch", "max_duration_ms"]
        epoch_dur = epoch_dur.merge(max_dur_per_epoch, on="epoch")
        epoch_dur["idle_time_ms"] = epoch_dur["max_duration_ms"] - epoch_dur["epoch_duration_ms"]
        epoch_dur["idle_time_ratio"] = epoch_dur["idle_time_ms"] / epoch_dur["max_duration_ms"]
        idle_agg = epoch_dur.groupby("epoch").agg({"idle_time_ratio": "mean"}).reset_index()

        color = NODE_PALETTE[i % len(NODE_PALETTE)]
        ax.plot(
            idle_agg["epoch"],
            idle_agg["idle_time_ratio"],
            color=color,
            linewidth=2,
            label=_get_short_name(exp_id),
        )
        has_data = True

        # Get phase switch epoch from first experiment (or any)
        if phase_switch_epoch is None and "phase" in df.columns:
            wafl_epochs = df[df["phase"] == "WAFL"]["epoch"]
            if not wafl_epochs.empty:
                phase_switch_epoch = wafl_epochs.min()

    if not has_data:
        plt.close(fig)
        return None

    # Add phase switch line
    if phase_switch_epoch is not None:
        ax.axvline(
            x=phase_switch_epoch,
            color=COLORS["phase_line"],
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
        )
        y_min, y_max = ax.get_ylim()
        ax.text(
            phase_switch_epoch,
            y_max * 0.95,
            " WAFL Start",
            color=COLORS["phase_line"],
            fontsize=10,
            fontweight="bold",
            va="top",
        )

    ax.set_title(f"Idle Time Ratio Comparison - {group_name}")
    ax.set_ylabel("Idle Time Ratio")
    ax.set_xlabel("Epoch")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "idle_time_comparison.png", dpi=150)
    plt.close(fig)
    return "idle_time_comparison.png"


def _generate_wasted_computation_comparison(experiments_data, group_name, output_dir):
    """Generate wasted computation comparison plot (SSP mode)."""
    wasted_data = []

    for exp_id, df in experiments_data.items():
        if df.empty or "wasted_ms" not in df.columns:
            continue

        wafl_df = df[df["phase"] == "WAFL"].copy()
        ssp_data = wafl_df[(wafl_df["wasted_ms"].notna()) & (wafl_df["wasted_ms"] > 0)]
        if ssp_data.empty:
            continue

        total_wasted_s = ssp_data["wasted_ms"].sum() / 1000
        wasted_data.append({"experiment": _get_short_name(exp_id), "wasted_s": total_wasted_s})

    if not wasted_data:
        return None

    wasted_df = pd.DataFrame(wasted_data)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(
        wasted_df["experiment"],
        wasted_df["wasted_s"],
        color=COLORS["wasted_bar"],
        alpha=0.8,
    )

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.1f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title(f"Total Wasted Computation (SSP) - {group_name}")
    ax.set_ylabel("Wasted Time [sec]")
    ax.set_xlabel("Experiment")

    # Explicitly set ticks and labels
    ax.set_xticks(range(len(wasted_df["experiment"])))
    ax.set_xticklabels(wasted_df["experiment"])
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")

    fig.tight_layout()
    fig.savefig(output_dir / "wasted_computation_comparison.png", dpi=150)
    plt.close(fig)
    return "wasted_computation_comparison.png"


def _generate_traffic_volume_comparison(experiments_data, group_name, output_dir):
    """Generate traffic volume per epoch comparison plot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    has_data = False

    for i, (exp_id, df) in enumerate(experiments_data.items()):
        if df.empty or "bytes_sent" not in df.columns:
            continue

        wafl_df = df[df["phase"] == "WAFL"].copy()
        # Use simple bytes_sent for total volume view
        traffic_data = wafl_df[wafl_df["bytes_sent"].notna()]
        if traffic_data.empty or traffic_data["bytes_sent"].sum() == 0:
            continue

        traffic_agg = traffic_data.groupby("epoch").agg({"bytes_sent": "sum"}).reset_index()
        traffic_agg["sent_mb"] = traffic_agg["bytes_sent"] / (1024 * 1024)

        color = NODE_PALETTE[i % len(NODE_PALETTE)]
        ax.plot(
            traffic_agg["epoch"],
            traffic_agg["sent_mb"],
            color=color,
            linewidth=2,
            label=_get_short_name(exp_id),
        )
        has_data = True

    if not has_data:
        plt.close(fig)
        return None

    ax.set_title(f"Traffic Volume per Epoch (WAFL Phase) - {group_name}")
    ax.set_ylabel("Sent Data [MB]")
    ax.set_xlabel("Epoch")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "traffic_volume_comparison.png", dpi=150)
    plt.close(fig)
    return "traffic_volume_comparison.png"


def _generate_compute_comm_comparison(experiments_data, group_name, output_dir):
    """Generate compute vs communication time comparison plot."""
    breakdown_data = []

    for exp_id, df in experiments_data.items():
        if df.empty or "compute_time_ms" not in df.columns or "comm_time_ms" not in df.columns:
            continue

        wafl_df = df[df["phase"] == "WAFL"].copy()
        valid = wafl_df[(wafl_df["compute_time_ms"].notna()) & (wafl_df["comm_time_ms"].notna())]
        if valid.empty:
            continue

        avg_compute = valid["compute_time_ms"].mean() / 1000
        avg_comm = valid["comm_time_ms"].mean() / 1000
        breakdown_data.append(
            {
                "experiment": _get_short_name(exp_id),
                "compute_s": avg_compute,
                "comm_s": avg_comm,
            }
        )

    if not breakdown_data:
        return None

    breakdown_df = pd.DataFrame(breakdown_data)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(breakdown_df))
    width = 0.35

    ax.bar(
        [i - width / 2 for i in x],
        breakdown_df["compute_s"],
        width,
        label="Compute",
        color=COLORS["goodput"],
        alpha=0.8,
    )
    ax.bar(
        [i + width / 2 for i in x],
        breakdown_df["comm_s"],
        width,
        label="Communication",
        color=COLORS["traffic"],
        alpha=0.8,
    )

    ax.set_ylabel("Average Time [sec]")
    ax.set_xlabel("Experiment")
    ax.set_title(f"Compute vs Communication Time (WAFL Phase) - {group_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(breakdown_df["experiment"])
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "compute_comm_comparison.png", dpi=150)
    plt.close(fig)
    return "compute_comm_comparison.png"


def _generate_goodput_comparison(experiments_data, group_name, output_dir):
    """Generate goodput comparison plot (WAFL phase only) using app_bytes_received."""
    fig, ax = plt.subplots(figsize=(12, 6))
    has_data = False

    for i, (exp_id, df) in enumerate(experiments_data.items()):
        if df.empty or "comm_time_ms" not in df.columns:
            continue

        plot_df = df[df["comm_time_ms"] > 50].copy()  # Min 50ms filter
        if plot_df.empty:
            continue

        target_col = "app_bytes_received"
        if target_col not in df.columns:
            target_col = "bytes_received"

        wafl_df = plot_df[plot_df["phase"] == "WAFL"].copy()
        goodput_data = wafl_df[wafl_df[target_col].notna()].copy()

        if goodput_data.empty or goodput_data[target_col].sum() == 0:
            continue

        # Goodput = Payload Bits / Time
        goodput_data["goodput_mbps"] = (goodput_data[target_col] * 8 / 1e6) / (goodput_data["comm_time_ms"] / 1000)
        goodput_agg = goodput_data.groupby("epoch")["goodput_mbps"].mean().reset_index()

        color = NODE_PALETTE[i % len(NODE_PALETTE)]
        ax.plot(
            goodput_agg["epoch"],
            goodput_agg["goodput_mbps"],
            color=color,
            linewidth=2,
            label=_get_short_name(exp_id),
        )
        has_data = True

    if not has_data:
        plt.close(fig)
        return None

    ax.set_title(f"Goodput Comparison (Payload) - {group_name}")
    ax.set_ylabel("Goodput [Mbps]")
    ax.set_xlabel("Epoch")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "goodput_comparison.png", dpi=150)
    plt.close(fig)
    return "goodput_comparison.png"


def _generate_effective_goodput_comparison(experiments_data, group_name, output_dir):
    """Generate Effective Goodput comparison plot (Original Size / Time)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    has_data = False

    for i, (exp_id, df) in enumerate(experiments_data.items()):
        if df.empty or "effective_goodput_mbps" not in df.columns:
            continue

        wafl_df = df[df["phase"] == "WAFL"].copy()
        plot_df = wafl_df[(wafl_df["effective_goodput_mbps"].notna())].copy()

        if plot_df.empty:
            continue

        agg = plot_df.groupby("epoch")["effective_goodput_mbps"].mean().reset_index()

        color = NODE_PALETTE[i % len(NODE_PALETTE)]
        ax.plot(
            agg["epoch"],
            agg["effective_goodput_mbps"],
            color=color,
            linewidth=2,
            label=_get_short_name(exp_id),
        )
        has_data = True

    if not has_data:
        plt.close(fig)
        return None

    ax.set_title(f"Effective Goodput Comparison (Original / Time) - {group_name}")
    ax.set_ylabel("Effective Goodput [Mbps]")
    ax.set_xlabel("Epoch")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "effective_goodput_comparison.png", dpi=150)
    plt.close(fig)
    return "effective_goodput_comparison.png"


def _generate_traffic_overhead_comparison(experiments_data, group_name, output_dir):
    """Generate Traffic Overhead Ratio comparison plot using app_bytes_sent."""
    overhead_data = []

    for exp_id, df in experiments_data.items():
        if df.empty or "bytes_sent" not in df.columns:
            continue

        target_app = "app_bytes_sent"
        if target_app not in df.columns:
            # Fallback for old logs? No, we enforced new logs.
            continue

        wafl_df = df[df["phase"] == "WAFL"].copy()
        valid = wafl_df[(wafl_df["bytes_sent"].notna()) & (wafl_df[target_app].notna())].copy()

        if not valid.empty and valid["bytes_sent"].sum() > 0:
            agg = valid.groupby("epoch").agg({"bytes_sent": "sum", "app_bytes_sent": "sum"}).reset_index()
            # Calculate sum of all epochs
            total_phy = agg["bytes_sent"].sum()
            total_app = agg["app_bytes_sent"].sum()

            ratio = total_phy / total_app if total_app > 0 else 0
            overhead_data.append({"experiment": _get_short_name(exp_id), "overhead_ratio": ratio})

    if not overhead_data:
        return None

    overhead_df = pd.DataFrame(overhead_data)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(overhead_df["experiment"], overhead_df["overhead_ratio"], color=COLORS["phase_line"], alpha=0.8)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}x", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title(f"Traffic Overhead Ratio Comparison (Physical / App) - {group_name}")
    ax.set_ylabel("Overhead Ratio")
    ax.set_xlabel("Experiment")
    ax.axhline(y=1.0, color="gray", linestyle="--", label="Ideal (1.0)")

    ax.set_xticks(range(len(overhead_df["experiment"])))
    ax.set_xticklabels(overhead_df["experiment"])
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "traffic_overhead_comparison.png", dpi=150)
    plt.close(fig)
    return "traffic_overhead_comparison.png"


def _generate_throughput_overhead_comparison(experiments_data, group_name, output_dir):
    """Generate Throughput Overhead comparison (Grouped Bar)."""
    tp_data = []

    for exp_id, df in experiments_data.items():
        if df.empty or "comm_time_ms" not in df.columns:
            continue

        wafl_df = df[df["phase"] == "WAFL"].copy()
        valid = wafl_df[wafl_df["comm_time_ms"] > 50].copy()  # Min 50ms filter

        if not valid.empty:
            # Per node avg
            valid["phy_mbps"] = (valid["bytes_sent"] * 8 / 1e6) / (valid["comm_time_ms"] / 1000)
            valid["app_mbps"] = (valid["app_bytes_sent"] * 8 / 1e6) / (valid["comm_time_ms"] / 1000)

            avg_phy = valid["phy_mbps"].mean()
            avg_app = valid["app_mbps"].mean()

            tp_data.append({"experiment": _get_short_name(exp_id), "Physical Throughput": avg_phy, "Goodput": avg_app})

    if not tp_data:
        return None

    tp_df = pd.DataFrame(tp_data)

    # Melting for grouped bar plot
    melted = tp_df.melt(id_vars="experiment", var_name="Metric", value_name="Mbps")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=melted, x="experiment", y="Mbps", hue="Metric", palette=[COLORS["traffic"], COLORS["goodput"]], ax=ax)

    ax.set_title(f"Throughput Comparison (Physical vs Goodput) - {group_name}")
    # Fix: Use tick_params for rotation instead of set_xticklabels(get_xticklabels())
    ax.tick_params(axis="x", rotation=45)
    # Align labels to right
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "throughput_overhead_comparison.png", dpi=150)
    plt.close(fig)
    return "throughput_overhead_comparison.png"


def _generate_accuracy_dist_comparison(experiments_data, group_name, output_dir):
    """Generate Accuracy Distribution Comparison (Box Plot)."""
    dist_data = []

    for exp_id, df in experiments_data.items():
        if df.empty or "test_accuracy" not in df.columns:
            continue
        wafl_df = df[df["phase"] == "WAFL"].dropna(subset=["test_accuracy"])
        if wafl_df.empty:
            continue

        # We want distribution of FINAL accuracy per node? Or distribution across all epochs?
        # Typically "Per-node accuracy" implies how well each node did.
        # Let's take the mean accuracy of each node over the last 10% of epochs or just all epochs?
        # Or simply the distribution of test_accuracy of all nodes in all epochs to show stability?
        # Let's show distribution of MEAN accuracy per node (Node Fairness).

        node_means = wafl_df.groupby("node")["test_accuracy"].mean().reset_index()
        node_means["experiment"] = _get_short_name(exp_id)
        dist_data.append(node_means)

    if not dist_data:
        return None

    full_df = pd.concat(dist_data)

    fig, ax = plt.subplots(figsize=(12, 6))
    # Fix: Slice palette to match number of experiments to avoid warning
    palette = NODE_PALETTE[: len(full_df["experiment"].unique())]
    sns.boxplot(data=full_df, x="experiment", y="test_accuracy", hue="experiment", legend=False, palette=palette, ax=ax)

    ax.set_title(f"Node-wise Mean Accuracy Distribution - {group_name}")
    ax.set_ylabel("Mean Test Accuracy")
    # Fix: Use tick_params
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")

    fig.tight_layout()
    plt.savefig(output_dir / "accuracy_dist_comparison.png", dpi=150)
    plt.close(fig)
    return "accuracy_dist_comparison.png"


def _generate_fec_overhead_comparison(experiments_data, group_name, output_dir):
    """Generate FEC Overhead Comparison."""
    fec_data = []

    for exp_id, df in experiments_data.items():
        if df.empty:
            continue
        wafl_df = df[df["phase"] == "WAFL"]

        enc = wafl_df["fec_encode_time_ms"].mean() if "fec_encode_time_ms" in wafl_df.columns else 0
        dec = wafl_df["fec_decode_time_ms"].mean() if "fec_decode_time_ms" in wafl_df.columns else 0

        if enc > 0 or dec > 0:
            fec_data.append({"experiment": _get_short_name(exp_id), "Encode": enc, "Decode": dec})

    if not fec_data:
        return None

    fec_df = pd.DataFrame(fec_data)
    fec_df = fec_df.set_index("experiment")

    fig, ax = plt.subplots(figsize=(12, 6))
    fec_df.plot(kind="bar", stacked=True, color=[COLORS["bar_primary"], COLORS["bar_secondary"]], ax=ax)

    ax.set_title(f"FEC Processing Overhead Comparison - {group_name}")
    ax.set_ylabel("Time [ms]")
    # Fix: Use tick_params
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")

    fig.tight_layout()
    plt.savefig(output_dir / "fec_overhead_comparison.png", dpi=150)
    plt.close(fig)
    return "fec_overhead_comparison.png"


def _generate_compression_comparison(experiments_data, group_name, output_dir):
    """Generate Compression Stats Comparison."""
    comp_data = []

    for exp_id, df in experiments_data.items():
        if df.empty:
            continue
        wafl_df = df[df["phase"] == "WAFL"]

        ratio = wafl_df["compression_ratio"].mean() if "compression_ratio" in wafl_df.columns else 1.0
        time = wafl_df["compression_time_ms"].mean() if "compression_time_ms" in wafl_df.columns else 0

        comp_data.append({"experiment": _get_short_name(exp_id), "Ratio": ratio, "Time (ms)": time})

    if not comp_data:
        return None

    comp_df = pd.DataFrame(comp_data)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Ratio as Bar
    sns.barplot(data=comp_df, x="experiment", y="Ratio", color=COLORS["goodput"], alpha=0.6, ax=ax1)
    ax1.set_ylabel("Compression Ratio (Lower is better)", color=COLORS["goodput"])
    ax1.set_ylim(0, 1.2)

    # Plot Time as Line on secondary axis
    ax2 = ax1.twinx()
    sns.lineplot(data=comp_df, x="experiment", y="Time (ms)", color=COLORS["wasted_bar"], marker="o", linewidth=2, ax=ax2)
    ax2.set_ylabel("Compression Time [ms]", color=COLORS["wasted_bar"])

    ax1.set_title(f"Compression Statistics Comparison - {group_name}")
    # Fix: Use tick_params
    ax1.tick_params(axis="x", rotation=45)
    plt.setp(ax1.get_xticklabels(), ha="right")

    fig.tight_layout()
    plt.savefig(output_dir / "compression_comparison.png", dpi=150)
    plt.close(fig)
    return "compression_comparison.png"


def _generate_protocol_distribution_comparison(experiments_data, group_name, output_dir):
    """Generate Protocol Distribution Comparison (TCP vs UDP)."""
    dist_data = []

    for exp_id, df in experiments_data.items():
        if df.empty:
            continue

        wafl_df = df[df["phase"] == "WAFL"]
        tcp = wafl_df["protocol_tcp_count"].sum() if "protocol_tcp_count" in wafl_df.columns else 0
        udp = wafl_df["protocol_udp_count"].sum() if "protocol_udp_count" in wafl_df.columns else 0
        # RUDP removed

        total = tcp + udp
        if total > 0:
            dist_data.append({"experiment": _get_short_name(exp_id), "TCP": tcp / total * 100, "UDP": udp / total * 100})

    if not dist_data:
        return None

    dist_df = pd.DataFrame(dist_data).set_index("experiment")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Custom colors: TCP(primary), UDP(goodput)
    colors = [COLORS["bar_primary"], COLORS["goodput"]]

    dist_df.plot(kind="bar", stacked=True, color=colors, ax=ax)

    ax.set_title(f"Protocol Distribution Comparison - {group_name}")
    ax.set_ylabel("Usage Percentage [%]")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    fig.tight_layout()
    plt.savefig(output_dir / "protocol_distribution_comparison.png", dpi=150)
    plt.close(fig)
    return "protocol_distribution_comparison.png"


def _generate_comparison_report(experiments_data, group_name, group_dir):
    """Generate Markdown comparison report for a group of experiments."""
    report_path = group_dir / "report.md"

    lines = []
    lines.append(f"# Comparison Report: {group_name}")
    lines.append("")

    # Experiments list
    lines.append("## Experiments")
    lines.append("")
    for exp_id in experiments_data.keys():
        lines.append(f"- {_get_short_name(exp_id)}")
    lines.append("")

    # Calculate metrics for each experiment
    lines.append("## Comparison Table")
    lines.append("")

    # Build detailed comparison table - Accuracy & Convergence
    lines.append("### Accuracy & Convergence")
    lines.append("")
    # Determine target threshold to display in header
    target_str = "90%" if "Experiment 4" in group_name else "80%"
    headers1 = ["Experiment", "Final Acc", "Max Acc", f"Time to {target_str}", "Total Epochs", "WAFL Epochs"]
    lines.append("| " + " | ".join(headers1) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers1)) + " |")

    all_metrics = {}
    for exp_id, df in experiments_data.items():
        metrics = _calculate_metrics_summary(df, group_name=group_name)
        all_metrics[exp_id] = metrics
        short_name = _get_short_name(exp_id)
        row = [short_name]
        row.append(f"{metrics.get('final_accuracy', 0) * 100:.2f}%" if "final_accuracy" in metrics else "N/A")
        row.append(f"{metrics.get('max_accuracy', 0) * 100:.2f}%" if "max_accuracy" in metrics else "N/A")
        row.append(f"{metrics.get('time_to_target', 0):.2f}s" if "time_to_target" in metrics else "N/A")
        row.append(f"{metrics.get('total_epochs', 0)}" if "total_epochs" in metrics else "N/A")
        row.append(f"{metrics.get('wafl_epochs', 0)}" if "wafl_epochs" in metrics else "N/A")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Build detailed comparison table - Performance
    lines.append("### Performance & Timing")
    lines.append("")
    headers2 = ["Experiment", "Avg Epoch (s)", "Total Time (s)", "Compute Time", "Comm Time"]
    lines.append("| " + " | ".join(headers2) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers2)) + " |")

    for exp_id, df in experiments_data.items():
        metrics = all_metrics[exp_id]
        short_name = _get_short_name(exp_id)
        row = [short_name]
        row.append(f"{metrics.get('avg_epoch_duration_s', 0):.2f}" if "avg_epoch_duration_s" in metrics else "N/A")
        row.append(f"{metrics.get('total_time_s', 0):.1f}" if "total_time_s" in metrics else "N/A")

        # Calculate avg compute and comm time
        wafl_df = df[df["phase"] == "WAFL"] if "phase" in df.columns else df
        avg_compute = wafl_df["compute_time_ms"].mean() / 1000 if "compute_time_ms" in wafl_df.columns else None
        avg_comm = wafl_df["comm_time_ms"].mean() / 1000 if "comm_time_ms" in wafl_df.columns else None
        row.append(f"{avg_compute:.2f}s" if avg_compute else "N/A")
        row.append(f"{avg_comm:.2f}s" if avg_comm else "N/A")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Build detailed comparison table - Communication
    lines.append("### Communication & Network")
    lines.append("")
    headers3 = ["Experiment", "Total Traffic (MB)", "Avg Goodput (Mbps)", "Avg Phy Throughput (Mbps)", "Overhead Ratio", "Avg Survival"]
    lines.append("| " + " | ".join(headers3) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers3)) + " |")

    for exp_id in experiments_data.keys():
        metrics = all_metrics[exp_id]
        short_name = _get_short_name(exp_id)
        row = [short_name]
        row.append(f"{metrics.get('total_traffic_mb', 0):.2f}" if "total_traffic_mb" in metrics else "N/A")
        row.append(f"{metrics.get('avg_goodput_mbps', 0):.2f}" if "avg_goodput_mbps" in metrics else "N/A")
        row.append(f"{metrics.get('avg_phy_throughput_mbps', 0):.2f}" if "avg_phy_throughput_mbps" in metrics else "N/A")
        row.append(f"{metrics.get('traffic_overhead_ratio', 0):.2f}x" if "traffic_overhead_ratio" in metrics else "N/A")
        row.append(f"{metrics.get('avg_survival_rate', 0) * 100:.1f}%" if "avg_survival_rate" in metrics else "N/A")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")

    # Generated Graphs Section
    lines.append("## Comparison Graphs")
    lines.append("")

    graph_descriptions = {
        "accuracy_comparison.png": "Test Accuracy Comparison",
        "loss_comparison.png": "Test Loss Comparison",
        "accuracy_dist_comparison.png": "Accuracy Distribution (Node Fairness)",
        "accuracy_vs_time_comparison.png": "Accuracy vs Time Comparison",
        "time_to_accuracy_comparison.png": "Time to Target Accuracy Comparison",
        "epoch_duration_comparison.png": "Epoch Duration Comparison",
        "compute_comm_comparison.png": "Compute vs Communication Time",
        "idle_time_comparison.png": "Idle Time Comparison",
        "survival_rate_comparison.png": "Survival Rate Comparison",
        "goodput_comparison.png": "Goodput Comparison",
        "effective_goodput_comparison.png": "Effective Goodput (Original/Time) Comparison",
        "throughput_comparison.png": "Throughput Comparison",
        "traffic_volume_comparison.png": "Traffic Volume Comparison",
        "cumulative_sent_data_comparison.png": "Cumulative Sent Data Comparison",
        "traffic_overhead_comparison.png": "Traffic Overhead Ratio Comparison",
        "throughput_overhead_comparison.png": "Throughput Overhead Comparison",
        "wasted_computation_comparison.png": "Wasted Computation (SSP)",
        "fec_overhead_comparison.png": "FEC Processing Overhead Comparison",
        "compression_comparison.png": "Compression Stats Comparison",
        "protocol_distribution_comparison.png": "Protocol Distribution Comparison",
    }

    for graph_file, description in graph_descriptions.items():
        if (group_dir / graph_file).exists():
            lines.append(f"### {description}")
            lines.append("")
            lines.append(f"![{description}]({graph_file})")
            lines.append("")

    # Write report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_path


def compare_experiments():
    """Compare experiments grouped by 'Experiment X:' pattern."""
    groups = _get_experiment_groups()

    if not groups:
        print("❌ No experiment groups found matching 'Experiment X:' pattern.")
        return

    print(f"📊 Found {len(groups)} experiment groups:")
    for group_name, experiments in groups.items():
        print(f"   {group_name}: {len(experiments)} experiments")

    comparison_dir = RESULTS_DIR / ".comparison"
    comparison_dir.mkdir(exist_ok=True)

    for group_name, experiment_ids in groups.items():
        print(f"\n{'=' * 60}")
        print(f"Comparing: {group_name} ({len(experiment_ids)} experiments)")
        print(f"{'=' * 60}")

        group_dir = comparison_dir / group_name.replace(" ", "_").replace(":", "")
        group_dir.mkdir(exist_ok=True)

        experiments_data = {}
        for exp_id in experiment_ids:
            print(f"  📂 Loading: {_get_short_name(exp_id)}")
            df = _load_experiment_data(exp_id)
            if not df.empty:
                experiments_data[exp_id] = df

        if len(experiments_data) < 2:
            print("  ⚠️ Need at least 2 experiments for comparison, skipping.")
            continue

        plots = [
            ("accuracy_comparison.png", _generate_accuracy_comparison),
            ("loss_comparison.png", _generate_loss_comparison),
            ("accuracy_dist_comparison.png", _generate_accuracy_dist_comparison),
            ("epoch_duration_comparison.png", _generate_duration_comparison),
            ("accuracy_time.png", _generate_accuracy_time_comparison),
            ("survival_rate_comparison.png", _generate_survival_rate_comparison),
            ("traffic_overhead_comparison.png", _generate_traffic_overhead_comparison),
            ("throughput_overhead_comparison.png", _generate_throughput_overhead_comparison),
            ("goodput_comparison.png", _generate_goodput_comparison),
            ("traffic_volume_comparison.png", _generate_traffic_volume_comparison),
            ("compute_comm_comparison.png", _generate_compute_comm_comparison),
            ("wasted_computation_comparison.png", _generate_wasted_computation_comparison),
            ("fec_overhead_comparison.png", _generate_fec_overhead_comparison),
            ("compression_comparison.png", _generate_compression_comparison),
            ("effective_goodput_comparison.png", _generate_effective_goodput_comparison),
            ("protocol_distribution_comparison.png", _generate_protocol_distribution_comparison),
        ]

        # Execute comparison plot generation in parallel using multiprocessing
        comparison_tasks = []

        # helper for constructing args
        def add_comp_task(func, *args):
            comparison_tasks.append((func, args))

        for plot_name, plot_func in plots:
            add_comp_task(plot_func, experiments_data, group_name, group_dir)

        generated = []
        print(f"  🔧 Generating {len(comparison_tasks)} comparison plots in parallel...")

        # Using a reasonable number of processes
        num_processes = min(multiprocessing.cpu_count(), len(comparison_tasks))

        with multiprocessing.Pool(processes=num_processes) as pool:
            # returns an iterator that yields results as soon as they are ready
            for res in pool.imap_unordered(_run_plot_task, comparison_tasks):
                if res:
                    generated.append(res)
                    print(f"  ✅ Generated {res}")

        # Generate comparison Markdown report
        print("  📝 Generating comparison report...")
        _generate_comparison_report(experiments_data, group_name, group_dir)

        print(f"  📈 {len(generated)} comparison plots + report generated in: {group_dir}")

    print(f"\n🎉 Comparison complete! Results in: {comparison_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze WAFL experiment results")
    parser.add_argument("--collect", action="store_true", help="Collect results from all nodes for all experiments (Overwrites/Updates)")
    parser.add_argument("--generate", action="store_true", help="Generate analysis graphs and validation reports for all experiments (Overwrites)")
    parser.add_argument("--additional", action="store_true", help="Generate additional plots for Experiment 1 (tradeoff and convergence)")
    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config()
    except Exception:
        # Fallback if config load fails (e.g., just analyzing locally without full config)
        config = {}

    if args.collect:
        print("📥 Starting data collection from all nodes...")
        experiments = get_all_experiments()
        if not experiments:
            print("✨ No experiments found in results/ to collect for.")
        else:
            print(f"📋 Found {len(experiments)} experiments. Updating results from nodes...")
            for exp_id in experiments:
                collect_results(exp_id, config)
        print("✅ Collection complete.")

    if args.generate:
        print("📊 Starting full analysis generation...")
        experiments = get_all_experiments()
        if not experiments:
            print("❌ No experiments found to analyze.")
        else:
            print(f"📋 Found {len(experiments)} experiments. generating individual reports...")
            # 1. Generate Individual Reports
            for exp_id in experiments:
                try:
                    analyze_results(exp_id)
                except Exception as e:
                    print(f"❌ Failed to analyze {exp_id}: {e}")

            # 2. Generate Comparison Reports
            print("\n📈 Generating comparison reports...")
            compare_experiments()

        print("✨ All generation tasks complete!")

    if args.additional:
        print("📊 Generating additional plots for Experiment 1...")
        _generate_additional_plots()
        print("✨ Additional plots generation complete!")

    if not args.collect and not args.generate and not args.additional:
        parser.print_help()


def _generate_additional_plots():
    """Generate additional plots for Experiment 1 for paper/presentation."""
    # Find Experiment 1, 2, 3, and 4 groups
    groups = _get_experiment_groups()
    target_groups = [g for g in groups.keys() if g.startswith("Experiment 1") or g.startswith("Experiment 2") or g.startswith("Experiment 3") or g.startswith("Experiment 4")]

    if not target_groups:
        print("❌ No Experiment 1 or 2 data found.")
        return

    for group_name in target_groups:
        print(f"\n📊 Generating additional plots for: {group_name}")
        experiments = groups[group_name]

        # Load data for all experiments in this group
        experiments_data = {}
        for exp_id in experiments:
            short_name = _get_short_name(exp_id)
            data = _load_experiment_data(exp_id)
            if data is not None and not data.empty:
                experiments_data[exp_id] = data
                print(f"  📂 Loading: {short_name}")

        if len(experiments_data) < 2:
            print("  ⚠️ Need at least 2 experiments, skipping.")
            continue

        # Create output directory
        safe_group_name = group_name.replace(" ", "_").replace(":", "").replace("(", "_").replace(")", "")
        output_dir = RESULTS_DIR / ".additional" / safe_group_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate plots based on experiment group
        if group_name.startswith("Experiment 1"):
            _generate_survival_epoch_tradeoff(experiments_data, group_name, output_dir)
            _generate_convergence_curve(experiments_data, group_name, output_dir)
        elif group_name.startswith("Experiment 2"):
            _generate_network_quality_overhead_correlation(experiments_data, group_name, output_dir)
            _generate_network_quality_survival_comparison(experiments_data, group_name, output_dir)
        elif group_name.startswith("Experiment 3"):
            _generate_epoch_breakdown_comparison(experiments_data, group_name, output_dir)
            _generate_convergence_curve(experiments_data, group_name, output_dir)
        elif group_name.startswith("Experiment 4"):
            _generate_epoch_breakdown_comparison(experiments_data, group_name, output_dir)
            _generate_convergence_curve(experiments_data, group_name, output_dir)

        print(f"  ✅ Additional plots generated in: {output_dir}")

        print(f"  ✅ Additional plots generated in: {output_dir}")


def _generate_survival_epoch_tradeoff(experiments_data, group_name, output_dir):
    """Generate Survival Rate, Epoch Duration, and Traffic Volume bar charts as separate files."""
    # Collect metrics for each experiment
    exp_names = []
    survival_rates = []
    epoch_durations = []
    traffic_volumes = []

    for exp_id, df in sorted(experiments_data.items(), key=lambda x: _get_experiment_sort_key(x[0])):
        short_name = _get_short_name(exp_id)
        exp_names.append(short_name)

        # Calculate survival rate
        if "survival_rate" in df.columns:
            wafl_df = df[df["phase"] == "WAFL"]
            if not wafl_df.empty:
                survival_rates.append(wafl_df["survival_rate"].mean() * 100)
            else:
                survival_rates.append(0)
        else:
            survival_rates.append(0)

        # Calculate average epoch duration
        if "epoch_duration_ms" in df.columns:
            wafl_df = df[df["phase"] == "WAFL"]
            if not wafl_df.empty:
                epoch_durations.append(wafl_df["epoch_duration_ms"].mean() / 1000)  # Convert to seconds
            else:
                epoch_durations.append(0)
        else:
            epoch_durations.append(0)

        # Calculate total traffic volume (bytes_sent)
        if "bytes_sent" in df.columns:
            wafl_df = df[df["phase"] == "WAFL"]
            if not wafl_df.empty:
                total_bytes = wafl_df["bytes_sent"].sum()
                traffic_volumes.append(total_bytes / (1024 * 1024))  # Convert to MB
            else:
                traffic_volumes.append(0)
        else:
            traffic_volumes.append(0)

    x = np.arange(len(exp_names))

    # Graph 1: Survival Rate (1:1 aspect ratio)
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    color1 = COLORS["goodput"]  # Green for survival rate
    ax1.bar(x, survival_rates, color=color1, alpha=0.8)
    ax1.set_xlabel("Method", fontsize=14)
    ax1.set_ylabel("Survival Rate [%]", fontsize=14)
    ax1.set_ylim(0, 105)
    ax1.set_xticks(x)
    ax1.set_xticklabels(exp_names, rotation=45, ha="right", fontsize=12)
    ax1.tick_params(axis="y", labelsize=12)
    ax1.set_title(f"Survival Rate by Method - {group_name}", fontsize=16)
    fig1.tight_layout()
    fig1.savefig(output_dir / "survival_rate_comparison.png", dpi=150)
    plt.close(fig1)
    print("  ✅ Generated survival_rate_comparison.png")

    # Graph 2: Epoch Duration (1:1 aspect ratio)
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    color2 = COLORS["bar_secondary"]  # Orange for epoch duration
    ax2.bar(x, epoch_durations, color=color2, alpha=0.8)
    ax2.set_xlabel("Method", fontsize=14)
    ax2.set_ylabel("Avg Epoch Duration [s]", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(exp_names, rotation=45, ha="right", fontsize=12)
    ax2.tick_params(axis="y", labelsize=12)
    ax2.set_title(f"Epoch Duration by Method - {group_name}", fontsize=16)
    fig2.tight_layout()
    fig2.savefig(output_dir / "epoch_duration_comparison.png", dpi=150)
    plt.close(fig2)
    print("  ✅ Generated epoch_duration_comparison.png")

    # Graph 3: Traffic Volume (1:1 aspect ratio)
    fig3, ax3 = plt.subplots(figsize=(8, 8))
    color3 = COLORS["traffic"]  # Teal for traffic
    ax3.bar(x, traffic_volumes, color=color3, alpha=0.8)
    ax3.set_xlabel("Method", fontsize=14)
    ax3.set_ylabel("Total Traffic Volume [MB]", fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(exp_names, rotation=45, ha="right", fontsize=12)
    ax3.tick_params(axis="y", labelsize=12)
    ax3.set_title(f"Traffic Volume by Method - {group_name}", fontsize=16)
    fig3.tight_layout()
    fig3.savefig(output_dir / "traffic_volume_comparison.png", dpi=150)
    plt.close(fig3)
    print("  ✅ Generated traffic_volume_comparison.png")

    plt.close(fig3)
    print("  ✅ Generated traffic_volume_comparison.png")


def _generate_epoch_breakdown_comparison(experiments_data, group_name, output_dir):
    """Generate Stacked Bar Chart for Epoch Duration Breakdown (Compute vs Comm)."""
    breakdown_data = []

    for exp_id, df in sorted(experiments_data.items(), key=lambda x: _get_experiment_sort_key(x[0])):
        if df.empty or "compute_time_ms" not in df.columns or "comm_time_ms" not in df.columns:
            continue

        wafl_df = df[df["phase"] == "WAFL"].copy()
        valid = wafl_df[(wafl_df["compute_time_ms"].notna()) & (wafl_df["comm_time_ms"].notna())]
        if valid.empty:
            continue

        # Get short name
        short_name = _get_short_name(exp_id)

        # Calculate means in seconds
        avg_compute = valid["compute_time_ms"].mean() / 1000
        avg_comm = valid["comm_time_ms"].mean() / 1000

        if "Experiment 4" in group_name:
            # Swap name format: Role (Method) -> Method (Role)
            if "Traditional (TCP)" in short_name:
                short_name = "TCP (Traditional)"
            elif "Competitor (TCP + SSP)" in short_name:
                short_name = "TCP + SSP (Competitor)"
            elif "Proposed (WAFL-Fast + SSP)" in short_name:
                short_name = "WAFL-Fast + SSP (Proposed)"

        breakdown_data.append({"Method": short_name, "Compute Time": avg_compute, "Comm Time": avg_comm})

    if not breakdown_data:
        return

    # Custom sort for Experiment 4
    if "Experiment 4" in group_name:
        order_map = {"TCP (Traditional)": 0, "TCP + SSP (Competitor)": 1, "WAFL-Fast + SSP (Proposed)": 2}
        # Sort breakdown_data based on the map, default to infinity if not found
        breakdown_data.sort(key=lambda x: order_map.get(x["Method"], float("inf")))

    df = pd.DataFrame(breakdown_data).set_index("Method")

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot stacked bar
    # Using specific colors: Compute (Blue/Dark), Comm (Orange/Highlight)
    colors = [COLORS["bar_primary"], COLORS["bar_secondary"]]
    df.plot(kind="bar", stacked=True, color=colors, ax=ax, alpha=0.9, width=0.6)

    ax.set_ylabel("Time [s]", fontsize=14)
    ax.set_xlabel("Method", fontsize=14)
    ax.set_title(f"Epoch Duration Breakdown - {group_name}", fontsize=16)

    # X-axis formatting
    ax.tick_params(axis="x", rotation=0, labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    # Legend
    ax.legend(title="Component", fontsize=12, loc="upper left")

    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    fig.savefig(output_dir / "epoch_breakdown_comparison.png", dpi=150)
    plt.close(fig)
    print("  ✅ Generated epoch_breakdown_comparison.png")


def _generate_convergence_curve(experiments_data, group_name, output_dir):
    """Generate convergence curve (Test Accuracy vs Wall-clock Time) for selected methods."""
    fig, ax = plt.subplots(figsize=(8, 8))

    colors_map = {
        "tcp": COLORS["mean_line"],  # Dark slate for TCP
        "proposed": COLORS["target_line"],  # Green for proposed
        "raw": COLORS["loss_fill"],  # Coral for raw
        "competitor": COLORS["bar_primary"],  # Blue for competitor
    }

    plotted = False
    plotted = False

    # Custom sort items for Experiment 4
    items = list(experiments_data.items())
    if "Experiment 4" in group_name:

        def exp4_sort_key(item):
            short_name = _get_short_name(item[0])
            if "Traditional" in short_name:
                return 0
            if "Competitor" in short_name:
                return 1
            if "Proposed" in short_name:
                return 2
            return 3  # Other

        items.sort(key=exp4_sort_key)
    else:
        items.sort(key=lambda x: _get_experiment_sort_key(x[0]))

    for exp_id, df in items:
        short_name = _get_short_name(exp_id)
        short_lower = short_name.lower()

        # Check if this is one of the target methods
        method_key = None
        if "tcp" in short_lower and "baseline" in short_lower:
            method_key = "tcp"
        elif "tcp" in short_lower and "bsp" in short_lower:  # Handle Experiment 3 naming
            method_key = "tcp"
        elif "traditional" in short_lower:  # Handle Experiment 4 naming
            method_key = "tcp"
        elif "proposed" in short_lower or ("fec" in short_lower and "compression" in short_lower and "nack" not in short_lower):
            method_key = "proposed"
        elif "wafl-fast" in short_lower:  # Handle Experiment 3 naming
            method_key = "proposed"
        elif "competitor" in short_lower:  # Handle Experiment 4 naming
            method_key = "competitor"
        elif "raw" in short_lower:
            method_key = "raw"

        if method_key is None:
            continue

        # Get WAFL phase data
        wafl_df = df[df["phase"] == "WAFL"].copy()
        if wafl_df.empty or "test_accuracy" not in wafl_df.columns or "epoch" not in wafl_df.columns:
            continue

        # Aggregate by epoch to smooth the curve (like accuracy_vs_time_comparison)
        if "wafl_relative_timestamp" in wafl_df.columns:
            epoch_stats = wafl_df.groupby("epoch").agg({"wafl_relative_timestamp": "mean", "test_accuracy": "mean"}).reset_index()
            x_data = epoch_stats["wafl_relative_timestamp"]
        elif "timestamp" in wafl_df.columns:
            epoch_stats = wafl_df.groupby("epoch").agg({"timestamp": "mean", "test_accuracy": "mean"}).reset_index()
            start_time = epoch_stats["timestamp"].min()
            x_data = epoch_stats["timestamp"] - start_time
        elif "epoch_duration_ms" in wafl_df.columns:
            epoch_stats = wafl_df.groupby("epoch").agg({"epoch_duration_ms": "mean", "test_accuracy": "mean"}).reset_index()
            x_data = epoch_stats["epoch_duration_ms"].cumsum() / 1000
        else:
            continue

        # Plot
        label = short_name
        if method_key == "tcp":
            label = "TCP (Traditional)"
        elif method_key == "proposed":
            label = "WAFL-Fast + SSP (Proposed)"
        elif method_key == "competitor":
            label = "TCP + SSP (Competitor)"
        elif method_key == "raw":
            label = "UDP (Raw)"

        ax.plot(x_data, epoch_stats["test_accuracy"] * 100, label=label, color=colors_map[method_key], linewidth=2)
        plotted = True

    if not plotted:
        print("  ⚠️ No matching experiments found for convergence curve")
        plt.close(fig)
        return

    # Determine target accuracy based on group
    target_acc = 0.90 if "Experiment 4" in group_name else TARGET_ACCURACY

    # Add target accuracy line
    ax.axhline(y=target_acc * 100, color=COLORS["phase_line"], linestyle="--", label=f"Target ({target_acc:.0%})", alpha=0.8, linewidth=2)

    ax.set_xlabel("Wall-clock Time [s]", fontsize=14)
    ax.set_ylabel("Test Accuracy [%]", fontsize=14)
    ax.set_title(f"Learning Convergence Curve - {group_name}", fontsize=16)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(60, 100 if target_acc > 0.85 else 90)  # Adjust Y-limit for higher target
    ax.tick_params(axis="both", labelsize=12)

    fig.tight_layout()
    fig.savefig(output_dir / "convergence_curve.png", dpi=150)
    plt.close(fig)
    print("  ✅ Generated convergence_curve.png")


def _generate_network_quality_overhead_correlation(experiments_data, group_name, output_dir):
    """Generate Overhead Ratio vs Network Quality correlation plot (Fig 6-3)."""
    # Organize data by condition and method
    data_map = _organize_exp2_data(experiments_data)
    conditions = ["Excellent", "Good", "Fair", "Poor"]

    tcp_overheads = []
    fast_overheads = []
    valid_conditions = []

    for cond in conditions:
        if cond not in data_map:
            continue

        methods = data_map[cond]
        has_data = False

        # Get TCP data
        if "TCP" in methods:
            df = methods["TCP"]
            # Calculate overhead ratio
            if "bytes_sent" in df.columns and "app_bytes_sent" in df.columns:
                wafl_df = df[df["phase"] == "WAFL"]
                total = wafl_df["bytes_sent"].sum()
                app = wafl_df["app_bytes_sent"].sum()
                ratio = total / (app + 1e-9) if app > 0 else 0
                tcp_overheads.append(ratio)
                has_data = True
            else:
                tcp_overheads.append(0)
        else:
            tcp_overheads.append(0)  # Or NaN

        # Get Fast data
        if "Fast" in methods:
            df = methods["Fast"]
            if "bytes_sent" in df.columns and "app_bytes_sent" in df.columns:
                wafl_df = df[df["phase"] == "WAFL"]
                total = wafl_df["bytes_sent"].sum()
                app = wafl_df["app_bytes_sent"].sum()
                ratio = total / (app + 1e-9) if app > 0 else 0
                fast_overheads.append(ratio)
                has_data = True
            else:
                fast_overheads.append(0)
        else:
            fast_overheads.append(0)

        if has_data:
            valid_conditions.append(cond)

    if not valid_conditions:
        print("  ⚠️ No valid data for overhead correlation plot.")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))

    x = range(len(valid_conditions))
    ax.plot(x, tcp_overheads, marker="o", linewidth=3, markersize=10, label="TCP", color=COLORS["mean_line"])
    ax.plot(x, fast_overheads, marker="s", linewidth=3, markersize=10, label="WAFL-Fast", color=COLORS["bar_secondary"])  # Orange/Yellowish

    ax.set_xticks(x)
    ax.set_xticklabels(valid_conditions, fontsize=12)
    ax.set_xlabel("Network Condition", fontsize=14)
    ax.set_ylabel("Overhead Ratio (Physical/App)", fontsize=14)
    ax.set_title(f"Network Quality vs Overhead Ratio - {group_name}", fontsize=16)

    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="y", labelsize=12)

    fig.tight_layout()
    fig.savefig(output_dir / "network_quality_overhead.png", dpi=150)
    plt.close(fig)
    print("  ✅ Generated network_quality_overhead.png")


def _generate_network_quality_survival_comparison(experiments_data, group_name, output_dir):
    """Generate Survival Rate comparison across network conditions (Fig 6-4)."""
    # Organize data
    data_map = _organize_exp2_data(experiments_data)
    conditions = ["Excellent", "Good", "Fair", "Poor"]

    tcp_survival = []
    fast_survival = []
    valid_conditions = []

    for cond in conditions:
        if cond not in data_map:
            continue

        methods = data_map[cond]
        has_data = False

        # Get TCP data
        if "TCP" in methods:
            df = methods["TCP"]
            if "survival_rate" in df.columns:
                wafl_df = df[df["phase"] == "WAFL"]
                val = wafl_df["survival_rate"].mean() * 100 if not wafl_df.empty else 0
                tcp_survival.append(val)
                has_data = True
            else:
                tcp_survival.append(0)
        else:
            tcp_survival.append(0)

        # Get Fast data
        if "Fast" in methods:
            df = methods["Fast"]
            if "survival_rate" in df.columns:
                wafl_df = df[df["phase"] == "WAFL"]
                val = wafl_df["survival_rate"].mean() * 100 if not wafl_df.empty else 0
                fast_survival.append(val)
                has_data = True
            else:
                fast_survival.append(0)
        else:
            fast_survival.append(0)

        if has_data:
            valid_conditions.append(cond)

    if not valid_conditions:
        print("  ⚠️ No valid data for survival comparison plot.")
        return

    # Plot Grouped Bar Chart
    fig, ax = plt.subplots(figsize=(8, 8))

    x = np.arange(len(valid_conditions))

    ax.set_ylabel("Survival Rate [%]", fontsize=14)
    ax.set_xlabel("Network Condition", fontsize=14)
    ax.set_title(f"Survival Rate vs Network Condition - {group_name}", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_conditions, fontsize=12)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="y", labelsize=12)

    # Add target line 60% if mentioned in requirements? (Requirement said "WAFL-Fast 60% line defense")
    ax.axhline(60, color="red", linestyle="--", alpha=0.5, label="60% Threshold")
    # Update legend to include threshold line
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=12, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_dir / "network_quality_survival.png", dpi=150)
    plt.close(fig)
    print("  ✅ Generated network_quality_survival.png")


def _organize_exp2_data(experiments_data):
    """Helper to organize Experiment 2 data by condition and method."""
    data_map = {}
    conditions = ["Excellent", "Good", "Fair", "Poor"]

    for exp_id, df in experiments_data.items():
        short_name = _get_short_name(exp_id)

        # Identify Condition
        cond = None
        for c in conditions:
            if c in short_name:
                cond = c
                break

        # Identify Method
        method = None
        if "TCP" in short_name:
            method = "TCP"
        elif "Fast" in short_name:
            method = "Fast"

        if cond and method:
            if cond not in data_map:
                data_map[cond] = {}
            data_map[cond][method] = df

    return data_map


if __name__ == "__main__":
    main()

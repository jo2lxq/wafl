import argparse
import json
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
import pandas as pd
import paramiko
import seaborn as sns

# Use non-interactive backend for parallel processing
matplotlib.use("Agg")

# Japanese font configuration for matplotlib
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

# Global seaborn theme for consistent styling across all plots
sns.set_theme(style="whitegrid")

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CTRL_DIR = PROJECT_ROOT / "ctrl"
CONFIG_FILE = CTRL_DIR / "execution_config.json"
RESULTS_DIR = PROJECT_ROOT / "results"

# Target accuracy for Time-to-Accuracy plot
TARGET_ACCURACY = 0.9

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


def get_latest_experiment_id():
    """Find the latest experiment directory in results/ based on timestamp in name."""
    if not RESULTS_DIR.exists():
        return None
    # List all directories in results/ excluding hidden ones like .deploy
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

    plt.figure(figsize=(12, 6))
    # Plot individual node times with light lines
    ax = sns.lineplot(
        data=epoch_dur,
        x="epoch",
        y="epoch_duration_s",
        hue="node",
        alpha=0.4,
        legend=False,
        palette=palette,
        estimator=None,
        linewidth=1.2,
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
    plt.title(f"Wall-clock Time per Epoch - {experiment_name}")
    plt.ylabel("Duration [sec]")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(analysis_dir / "epoch_duration.png", dpi=150)
    plt.close()
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

    plt.figure(figsize=(12, 6))
    ax = plt.subplot(1, 1, 1)
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
    pass

    # Add phase switch line
    add_phase_line_func(ax, df, x_col="epoch", text_position="top")

    plt.title(f"Idle Time Ratio (Sync Wait Time) - {experiment_name}")
    plt.tight_layout()
    plt.savefig(analysis_dir / "idle_time_ratio.png", dpi=150)
    plt.close()
    return "idle_time_ratio.png"


def _generate_wasted_computation_plot(df, experiment_name, analysis_dir):
    """Generate wasted computation plot (WAFL phase only)."""
    has_wasted_data = False
    if not df.empty and "wasted_ms" in df.columns:
        # Filter to WAFL phase only
        wafl_df = df[df["phase"] == "WAFL"].copy()
        ssp_data = wafl_df[(wafl_df["wasted_ms"].notna()) & (wafl_df["wasted_ms"] > 0)].copy()
        if not ssp_data.empty:
            has_wasted_data = True
            wasted_per_epoch = ssp_data.groupby("epoch").agg({"wasted_ms": "sum", "batches_processed": "sum"}).reset_index()
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

            ax2 = ax1.twinx()
            sns.lineplot(
                data=wasted_per_epoch,
                x="epoch",
                y="batches_processed",
                color=COLORS["wasted_line"],
                linewidth=2.5,
                marker="o",
                markersize=8,  # sns marker size logic is different. default is usually fine.
                label="Incomplete Batches",
                ax=ax2,
            )
            ax2.set_ylabel("Incomplete Batches (before force-skip)", color=COLORS["wasted_line"])
            ax2.tick_params(axis="y", labelcolor=COLORS["wasted_line"])

            plt.title(f"Wasted Computation (SSP Force-Skip) - {experiment_name}")
            fig.tight_layout()
            plt.savefig(analysis_dir / "wasted_computation.png", dpi=150)
            plt.close()
            return "wasted_computation.png"

    if not has_wasted_data:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(
            0.5,
            0.5,
            "No wasted computation data\n(SSP not enabled or no force-skips occurred)",
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
    plt.figure(figsize=(12, 6))
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

            ax = sns.lineplot(
                data=udp_data,
                x="epoch",
                y="survival_rate",
                hue="node",
                alpha=0.4,
                legend=False,
                palette=palette,
                estimator=None,
                linewidth=1.2,
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
                plt.title(f"Survival Rate (UDP/FEC) - {experiment_name}\n(100% survival - no packet loss or FEC recovery successful)")
            else:
                plt.title(f"Survival Rate (UDP/FEC) - {experiment_name}")
            plt.ylabel("Survival Rate")
            plt.xlabel("Epoch")
            plt.legend()

    if not has_meaningful_survival_data:
        ax = plt.gca()
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
        plt.title(f"Survival Rate (UDP/FEC) - {experiment_name}")
    plt.tight_layout()
    plt.savefig(analysis_dir / "survival_rate.png", dpi=150)
    plt.close()
    return "survival_rate.png"


def _generate_goodput_plot(df, experiment_name, analysis_dir):
    """Generate goodput plot showing both sent and received throughput (WAFL phase only)."""
    plt.figure(figsize=(12, 6))
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

            ax = plt.subplot(1, 1, 1)

            # Plot estimated physical throughput (includes retransmissions)
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
                plt.title(f"Throughput (Sent vs Goodput) - {experiment_name}\n(Avg Retrans Factor: {avg_retrans:.2f}x)")
            else:
                plt.title(f"Throughput (Sent vs Goodput) - {experiment_name}")

    if not has_meaningful_goodput_data:
        ax = plt.gca()
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
        plt.title(f"Goodput (Effective Throughput) - {experiment_name}")
    plt.tight_layout()
    plt.savefig(analysis_dir / "goodput.png", dpi=150)
    plt.close()
    return "goodput.png"


def _generate_traffic_volume_plot(df, experiment_name, analysis_dir):
    """Generate traffic volume plot with cumulative line (WAFL phase only)."""
    plt.figure(figsize=(12, 6))
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

            fig, ax1 = plt.subplots(figsize=(12, 6))

            # Bar chart for per-epoch traffic
            # Bar chart for per-epoch traffic
            # Bar chart for per-epoch traffic (Use matplotlib bar for correct numeric x-axis alignment)
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

            plt.title(f"Traffic Volume (WAFL Phase) - {experiment_name}")
            fig.tight_layout()
            plt.savefig(analysis_dir / "traffic_volume.png", dpi=150)
            plt.close()
            return "traffic_volume.png"

    if not has_meaningful_traffic_data:
        ax = plt.gca()
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
        plt.title(f"Traffic Volume - {experiment_name}")
    plt.tight_layout()
    plt.savefig(analysis_dir / "traffic_volume.png", dpi=150)
    plt.close()
    return "traffic_volume.png"


def _generate_transfer_time_plot(df, experiment_name, analysis_dir):
    """Generate total transfer time plot with compression time visibility."""

    if not df.empty and "compression_time_ms" in df.columns and "epoch_duration_ms" in df.columns:
        transfer_data = df[(df["compression_time_ms"].notna()) & (df["epoch_duration_ms"].notna())].copy()
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

            plt.tight_layout()
            plt.savefig(analysis_dir / "total_transfer_time.png", dpi=150)
            plt.close()
            return "total_transfer_time.png"

    # No meaningful data case
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
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
    plt.title(f"Total Transfer Time (T_comm + T_comp) - {experiment_name}")
    plt.tight_layout()
    plt.savefig(analysis_dir / "total_transfer_time.png", dpi=150)
    plt.close()
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
        edgecolor="none",
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
        edgecolor="none",
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
            edgecolor="none",
            hatch="//",  # Add hatch to distinguish
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Time [sec]")
    ax.legend(loc="upper right")
    plt.title(f"Time Breakdown (Compute vs Comm vs Wait) - {experiment_name}")
    plt.tight_layout()
    plt.savefig(analysis_dir / "compute_comm_breakdown.png", dpi=150)
    plt.close()
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

    plt.figure(figsize=(12, 6))
    epoch_stats = wafl_data.groupby("epoch").agg(
        {
            "wafl_relative_timestamp": "mean",
            "test_accuracy": ["mean", "std", "min", "max"],
        }
    )
    epoch_stats.columns = ["timestamp", "mean", "std", "min", "max"]
    epoch_stats = epoch_stats.reset_index()
    epoch_stats["std"] = epoch_stats["std"].fillna(0)

    ax = plt.gca()

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

    plt.title(f"Time-to-Accuracy (WAFL Phase, Target: {TARGET_ACCURACY:.0%}) - {experiment_name}")
    plt.ylabel("Test Accuracy")
    plt.xlabel("Elapsed Time from WAFL Start [sec]")
    plt.ylim(0, 1.05)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(analysis_dir / "time_to_accuracy.png", dpi=150)
    plt.close()
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

    plt.figure(figsize=(12, 6))

    # Plot train accuracy first (background, thinner, more transparent)
    train_data = acc_df[acc_df["metric"] == "train_accuracy"]
    test_data = acc_df[acc_df["metric"] == "test_accuracy"]

    ax = sns.lineplot(
        data=train_data,
        x="epoch",
        y="value",
        color=COLORS["accuracy_fill"],
        linewidth=1.5,
        alpha=0.5,
        label="Train Accuracy",
        errorbar="sd",
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
    plt.title(f"Accuracy over Epochs (Mean +/- SD) - {experiment_name}")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(analysis_dir / "accuracy_mean.png", dpi=150)
    plt.close()
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

    plt.figure(figsize=(12, 6))

    # Plot train loss first (background, thinner, more transparent)
    train_data = loss_df[loss_df["metric"] == "train_loss"]
    test_data = loss_df[loss_df["metric"] == "test_loss"]

    ax = sns.lineplot(
        data=train_data,
        x="epoch",
        y="value",
        color=COLORS["loss_fill"],
        linewidth=1.5,
        alpha=0.5,
        label="Train Loss",
        errorbar="sd",
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
    plt.title(f"Loss over Epochs (Mean +/- SD) - {experiment_name}")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(analysis_dir / "loss_mean.png", dpi=150)
    plt.close()
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

    plt.figure(figsize=(12, 6))
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
    )
    ax = sns.lineplot(
        data=plot_df,
        x="epoch",
        y="test_accuracy",
        color=COLORS["mean_line"],
        linewidth=2.5,
        label="Mean",
        errorbar=None,
    )
    add_phase_line_func(ax, plot_df, "epoch", text_position="bottom")
    plt.title(f"Node-wise Test Accuracy - {experiment_name}")
    plt.ylabel("Test Accuracy")
    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(analysis_dir / "accuracy_nodes.png", dpi=150)
    plt.close()
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

    plt.figure(figsize=(12, 6))
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
    )
    ax = sns.lineplot(
        data=plot_df,
        x="epoch",
        y="test_loss",
        color=COLORS["mean_line"],
        linewidth=2.5,
        label="Mean",
        errorbar=None,
    )
    add_phase_line_func(ax, plot_df, "epoch")
    plt.title(f"Node-wise Test Loss - {experiment_name}")
    plt.ylabel("Test Loss")
    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(analysis_dir / "loss_nodes.png", dpi=150)
    plt.close()
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

    plt.figure(figsize=(12, 6))

    # Calculate epoch stats with mean time and accuracy
    epoch_stats = plot_data.groupby("epoch").agg({"timestamp": "mean", "test_accuracy": ["mean", "std"]})
    epoch_stats.columns = ["timestamp", "mean", "std"]
    epoch_stats = epoch_stats.reset_index()
    epoch_stats["std"] = epoch_stats["std"].fillna(0)

    ax = plt.gca()

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

    plt.title(f"Test Accuracy vs Time - {experiment_name}")
    plt.ylabel("Test Accuracy")
    plt.xlabel("Elapsed Time [sec]")
    plt.ylim(0, 1.05)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(analysis_dir / "accuracy_vs_time.png", dpi=150)
    plt.close()
    return "accuracy_vs_time.png"


def _generate_fec_overhead_plot(df, experiment_name, analysis_dir):
    """Generate FEC processing overhead plot (encode + decode time)."""
    plt.figure(figsize=(12, 6))
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

                fig, ax = plt.subplots(figsize=(12, 6))
                bar_width = 0.8

                # Stacked bar chart: Encode (bottom) + Decode (top)
                ax.bar(
                    fec_agg["epoch"],
                    fec_agg["fec_encode_time_ms"],
                    bar_width,
                    label="FEC Encode",
                    color=COLORS["bar_primary"],
                    alpha=0.9,
                    edgecolor="none",
                )
                ax.bar(
                    fec_agg["epoch"],
                    fec_agg["fec_decode_time_ms"],
                    bar_width,
                    bottom=fec_agg["fec_encode_time_ms"],
                    label="FEC Decode",
                    color=COLORS["bar_secondary"],
                    alpha=0.9,
                    edgecolor="none",
                )

                ax.set_xlabel("Epoch")
                ax.set_ylabel("Time [ms]")
                ax.legend(loc="upper right")

                # Calculate average overhead for title
                avg_encode = fec_agg["fec_encode_time_ms"].mean()
                avg_decode = fec_agg["fec_decode_time_ms"].mean()
                avg_total = avg_encode + avg_decode

                plt.title(f"FEC Processing Overhead - {experiment_name}\n(Avg: Encode={avg_encode:.1f}ms, Decode={avg_decode:.1f}ms, Total={avg_total:.1f}ms)")
                plt.tight_layout()
                plt.savefig(analysis_dir / "fec_overhead.png", dpi=150)
                plt.close()
                return "fec_overhead.png"

    if not has_meaningful_data:
        ax = plt.gca()
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
        plt.title(f"FEC Processing Overhead - {experiment_name}")

    plt.tight_layout()
    plt.savefig(analysis_dir / "fec_overhead.png", dpi=150)
    plt.close()
    return "fec_overhead.png"


def _generate_cumulative_sent_data_plot(df, experiment_name, analysis_dir):
    """Generate cumulative sent data (bytes_sent) bar plot."""
    plt.figure(figsize=(12, 6))
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

            ax = plt.subplot(1, 1, 1)
            ax = plt.subplot(1, 1, 1)
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
            plt.title(f"Cumulative Sent Data - {experiment_name}")

    if not has_meaningful_data:
        ax = plt.gca()
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
        plt.title(f"Cumulative Sent Data - {experiment_name}")

    plt.tight_layout()
    plt.savefig(analysis_dir / "cumulative_sent_data.png", dpi=150)
    plt.close()
    return "cumulative_sent_data.png"


def _generate_rudp_retransmission_plot(df, experiment_name, analysis_dir, add_phase_line_func):
    """Generate RUDP retransmission count plot."""
    plt.figure(figsize=(12, 6))
    has_meaningful_data = False

    if not df.empty and "rudp_retransmissions" in df.columns:
        # Filter to WAFL phase only
        wafl_df = df[df["phase"] == "WAFL"].copy()
        rudp_data = wafl_df[wafl_df["rudp_retransmissions"].notna()].copy()

        if not rudp_data.empty and rudp_data["rudp_retransmissions"].sum() > 0:
            has_meaningful_data = True
            num_nodes = rudp_data["node"].nunique()
            palette = NODE_PALETTE[:num_nodes] if num_nodes <= len(NODE_PALETTE) else "husl"

            ax = sns.lineplot(
                data=rudp_data,
                x="epoch",
                y="rudp_retransmissions",
                hue="node",
                alpha=0.4,
                legend=False,
                palette=palette,
                estimator=None,
                linewidth=1.2,
            )
            sns.lineplot(
                data=rudp_data,
                x="epoch",
                y="rudp_retransmissions",
                color=COLORS["mean_line"],
                linewidth=2.5,
                label="Mean",
                errorbar=None,
                ax=ax,
            )
            add_phase_line_func(ax, df, "epoch")
            plt.title(f"RUDP Retransmissions - {experiment_name}")
            plt.ylabel("Retransmissions")
            plt.xlabel("Epoch")
            plt.legend()

    if not has_meaningful_data:
        ax = plt.gca()
        ax.text(
            0.5,
            0.5,
            "RUDP not enabled\n(no retransmission data)",
            ha="center",
            va="center",
            fontsize=14,
            color=COLORS["no_data"],
            transform=ax.transAxes,
        )
        ax.axis("off")
        plt.title(f"RUDP Retransmissions - {experiment_name}")

    plt.tight_layout()
    plt.savefig(analysis_dir / "rudp_retransmissions.png", dpi=150)
    plt.close()
    return "rudp_retransmissions.png"


def _generate_rudp_rtt_plot(df, experiment_name, analysis_dir, add_phase_line_func):
    """Generate RUDP average RTT plot."""
    plt.figure(figsize=(12, 6))
    has_meaningful_data = False

    if not df.empty and "rudp_avg_rtt_ms" in df.columns:
        # Filter to WAFL phase only
        wafl_df = df[df["phase"] == "WAFL"].copy()
        rtt_data = wafl_df[wafl_df["rudp_avg_rtt_ms"].notna()].copy()

        if not rtt_data.empty and rtt_data["rudp_avg_rtt_ms"].sum() > 0:
            has_meaningful_data = True
            num_nodes = rtt_data["node"].nunique()
            palette = NODE_PALETTE[:num_nodes] if num_nodes <= len(NODE_PALETTE) else "husl"

            ax = sns.lineplot(
                data=rtt_data,
                x="epoch",
                y="rudp_avg_rtt_ms",
                hue="node",
                alpha=0.4,
                legend=False,
                palette=palette,
                estimator=None,
                linewidth=1.2,
            )
            sns.lineplot(
                data=rtt_data,
                x="epoch",
                y="rudp_avg_rtt_ms",
                color=COLORS["mean_line"],
                linewidth=2.5,
                label="Mean",
                errorbar=None,
                ax=ax,
            )
            add_phase_line_func(ax, df, "epoch")
            plt.title(f"RUDP Average RTT - {experiment_name}")
            plt.ylabel("RTT [ms]")
            plt.xlabel("Epoch")
            plt.legend()

    if not has_meaningful_data:
        ax = plt.gca()
        ax.text(
            0.5,
            0.5,
            "RUDP not enabled\n(no RTT data)",
            ha="center",
            va="center",
            fontsize=14,
            color=COLORS["no_data"],
            transform=ax.transAxes,
        )
        ax.axis("off")
        plt.title(f"RUDP Average RTT - {experiment_name}")

    plt.tight_layout()
    plt.savefig(analysis_dir / "rudp_rtt.png", dpi=150)
    plt.close()
    return "rudp_rtt.png"


def _generate_rudp_aging_plot(df, experiment_name, analysis_dir, add_phase_line_func):
    """Generate E-RUDP aged packets plot."""
    plt.figure(figsize=(12, 6))
    has_meaningful_data = False

    if not df.empty and "rudp_aged_packets" in df.columns:
        # Filter to WAFL phase only
        wafl_df = df[df["phase"] == "WAFL"].copy()
        aging_data = wafl_df[wafl_df["rudp_aged_packets"].notna()].copy()

        if not aging_data.empty and aging_data["rudp_aged_packets"].sum() > 0:
            has_meaningful_data = True

            # Aggregate by epoch
            aging_agg = aging_data.groupby("epoch").agg({"rudp_aged_packets": "sum"}).reset_index()

            ax = plt.subplot(1, 1, 1)
            ax = plt.subplot(1, 1, 1)
            ax.bar(
                aging_agg["epoch"],
                aging_agg["rudp_aged_packets"],
                color=COLORS["wasted_bar"],
                alpha=0.8,
                edgecolor=COLORS["wasted_bar"],
                linewidth=0.5,
                width=0.6,
            )
            ax.set_ylabel("Aged Packets")
            ax.set_xlabel("Epoch")
            plt.title(f"E-RUDP Aged Packets - {experiment_name}")

    if not has_meaningful_data:
        ax = plt.gca()
        ax.text(
            0.5,
            0.5,
            "E-RUDP not enabled or no packets aged\n(aging_limit not exceeded)",
            ha="center",
            va="center",
            fontsize=14,
            color=COLORS["no_data"],
            transform=ax.transAxes,
        )
        ax.axis("off")
        plt.title(f"E-RUDP Aged Packets - {experiment_name}")

    plt.tight_layout()
    plt.savefig(analysis_dir / "rudp_aging.png", dpi=150)
    plt.close()
    return "rudp_aging.png"


def _generate_udp_dynamic_params_plot(df, experiment_name, analysis_dir, add_phase_line_func):
    """Generate UDP dynamic parameters plot (Parity & Pacing)."""
    plt.figure(figsize=(12, 8))
    has_meaningful_data = False

    if not df.empty and ("udp_avg_parity" in df.columns or "udp_avg_pacing_ms" in df.columns):
        wafl_df = df[df["phase"] == "WAFL"].copy()

        # Check if there is any non-zero data
        has_parity = "udp_avg_parity" in wafl_df.columns and wafl_df["udp_avg_parity"].sum() > 0
        has_pacing = "udp_avg_pacing_ms" in wafl_df.columns and wafl_df["udp_avg_pacing_ms"].sum() > 0

        if has_parity or has_pacing:
            has_meaningful_data = True

            # Aggregate per epoch (mean across nodes)
            agg_funcs = {}
            if "udp_avg_parity" in wafl_df.columns:
                agg_funcs["udp_avg_parity"] = "mean"
            if "udp_avg_pacing_ms" in wafl_df.columns:
                agg_funcs["udp_avg_pacing_ms"] = "mean"

            param_agg = wafl_df.groupby("epoch").agg(agg_funcs).reset_index()

            # Subplot 1: Average Parity
            ax1 = plt.subplot(2, 1, 1)
            if has_parity:
                sns.lineplot(data=param_agg, x="epoch", y="udp_avg_parity", color=COLORS["wasted_bar"], linewidth=2, label="Avg Parity (m-k)", ax=ax1)
            # add_phase_line_func(ax1, df, "epoch") # Removed for Clean Dynamic Plot
            ax1.set_title(f"Dynamic UDP Parameters - {experiment_name}")
            ax1.set_ylabel("Avg Parity")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Subplot 2: Average Pacing
            ax2 = plt.subplot(2, 1, 2, sharex=ax1)
            if has_pacing:
                sns.lineplot(data=param_agg, x="epoch", y="udp_avg_pacing_ms", color=COLORS["goodput"], linewidth=2, label="Avg Pacing Delay [ms]", ax=ax2)
            # add_phase_line_func(ax2, df, "epoch") # Removed
            ax2.set_ylabel("Pacing [ms]")
            ax2.set_xlabel("Epoch")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

    if not has_meaningful_data:
        plt.clf()  # Clear standard layout
        ax = plt.gca()
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
        plt.title(f"Dynamic UDP Parameters - {experiment_name}")

    plt.tight_layout()
    plt.savefig(analysis_dir / "udp_dynamic_params.png", dpi=150)
    plt.close()
    return "udp_dynamic_params.png"


def _generate_protocol_distribution_plot(df, experiment_name, analysis_dir, add_phase_line_func):
    """Generate protocol distribution plot (Dynamic Mode)."""
    plt.figure(figsize=(12, 6))
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

            # Normalize to percentage? Or raw counts?
            # User wants to know "how it changed".
            # Stacked bar 100% is good to show ratio.
            # But raw counts shows volume too.
            # Let's do Stacked Bar (Raw Counts) to show scale as well.

            ax = plt.subplot(1, 1, 1)

            bottom = None
            labels = {"protocol_tcp_count": "TCP", "protocol_udp_count": "UDP", "protocol_rudp_count": "RUDP"}
            colors = {"protocol_tcp_count": "#3498db", "protocol_udp_count": "#e67e22", "protocol_rudp_count": "#9b59b6"}  # Blue, Orange, Purple

            for col in avail_cols:
                if col not in proto_agg.columns:
                    continue

                # Check if this protocol was actually used
                if proto_agg[col].sum() == 0:
                    continue

                ax.bar(proto_agg["epoch"], proto_agg[col], label=labels[col], bottom=bottom, color=colors.get(col, "gray"), alpha=0.8, width=0.8)
                if bottom is None:
                    bottom = proto_agg[col]
                else:
                    bottom += proto_agg[col]

            # add_phase_line_func(ax, df, "epoch") # Removed
            ax.set_ylabel("Total Transfers (Count)")
            ax.set_xlabel("Epoch")
            ax.legend()
            ax.set_title(f"Dynamic Protocol Distribution - {experiment_name}")
            ax.grid(True, axis="y", alpha=0.3)

    if not has_meaningful_data:
        ax = plt.gca()
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
        plt.title(f"Dynamic Protocol Distribution - {experiment_name}")

    plt.tight_layout()
    plt.savefig(analysis_dir / "protocol_distribution.png", dpi=150)
    plt.close()
    return "protocol_distribution.png"


def _generate_udp_recovery_plot(df, experiment_name, analysis_dir, add_phase_line_func):
    """Generate UDP FEC recovery stats (Success vs Fail) plot."""
    plt.figure(figsize=(12, 6))
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

            ax = plt.subplot(1, 1, 1)
            bar_width = 0.8

            # Success (bottom), Fail (top)
            # Failures are usually small, so stacking on top makes sense?
            # Or side-by-side? Failures are critical. Let's stack.

            s_vals = rec_agg["fec_recovery_success"] if has_success else pd.Series([0] * len(rec_agg))
            f_vals = rec_agg["fec_recovery_fail"] if has_fail else pd.Series([0] * len(rec_agg))

            ax.bar(rec_agg["epoch"], s_vals, label="FEC Recovered", color=COLORS["bar_primary"], alpha=0.9, width=bar_width)
            ax.bar(rec_agg["epoch"], f_vals, bottom=s_vals, label="Recovery Failed", color=COLORS["control_server"], alpha=0.9, width=bar_width)

            # add_phase_line_func(ax, df, "epoch") # Removed
            ax.set_ylabel("Models (Count)")
            ax.set_xlabel("Epoch")
            ax.legend()
            ax.set_title(f"UDP FEC Recovery Status - {experiment_name}")
            ax.grid(True, axis="y", alpha=0.3)

    if not has_meaningful_data:
        ax = plt.gca()
        ax.text(0.5, 0.5, "UDP FEC stats N/A", ha="center", va="center", color=COLORS["no_data"], transform=ax.transAxes)
        ax.axis("off")
        plt.title(f"UDP FEC Recovery Status - {experiment_name}")

    plt.tight_layout()
    plt.savefig(analysis_dir / "udp_recovery.png", dpi=150)
    plt.close()
    return "udp_recovery.png"


def _generate_rudp_failure_plot(df, experiment_name, analysis_dir, add_phase_line_func):
    """Generate RUDP failure (Max Retries Reached) plot."""
    plt.figure(figsize=(12, 6))
    has_meaningful_data = False

    if not df.empty and "rudp_max_retries_reached" in df.columns:
        wafl_df = df[df["phase"] == "WAFL"].copy()
        if wafl_df["rudp_max_retries_reached"].sum() > 0:
            has_meaningful_data = True

            agg = wafl_df.groupby("epoch").agg({"rudp_max_retries_reached": "sum"}).reset_index()

            ax = plt.subplot(1, 1, 1)
            ax = plt.subplot(1, 1, 1)
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
        ax = plt.gca()
        ax.text(0.5, 0.5, "RUDP Failures N/A", ha="center", va="center", color=COLORS["no_data"], transform=ax.transAxes)
        ax.axis("off")
        plt.title(f"RUDP Max Retry Failures - {experiment_name}")

    plt.tight_layout()
    plt.savefig(analysis_dir / "rudp_failures.png", dpi=150)
    plt.close()
    return "rudp_failures.png"


def _generate_rudp_control_overhead_plot(df, experiment_name, analysis_dir, add_phase_line_func):
    """Generate RUDP Control Packet Overhead plot."""
    plt.figure(figsize=(12, 6))
    has_meaningful_data = False

    cols = ["rudp_acks_sent", "rudp_eaks_sent", "rudp_nacks_sent"]
    avail_cols = [c for c in cols if c in df.columns]

    if not df.empty and avail_cols:
        wafl_df = df[df["phase"] == "WAFL"].copy()
        if wafl_df[avail_cols].sum().sum() > 0:
            has_meaningful_data = True

            # agg = wafl_df.groupby("epoch").agg({c: "mean" for c in avail_cols}).reset_index()

            ax = sns.lineplot(data=wafl_df, x="epoch", y="rudp_acks_sent", label="ACKs", errorbar=None)
            if "rudp_eaks_sent" in wafl_df.columns:
                sns.lineplot(data=wafl_df, x="epoch", y="rudp_eaks_sent", label="EAKs", errorbar=None, ax=ax)
            if "rudp_nacks_sent" in wafl_df.columns:
                sns.lineplot(data=wafl_df, x="epoch", y="rudp_nacks_sent", label="NACKs", errorbar=None, ax=ax)  # Should be 0 usually?

            add_phase_line_func(ax, df, "epoch")
            ax.set_ylabel("Control Packets (Avg per Node)")
            ax.set_xlabel("Epoch")
            ax.set_title(f"RUDP Control Overhead - {experiment_name}")
            ax.legend()
            ax.grid(True, alpha=0.3)

    if not has_meaningful_data:
        ax = plt.gca()
        ax.text(0.5, 0.5, "RUDP Overhead N/A", ha="center", va="center", color=COLORS["no_data"], transform=ax.transAxes)
        ax.axis("off")
        plt.title(f"RUDP Control Overhead - {experiment_name}")

    plt.tight_layout()
    plt.savefig(analysis_dir / "rudp_control_overhead.png", dpi=150)
    plt.close()
    return "rudp_control_overhead.png"


def _generate_compression_stats_plot(df, experiment_name, analysis_dir, add_phase_line_func):
    """Generate Compression Ratio and Time plot."""
    plt.figure(figsize=(12, 8))
    has_meaningful_data = False

    if not df.empty and "compression_ratio" in df.columns:
        wafl_df = df[df["phase"] == "WAFL"].copy()

        # Check if compression was active (ratio > 1.05 or explicit)
        if wafl_df["compression_ratio"].mean() > 1.01:
            has_meaningful_data = True

            agg = wafl_df.groupby("epoch").agg({"compression_ratio": "mean", "compression_time_ms": "mean"}).reset_index()

            # Subplot 1: Ratio
            ax1 = plt.subplot(2, 1, 1)
            sns.lineplot(data=agg, x="epoch", y="compression_ratio", color=COLORS["goodput"], linewidth=2, ax=ax1)
            add_phase_line_func(ax1, df, "epoch")
            ax1.set_title(f"Compression Statistics - {experiment_name}")
            ax1.set_ylabel("Compression Ratio")
            ax1.grid(True, alpha=0.3)

            # Subplot 2: Time
            ax2 = plt.subplot(2, 1, 2, sharex=ax1)
            sns.lineplot(data=agg, x="epoch", y="compression_time_ms", color=COLORS["wasted_bar"], linewidth=2, ax=ax2)
            add_phase_line_func(ax2, df, "epoch")
            ax2.set_ylabel("Compression Time [ms]")
            ax2.set_xlabel("Epoch")
            ax2.grid(True, alpha=0.3)

    if not has_meaningful_data:
        plt.clf()
        ax = plt.gca()
        ax.text(0.5, 0.5, "Compression not used", ha="center", va="center", color=COLORS["no_data"], transform=ax.transAxes)
        ax.axis("off")
        plt.title(f"Compression Statistics - {experiment_name}")

    plt.tight_layout()
    plt.savefig(analysis_dir / "compression_stats.png", dpi=150)
    plt.close()
    return "compression_stats.png"


def _calculate_metrics_summary(df, wafl_phase_start_relative=None):
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

    # 3. Time to Target Accuracy (90%)
    if "wafl_relative_timestamp" in wafl_clean.columns and "test_accuracy" in wafl_clean.columns:
        epoch_acc = wafl_clean.groupby("epoch").agg({"test_accuracy": "mean", "wafl_relative_timestamp": "mean"}).reset_index()
        reached = epoch_acc[epoch_acc["test_accuracy"] >= TARGET_ACCURACY]
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
        valid = wafl_df[(wafl_df["comm_time_ms"].notna()) & (wafl_df["comm_time_ms"] > 0)]
        if not valid.empty and valid["bytes_received"].sum() > 0:
            total_bytes = valid["bytes_received"].sum()
            total_time_s = valid["comm_time_ms"].sum() / 1000
            avg_goodput = (total_bytes * 8 / 1e6) / total_time_s
            metrics["avg_goodput_mbps"] = avg_goodput

    # 9. Number of Nodes
    if "node" in df.columns:
        metrics["num_nodes"] = df["node"].nunique()

    # 10. Total Epochs
    if "epoch" in df.columns:
        metrics["total_epochs"] = int(df["epoch"].max())

    # 11. WAFL Epochs
    if "epoch" in wafl_df.columns and not wafl_df.empty:
        metrics["wafl_epochs"] = int(wafl_df["epoch"].max() - wafl_df["epoch"].min() + 1)

    # 12. RUDP Retransmissions (total)
    if "rudp_retransmissions" in wafl_df.columns:
        total_retrans = wafl_df["rudp_retransmissions"].sum()
        if pd.notna(total_retrans) and total_retrans > 0:
            metrics["rudp_retransmissions"] = int(total_retrans)

    # 13. FEC Processing Time (encode + decode)
    if "fec_encode_time_ms" in wafl_df.columns:
        avg_encode = wafl_df["fec_encode_time_ms"].mean()
        if pd.notna(avg_encode) and avg_encode > 0:
            metrics["avg_fec_encode_ms"] = avg_encode
    if "fec_decode_time_ms" in wafl_df.columns:
        avg_decode = wafl_df["fec_decode_time_ms"].mean()
        if pd.notna(avg_decode) and avg_decode > 0:
            metrics["avg_fec_decode_ms"] = avg_decode

    # 14. UDP Recovery Stats
    if "fec_recovery_success" in wafl_df.columns:
        metrics["udp_recovery_success"] = int(wafl_df["fec_recovery_success"].sum())
    if "fec_recovery_fail" in wafl_df.columns:
        metrics["udp_recovery_fail"] = int(wafl_df["fec_recovery_fail"].sum())

    # 15. RUDP Failures
    if "rudp_max_retries_reached" in wafl_df.columns:
        metrics["rudp_max_retries_total"] = int(wafl_df["rudp_max_retries_reached"].sum())

    # 16. Compression Stats
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
    if "rudp_retransmissions" in metrics:
        lines.append(f"| RUDP Retransmissions | {metrics['rudp_retransmissions']} |")
    if "avg_fec_encode_ms" in metrics:
        lines.append(f"| Avg FEC Encode Time | {metrics['avg_fec_encode_ms']:.2f} ms |")
    if "avg_fec_decode_ms" in metrics:
        lines.append(f"| Avg FEC Decode Time | {metrics['avg_fec_decode_ms']:.2f} ms |")
    if "avg_fec_encode_ms" in metrics and "avg_fec_decode_ms" in metrics:
        total_fec = metrics["avg_fec_encode_ms"] + metrics["avg_fec_decode_ms"]
        lines.append(f"| Avg Total FEC Overhead | {total_fec:.2f} ms |")
    if "udp_recovery_success" in metrics and "udp_recovery_fail" in metrics:
        total_rec = metrics["udp_recovery_success"] + metrics["udp_recovery_fail"]
        rate = metrics["udp_recovery_success"] / total_rec if total_rec > 0 else 0
        lines.append(f"| UDP FEC Recovery Rate | {rate:.1%} ({metrics['udp_recovery_success']}/{total_rec}) |")
    if "rudp_max_retries_total" in metrics:
        lines.append(f"| RUDP Max Retry Failures | {metrics['rudp_max_retries_total']} |")
    if "compression_ratio_avg" in metrics:
        lines.append(f"| Avg Compression Ratio | {metrics['compression_ratio_avg']:.2f}x |")
    if "compression_time_avg" in metrics:
        lines.append(f"| Avg Compression Time | {metrics['compression_time_avg']:.2f} ms |")
    lines.append("")
    # Dynamic Protocol Stats Section
    if "protocol_tcp_count" in df.columns:
        lines.append("## Dynamic Protocol Statistics")
        lines.append("")
        headers_dyn = ["Epoch", "TCP Transfers", "UDP Transfers", "RUDP Transfers", "Total Transfers"]
        lines.append("| " + " | ".join(headers_dyn) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers_dyn)) + " |")

        wafl_df = df[df["phase"] == "WAFL"].copy()
        count_cols = ["protocol_tcp_count", "protocol_udp_count", "protocol_rudp_count"]
        # Ensure cols exist
        valid_cols = [c for c in count_cols if c in wafl_df.columns]

        if valid_cols:
            proto_agg = wafl_df.groupby("epoch")[valid_cols].sum().reset_index()
            for _, row_data in proto_agg.iterrows():
                tcp_c = int(row_data.get("protocol_tcp_count", 0))
                udp_c = int(row_data.get("protocol_udp_count", 0))
                rudp_c = int(row_data.get("protocol_rudp_count", 0))
                total_c = tcp_c + udp_c + rudp_c
                lines.append(f"| {int(row_data['epoch'])} | {tcp_c} | {udp_c} | {rudp_c} | {total_c} |")
    lines.append("")
    lines.append("")

    # Generated Graphs Section
    lines.append("## Generated Graphs")
    lines.append("")

    graph_descriptions = {
        "accuracy_mean.png": "Train and Test Accuracy (Mean ± SD)",
        "accuracy_nodes.png": "Node-wise Test Accuracy",
        "accuracy_vs_time.png": "Test Accuracy vs Elapsed Time",
        "loss_mean.png": "Train and Test Loss (Mean ± SD)",
        "loss_nodes.png": "Node-wise Test Loss",
        "time_to_accuracy.png": f"Time to Target Accuracy ({TARGET_ACCURACY * 100:.0f}%)",
        "epoch_duration.png": "Wall-clock Time per Epoch",
        "idle_time_ratio.png": "Idle Time Ratio (Sync Wait)",
        "compute_comm_breakdown.png": "Compute vs Communication Time",
        "survival_rate.png": "Survival Rate (UDP/FEC)",
        "goodput.png": "Throughput (Sent vs Goodput)",
        "traffic_volume.png": "Traffic Volume per Epoch",
        "total_transfer_time.png": "Transfer Time Breakdown",
        "wasted_computation.png": "Wasted Computation (SSP)",
        "fec_overhead.png": "FEC Processing Overhead (Encode + Decode)",
        "rudp_retransmissions.png": "RUDP Retransmissions",
        "rudp_rtt.png": "RUDP Average RTT",
        "rudp_aging.png": "E-RUDP Aged Packets",
        "udp_dynamic_params.png": "UDP Dynamic Parameters (Parity & Pacing)",
        "protocol_distribution.png": "Protocol Distribution (Dynamic Mode)",
        "udp_recovery.png": "UDP FEC Recovery Status (Success vs Fail)",
        "rudp_failures.png": "RUDP Failures (Max Retries Reached)",
        "rudp_control_overhead.png": "RUDP Control Overhead (ACKs/NACKs)",
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

    # Helper to add phase switch line
    def add_phase_line(ax, data, x_col="epoch", text_position="top"):
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

    # Define all plot generation tasks
    plot_tasks = [
        (
            "epoch_duration.png",
            lambda: _generate_epoch_duration_plot(df, experiment_name, analysis_dir, add_phase_line, exp_dir),
        ),
        (
            "idle_time_ratio.png",
            lambda: _generate_idle_time_plot(df, experiment_name, analysis_dir, add_phase_line),
        ),
        (
            "wasted_computation.png",
            lambda: _generate_wasted_computation_plot(df, experiment_name, analysis_dir),
        ),
        (
            "survival_rate.png",
            lambda: _generate_survival_rate_plot(df, experiment_name, analysis_dir, add_phase_line),
        ),
        (
            "goodput.png",
            lambda: _generate_goodput_plot(df, experiment_name, analysis_dir),
        ),
        (
            "traffic_volume.png",
            lambda: _generate_traffic_volume_plot(df, experiment_name, analysis_dir),
        ),
        (
            "total_transfer_time.png",
            lambda: _generate_transfer_time_plot(df, experiment_name, analysis_dir),
        ),
        (
            "time_to_accuracy.png",
            lambda: _generate_time_to_accuracy_plot(df, experiment_name, analysis_dir),
        ),
        (
            "accuracy_mean.png",
            lambda: _generate_accuracy_mean_plot(df, experiment_name, analysis_dir, add_phase_line),
        ),
        (
            "loss_mean.png",
            lambda: _generate_loss_mean_plot(df, experiment_name, analysis_dir, add_phase_line),
        ),
        (
            "accuracy_nodes.png",
            lambda: _generate_accuracy_nodes_plot(df, experiment_name, analysis_dir, add_phase_line),
        ),
        (
            "loss_nodes.png",
            lambda: _generate_loss_nodes_plot(df, experiment_name, analysis_dir, add_phase_line),
        ),
        (
            "compute_comm_breakdown.png",
            lambda: _generate_compute_comm_breakdown_plot(df, experiment_name, analysis_dir),
        ),
        # FEC overhead plot (UDP with FEC)
        (
            "fec_overhead.png",
            lambda: _generate_fec_overhead_plot(df, experiment_name, analysis_dir),
        ),
        # RUDP/E-RUDP specific plots
        (
            "rudp_retransmissions.png",
            lambda: _generate_rudp_retransmission_plot(df, experiment_name, analysis_dir, add_phase_line),
        ),
        (
            "rudp_rtt.png",
            lambda: _generate_rudp_rtt_plot(df, experiment_name, analysis_dir, add_phase_line),
        ),
        (
            "rudp_aging.png",
            lambda: _generate_rudp_aging_plot(df, experiment_name, analysis_dir, add_phase_line),
        ),
        # Dynamic Parameters
        (
            "udp_dynamic_params.png",
            lambda: _generate_udp_dynamic_params_plot(df, experiment_name, analysis_dir, add_phase_line),
        ),
        (
            "protocol_distribution.png",
            lambda: _generate_protocol_distribution_plot(df, experiment_name, analysis_dir, add_phase_line),
        ),
        # New Detailed Stats
        (
            "udp_recovery.png",
            lambda: _generate_udp_recovery_plot(df, experiment_name, analysis_dir, add_phase_line),
        ),
        (
            "rudp_failures.png",
            lambda: _generate_rudp_failure_plot(df, experiment_name, analysis_dir, add_phase_line),
        ),
        (
            "rudp_control_overhead.png",
            lambda: _generate_rudp_control_overhead_plot(df, experiment_name, analysis_dir, add_phase_line),
        ),
        (
            "compression_stats.png",
            lambda: _generate_compression_stats_plot(df, experiment_name, analysis_dir, add_phase_line),
        ),
    ]

    # Execute plot generation sequentially
    # Note: ThreadPoolExecutor was removed due to matplotlib global state issues
    # causing corrupted/transparent images in parallel execution
    generated_plots = []
    print(f"  🔧 Generating {len(plot_tasks)} plots...")

    for task_name, task_func in plot_tasks:
        try:
            result = task_func()
            if result:
                generated_plots.append(result)
                print(f"  ✅ Generated {result}")
        except Exception as e:
            print(f"  ❌ Failed to generate {task_name}: {e}")

    # Ensure all figures are closed after plot generation to free memory
    plt.close("all")

    # Generate Markdown report
    print("  📝 Generating Markdown report...")
    _generate_experiment_report(df, experiment_id, experiment_name, analysis_dir, wafl_phase_start_relative)

    print(f"✨ Analysis complete. {len(generated_plots)} plots + report generated in: {analysis_dir}")


# =============================================================================
# Cross-Experiment Comparison Functions
# =============================================================================


def _get_experiment_groups():
    """Group experiments by 'Experiment X:' pattern.

    For Experiment 0, further group by network condition (excellent, good, fair, poor)
    to enable separate comparison graphs for each condition.
    """
    if not RESULTS_DIR.exists():
        return {}

    groups = {}
    for d in RESULTS_DIR.iterdir():
        if d.is_dir() and not d.name.startswith("."):
            # Extract experiment group pattern like "Experiment 1:" or "Experiment 2:"
            match = re.match(r"^(Experiment \d+:)", d.name)
            if match:
                base_group = match.group(1).strip(":")

                # For Experiment 0, further group by network condition
                if base_group == "Experiment 0":
                    # Extract network condition (excellent, good, fair, poor)
                    condition_match = re.match(r"^Experiment 0: (excellent|good|fair|poor)", d.name)
                    if condition_match:
                        condition = condition_match.group(1)
                        group_name = f"Experiment 0 ({condition})"
                    else:
                        # Fallback if condition not found
                        group_name = base_group
                else:
                    group_name = base_group

                if group_name not in groups:
                    groups[group_name] = []
                groups[group_name].append(d.name)

    # Sort experiments within each group
    for group in groups:
        groups[group].sort()

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

    Extracts the distinguishing part from experiment names like:
    - "Experiment 1: Synchronous Scalability Verification - SSP (p=0.6)-20251216T114459"
      → "SSP (p=0.6)"
    - "Experiment 3: Communication and Computation Trade-off Optimization - Adaptive Compression-20251217T031947"
      → "Adaptive Compression"
    """
    # First, remove timestamp suffix
    name = re.sub(r"-\d{8}T\d{6}$", "", experiment_id)

    # Try to extract "Experiment X: Title - VariantName" format
    # Use the last " - " as the separator to handle titles with hyphens
    match = re.match(r"^Experiment \d+: .+ - (.+)$", name)
    if match:
        return match.group(1)

    # Fallback: return the name without timestamp
    return name


def _generate_accuracy_comparison(experiments_data, group_name, output_dir):
    """Generate accuracy comparison plot."""
    plt.figure(figsize=(12, 6))
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
        plt.plot(
            acc_mean["epoch"],
            acc_mean["test_accuracy"],
            color=color,
            linewidth=2,
            label=_get_short_name(exp_id),
        )

    ax = plt.gca()

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

    plt.title(f"Test Accuracy Comparison - {group_name}")
    plt.ylabel("Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylim(0, 1.05)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_comparison.png", dpi=150)
    plt.close()
    return "accuracy_comparison.png"


def _generate_loss_comparison(experiments_data, group_name, output_dir):
    """Generate loss comparison plot."""
    plt.figure(figsize=(12, 6))
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
        plt.plot(
            loss_mean["epoch"],
            loss_mean["test_loss"],
            color=color,
            linewidth=2,
            label=_get_short_name(exp_id),
        )

    ax = plt.gca()

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

    plt.title(f"Test Loss Comparison - {group_name}")
    plt.ylabel("Test Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / "loss_comparison.png", dpi=150)
    plt.close()
    return "loss_comparison.png"


def _generate_duration_comparison(experiments_data, group_name, output_dir):
    """Generate epoch duration comparison plot."""
    plt.figure(figsize=(12, 6))
    phase_switch_epoch = None

    for i, (exp_id, df) in enumerate(experiments_data.items()):
        if df.empty or "epoch_duration_ms" not in df.columns:
            continue

        plot_df = df[df["epoch_duration_ms"].notna()].copy()
        if plot_df.empty:
            continue

        # Get phase switch epoch from first experiment
        if phase_switch_epoch is None and "phase" in df.columns:
            wafl_epochs = df[df["phase"] == "WAFL"]["epoch"]
            if not wafl_epochs.empty:
                phase_switch_epoch = wafl_epochs.min()

        dur_mean = plot_df.groupby("epoch")["epoch_duration_ms"].mean().reset_index()
        dur_mean["duration_s"] = dur_mean["epoch_duration_ms"] / 1000

        color = NODE_PALETTE[i % len(NODE_PALETTE)]
        plt.plot(
            dur_mean["epoch"],
            dur_mean["duration_s"],
            color=color,
            linewidth=2,
            label=_get_short_name(exp_id),
        )

    ax = plt.gca()

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

    plt.title(f"Epoch Duration Comparison - {group_name}")
    plt.ylabel("Duration [sec]")
    plt.xlabel("Epoch")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / "epoch_duration_comparison.png", dpi=150)
    plt.close()
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

    plt.figure(figsize=(12, 6))
    colors = [COLORS["goodput"] if r else COLORS["loss_fill"] for r in tta_df["reached"]]
    bars = plt.bar(tta_df["experiment"], tta_df["time_to_accuracy"], color=colors, alpha=0.8)

    for bar, reached in zip(bars, tta_df["reached"]):
        height = bar.get_height()
        suffix = "" if reached else " (not reached)"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.1f}s{suffix}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.title(f"Time to {TARGET_ACCURACY:.0%} Accuracy (from WAFL Start) - {group_name}")
    plt.ylabel("Time from WAFL Start [sec]")
    plt.xlabel("Experiment")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "time_to_accuracy_comparison.png", dpi=150)
    plt.close()
    return "time_to_accuracy_comparison.png"


def _generate_survival_rate_comparison(experiments_data, group_name, output_dir):
    """Generate survival rate comparison plot."""
    plt.figure(figsize=(12, 6))
    has_data = False

    for i, (exp_id, df) in enumerate(experiments_data.items()):
        if df.empty or "survival_rate" not in df.columns:
            continue

        plot_df = df[df["survival_rate"].notna()].copy()
        if plot_df.empty or (plot_df["survival_rate"] == 1.0).all():
            continue

        has_data = True
        sr_mean = plot_df.groupby("epoch")["survival_rate"].mean().reset_index()

        color = NODE_PALETTE[i % len(NODE_PALETTE)]
        plt.plot(
            sr_mean["epoch"],
            sr_mean["survival_rate"],
            color=color,
            linewidth=2,
            label=_get_short_name(exp_id),
        )

    if not has_data:
        plt.close()
        return None

    plt.title(f"Survival Rate Comparison (UDP/FEC) - {group_name}")
    plt.ylabel("Survival Rate")
    plt.xlabel("Epoch")
    plt.ylim(0, 1.05)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "survival_rate_comparison.png", dpi=150)
    plt.close()
    return "survival_rate_comparison.png"


def _generate_throughput_comparison(experiments_data, group_name, output_dir):
    """Generate throughput comparison plot (WAFL phase only)."""
    plt.figure(figsize=(12, 6))
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
        plt.plot(
            tp_mean["epoch"],
            tp_mean["throughput_mbps"],
            color=color,
            linewidth=2,
            label=_get_short_name(exp_id),
        )

    if not has_data:
        plt.close()
        return None

    plt.title(f"Throughput Comparison - {group_name}")
    plt.ylabel("Throughput [Mbps]")
    plt.xlabel("Epoch")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / "throughput_comparison.png", dpi=150)
    plt.close()
    return "throughput_comparison.png"


def _generate_accuracy_vs_time_comparison(experiments_data, group_name, output_dir):
    """Generate test accuracy vs time comparison plot (WAFL phase, aligned at WAFL start)."""
    plt.figure(figsize=(12, 6))

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
        plt.plot(
            epoch_stats["wafl_relative_timestamp"],
            epoch_stats["test_accuracy"],
            color=color,
            linewidth=2,
            label=_get_short_name(exp_id),
        )

    # Add target accuracy line
    ax = plt.gca()
    ax.axhline(
        y=TARGET_ACCURACY,
        color=COLORS["target_line"],
        linestyle="--",
        linewidth=2,
        label=f"Target ({TARGET_ACCURACY:.0%})",
    )

    plt.title(f"Test Accuracy vs Time (from WAFL Start) - {group_name}")
    plt.ylabel("Test Accuracy")
    plt.xlabel("Time from WAFL Start [sec]")
    plt.ylim(0, 1.05)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_vs_time_comparison.png", dpi=150)
    plt.close()
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

    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        cumulative_df["experiment"],
        cumulative_df["cumulative_mb"],
        color=COLORS["traffic"],
        alpha=0.8,
        edgecolor=COLORS["traffic"],
    )

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.1f} MB",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.title(f"Total Sent Data Comparison - {group_name}")
    plt.ylabel("Cumulative Sent Data [MB]")
    plt.xlabel("Experiment")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "cumulative_sent_data_comparison.png", dpi=150)
    plt.close()
    return "cumulative_sent_data_comparison.png"


def _generate_idle_time_comparison(experiments_data, group_name, output_dir):
    """Generate idle time ratio comparison plot."""
    plt.figure(figsize=(12, 6))
    has_data = False

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
        plt.plot(
            idle_agg["epoch"],
            idle_agg["idle_time_ratio"],
            color=color,
            linewidth=2,
            label=_get_short_name(exp_id),
        )
        has_data = True

    if not has_data:
        plt.close()
        return None

    plt.title(f"Idle Time Ratio Comparison - {group_name}")
    plt.ylabel("Idle Time Ratio")
    plt.xlabel("Epoch")
    plt.ylim(0, 1)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / "idle_time_comparison.png", dpi=150)
    plt.close()
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

    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        wasted_df["experiment"],
        wasted_df["wasted_s"],
        color=COLORS["wasted_bar"],
        alpha=0.8,
    )

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.1f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.title(f"Total Wasted Computation (SSP) - {group_name}")
    plt.ylabel("Wasted Time [sec]")
    plt.xlabel("Experiment")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "wasted_computation_comparison.png", dpi=150)
    plt.close()
    return "wasted_computation_comparison.png"


def _generate_compute_comm_comparison(experiments_data, group_name, output_dir):
    """Generate computation vs communication time comparison plot (averaged over nodes)."""
    comp_data = []

    for exp_id, df in experiments_data.items():
        if df.empty or "compute_time_ms" not in df.columns or "comm_time_ms" not in df.columns:
            continue

        wafl_df = df[df["phase"] == "WAFL"].copy()
        valid_data = wafl_df[(wafl_df["compute_time_ms"].notna()) & (wafl_df["comm_time_ms"].notna())]
        if valid_data.empty:
            continue

        # Average per epoch first, then average over epochs (or just average all)
        # Using simple mean over all data points in WAFL phase
        avg_compute = valid_data["compute_time_ms"].mean() / 1000  # ms to s
        avg_comm = valid_data["comm_time_ms"].mean() / 1000

        comp_data.append(
            {
                "experiment": _get_short_name(exp_id),
                "Computation": avg_compute,
                "Communication": avg_comm,
            }
        )

    if not comp_data:
        return None

    # Transform for Seaborn (melt)
    comp_df = pd.DataFrame(comp_data)

    # Create the stacked bar plot manually since sns doesn't support stacked bars natively easily
    # We plot Total (Comp+Comm) then Comm on top? No, usually Comp bottom, Comm top.
    # Plot Total (Comp+Comm) first, then Computation on top? No that masks it.
    # Plot Total, then Computation.

    comp_df["Total"] = comp_df["Computation"] + comp_df["Communication"]

    plt.figure(figsize=(12, 6))

    # Plot Total (Comm acts as background top part)
    sns.barplot(
        data=comp_df,
        x="experiment",
        y="Total",
        color=COLORS["bar_secondary"],  # Orange for Comm (top)
        label="Communication",
    )

    # Plot Computation (overlay bottom part)
    sns.barplot(
        data=comp_df,
        x="experiment",
        y="Computation",
        color=COLORS["bar_primary"],  # Blue for Comp (bottom)
        label="Computation",
    )

    # Add text labels
    for i, row in comp_df.iterrows():
        # Label Total
        plt.text(i, row["Total"], f"{row['Total']:.1f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")
        # Label components if large enough
        if row["Computation"] > 0.1:
            plt.text(i, row["Computation"] / 2, f"{row['Computation']:.1f}s", ha="center", va="center", color="white", fontsize=8)
        if row["Communication"] > 0.1:
            plt.text(i, row["Computation"] + row["Communication"] / 2, f"{row['Communication']:.1f}s", ha="center", va="center", color="white", fontsize=8)

    plt.title(f"Avg Computation vs Communication Time (WAFL Phase) - {group_name}")
    plt.ylabel("Time [sec]")
    plt.xlabel("Experiment")
    plt.xticks(rotation=45, ha="right")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / "compute_comm_comparison.png", dpi=150)
    plt.close()
    return "compute_comm_comparison.png"


def _generate_goodput_comparison(experiments_data, group_name, output_dir):
    """Generate goodput comparison plot (WAFL phase only)."""
    plt.figure(figsize=(12, 6))
    has_data = False

    for i, (exp_id, df) in enumerate(experiments_data.items()):
        if df.empty or "bytes_received" not in df.columns or "epoch_duration_ms" not in df.columns:
            continue

        wafl_df = df[df["phase"] == "WAFL"].copy()
        goodput_data = wafl_df[(wafl_df["bytes_received"].notna()) & (wafl_df["comm_time_ms"].notna()) & (wafl_df["comm_time_ms"] > 0)].copy()

        # Check for app_bytes_received
        if "app_bytes_received" in goodput_data.columns and goodput_data["app_bytes_received"].sum() > 0:
            rx_col = "app_bytes_received"
        else:
            rx_col = "bytes_received"

        if goodput_data.empty or goodput_data[rx_col].sum() == 0:
            continue

        # Use comm_time_ms for true Goodput (Transfer Throughput)
        goodput_data["goodput_mbps"] = (goodput_data[rx_col] * 8 / 1e6) / (goodput_data["comm_time_ms"] / 1000)
        goodput_agg = goodput_data.groupby("epoch")["goodput_mbps"].mean().reset_index()

        color = NODE_PALETTE[i % len(NODE_PALETTE)]
        plt.plot(
            goodput_agg["epoch"],
            goodput_agg["goodput_mbps"],
            color=color,
            linewidth=2,
            label=_get_short_name(exp_id),
        )
        has_data = True

    if not has_data:
        plt.close()
        return None

    plt.title(f"Goodput Comparison (WAFL Phase) - {group_name}")
    plt.ylabel("Goodput [Mbps]")
    plt.xlabel("Epoch")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / "goodput_comparison.png", dpi=150)
    plt.close()
    return "goodput_comparison.png"


def _generate_traffic_volume_comparison(experiments_data, group_name, output_dir):
    """Generate traffic volume per epoch comparison plot."""
    plt.figure(figsize=(12, 6))
    has_data = False

    for i, (exp_id, df) in enumerate(experiments_data.items()):
        if df.empty or "bytes_sent" not in df.columns:
            continue

        wafl_df = df[df["phase"] == "WAFL"].copy()
        traffic_data = wafl_df[wafl_df["bytes_sent"].notna()]
        if traffic_data.empty or traffic_data["bytes_sent"].sum() == 0:
            continue

        traffic_agg = traffic_data.groupby("epoch").agg({"bytes_sent": "sum"}).reset_index()
        traffic_agg["sent_mb"] = traffic_agg["bytes_sent"] / (1024 * 1024)

        color = NODE_PALETTE[i % len(NODE_PALETTE)]
        plt.plot(
            traffic_agg["epoch"],
            traffic_agg["sent_mb"],
            color=color,
            linewidth=2,
            label=_get_short_name(exp_id),
        )
        has_data = True

    if not has_data:
        plt.close()
        return None

    plt.title(f"Traffic Volume per Epoch (WAFL Phase) - {group_name}")
    plt.ylabel("Sent Data [MB]")
    plt.xlabel("Epoch")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / "traffic_volume_comparison.png", dpi=150)
    plt.close()
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

    sns.set_theme(style="darkgrid")
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
    ax.set_xticklabels(breakdown_df["experiment"], rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "compute_comm_comparison.png", dpi=150)
    plt.close()
    return "compute_comm_comparison.png"


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
    headers1 = ["Experiment", "Final Acc", "Max Acc", "Time to 90%", "Total Epochs", "WAFL Epochs"]
    lines.append("| " + " | ".join(headers1) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers1)) + " |")

    all_metrics = {}
    for exp_id, df in experiments_data.items():
        metrics = _calculate_metrics_summary(df)
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
    headers3 = ["Experiment", "Total Traffic (MB)", "Avg Goodput (Mbps)", "Avg Survival", "RUDP Retrans"]
    lines.append("| " + " | ".join(headers3) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers3)) + " |")

    for exp_id in experiments_data.keys():
        metrics = all_metrics[exp_id]
        short_name = _get_short_name(exp_id)
        row = [short_name]
        row.append(f"{metrics.get('total_traffic_mb', 0):.2f}" if "total_traffic_mb" in metrics else "N/A")
        row.append(f"{metrics.get('avg_goodput_mbps', 0):.2f}" if "avg_goodput_mbps" in metrics else "N/A")
        row.append(f"{metrics.get('avg_survival_rate', 0) * 100:.1f}%" if "avg_survival_rate" in metrics else "N/A")
        row.append(f"{metrics.get('rudp_retransmissions', 0)}" if "rudp_retransmissions" in metrics else "N/A")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")

    # Generated Graphs Section
    lines.append("## Comparison Graphs")
    lines.append("")

    graph_descriptions = {
        "accuracy_comparison.png": "Test Accuracy Comparison",
        "loss_comparison.png": "Test Loss Comparison",
        "epoch_duration_comparison.png": "Epoch Duration Comparison",
        "time_to_accuracy_comparison.png": "Time to Target Accuracy",
        "survival_rate_comparison.png": "Survival Rate Comparison",
        "throughput_comparison.png": "Throughput Comparison",
        "accuracy_vs_time_comparison.png": "Accuracy vs Time Comparison",
        "cumulative_sent_data_comparison.png": "Cumulative Sent Data",
        "idle_time_comparison.png": "Idle Time Comparison",
        "wasted_computation_comparison.png": "Wasted Computation",
        "goodput_comparison.png": "Goodput Comparison",
        "traffic_volume_comparison.png": "Traffic Volume Comparison",
        "compute_comm_comparison.png": "Compute vs Communication Time",
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
            ("epoch_duration_comparison.png", _generate_duration_comparison),
            ("time_to_accuracy_comparison.png", _generate_time_to_accuracy_comparison),
            ("survival_rate_comparison.png", _generate_survival_rate_comparison),
            ("throughput_comparison.png", _generate_throughput_comparison),
            ("accuracy_vs_time_comparison.png", _generate_accuracy_vs_time_comparison),
            (
                "cumulative_sent_data_comparison.png",
                _generate_cumulative_sent_data_comparison,
            ),
            ("idle_time_comparison.png", _generate_idle_time_comparison),
            (
                "wasted_computation_comparison.png",
                _generate_wasted_computation_comparison,
            ),
            ("goodput_comparison.png", _generate_goodput_comparison),
            ("traffic_volume_comparison.png", _generate_traffic_volume_comparison),
            ("compute_comm_comparison.png", _generate_compute_comm_comparison),
        ]

        # Execute comparison plot generation sequentially
        # Note: ThreadPoolExecutor was removed due to matplotlib global state issues
        generated = []
        print(f"  🔧 Generating {len(plots)} comparison plots...")

        for plot_name, plot_func in plots:
            try:
                result = plot_func(experiments_data, group_name, group_dir)
                if result:
                    generated.append(result)
                    print(f"  ✅ Generated {result}")
            except Exception as e:
                print(f"  ❌ Failed to generate {plot_name}: {e}")

        # Generate comparison Markdown report
        print("  📝 Generating comparison report...")
        _generate_comparison_report(experiments_data, group_name, group_dir)

        print(f"  📈 {len(generated)} comparison plots + report generated in: {group_dir}")

    print(f"\n🎉 Comparison complete! Results in: {comparison_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze WAFL experiment results (local)")
    parser.add_argument("--id", help="Experiment ID (default: latest)")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyze all experiments without 'analysis' folder",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate comparison graphs for experiments grouped by 'Experiment X:' pattern",
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Collect results from nodes",
    )
    args = parser.parse_args()

    # Note: This script now performs LOCAL analysis only.
    # Use mise.toml tasks or rsync manually to collect data from management server.

    if args.collect:
        config = load_config()
        targets = []
        if args.id:
            targets = [args.id]
        elif args.all:
            targets = get_all_experiments()
        else:
            # Default behavior for --collect: Collect ALL experiments to ensure updates
            # Verify/update existing data even if analyzed
            print("ℹ️  No ID specified. Checking ALL experiments for new data...")
            targets = get_all_experiments()

        if not targets:
            print("✨ No experiments need collection.")
        else:
            print(f"📥 Collecting results for {len(targets)} experiments...")
            for exp_id in targets:
                collect_results(exp_id, config)

    if args.compare:
        # Cross-experiment comparison mode
        compare_experiments()
    elif args.all:
        # Find all experiments and re-analyze them
        experiments = get_all_experiments()
        if not experiments:
            print("❌ No experiments found.")
            return

        print(f"📋 Found {len(experiments)} experiments to analyze:")
        for exp_id in experiments:
            print(f"   - {exp_id}")
        print()

        # Process each experiment, removing existing analysis first
        import shutil

        for i, exp_id in enumerate(experiments):
            print(f"\n{'=' * 60}")
            print(f"[{i + 1}/{len(experiments)}] Processing: {exp_id}")
            print(f"{'=' * 60}")

            # Remove existing analysis folder to force re-generation
            analysis_dir = RESULTS_DIR / exp_id / "analysis"
            if analysis_dir.exists():
                shutil.rmtree(analysis_dir)
                print("  🗑️ Removed existing analysis folder")

            analyze_results(exp_id)

        print(f"\n🎉 All {len(experiments)} experiments processed!")
    else:
        # Single experiment mode
        exp_id = args.id or get_latest_experiment_id()
        if not exp_id:
            print("❌ No experiment ID found.")
            sys.exit(1)
        analyze_results(exp_id)


if __name__ == "__main__":
    main()

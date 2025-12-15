import argparse
import json
import os
import re
import sys
from concurrent.futures import (  # Used in collect_results
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

        files_to_collect = ["metrics.csv", "resources.csv", "model.pth"]
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


def _generate_epoch_duration_plot(df, experiment_name, analysis_dir, add_phase_line_func):
    """Generate wall-clock time per epoch plot."""
    if df.empty or "epoch_duration_ms" not in df.columns:
        return None

    epoch_dur = df[df["epoch_duration_ms"].notna()].copy()
    if epoch_dur.empty:
        return None

    epoch_dur["epoch_duration_s"] = epoch_dur["epoch_duration_ms"] / 1000
    num_nodes = epoch_dur["node"].nunique()
    palette = NODE_PALETTE[:num_nodes] if num_nodes <= len(NODE_PALETTE) else "husl"

    plt.figure(figsize=(12, 6))
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
    sns.lineplot(
        data=epoch_dur,
        x="epoch",
        y="epoch_duration_s",
        color=COLORS["mean_line"],
        linewidth=2.5,
        label="Mean",
        errorbar=None,
        ax=ax,
    )
    add_phase_line_func(ax, df, "epoch")
    plt.title(f"Wall-clock Time per Epoch - {experiment_name}")
    plt.ylabel("Duration [sec]")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(analysis_dir / "epoch_duration.png", dpi=150)
    plt.close()
    return "epoch_duration.png"


def _generate_idle_time_plot(df, experiment_name, analysis_dir):
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
    ax.bar(
        idle_agg["epoch"],
        idle_agg["idle_time_ratio"] * 100,
        color=COLORS["idle"],
        alpha=0.8,
        edgecolor=COLORS["idle"],
        linewidth=0.5,
    )
    ax.set_ylabel("Idle Time Ratio [%]", color=COLORS["mean_line"])
    ax.set_xlabel("Epoch")
    ax.tick_params(axis="y", labelcolor=COLORS["mean_line"])
    ax.set_ylim(0, 100)
    plt.title(f"Idle Time Ratio (Sync Wait Time) - {experiment_name}")
    plt.tight_layout()
    plt.savefig(analysis_dir / "idle_time_ratio.png", dpi=150)
    plt.close()
    return "idle_time_ratio.png"


def _generate_wasted_computation_plot(df, experiment_name, analysis_dir):
    """Generate wasted computation plot."""
    has_wasted_data = False
    if not df.empty and "wasted_ms" in df.columns:
        ssp_data = df[(df["wasted_ms"].notna()) & (df["wasted_ms"] > 0)].copy()
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
            )
            ax1.set_ylabel("Wasted Time [sec]", color=COLORS["wasted_bar"])
            ax1.tick_params(axis="y", labelcolor=COLORS["wasted_bar"])
            ax1.set_xlabel("Epoch")

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
            color="#666666",
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
    """Generate survival rate plot."""
    plt.figure(figsize=(12, 6))
    has_meaningful_survival_data = False

    if not df.empty and "survival_rate" in df.columns:
        udp_data = df[df["survival_rate"].notna()].copy()
        udp_used = False
        if "bytes_sent" in df.columns and "bytes_received" in df.columns:
            udp_used = (df["bytes_sent"].sum() > 0) or (df["bytes_received"].sum() > 0)

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
            add_phase_line_func(ax, df, "epoch")

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
            color="#666666",
            transform=ax.transAxes,
        )
        ax.axis("off")
        plt.title(f"Survival Rate (UDP/FEC) - {experiment_name}")
    plt.tight_layout()
    plt.savefig(analysis_dir / "survival_rate.png", dpi=150)
    plt.close()
    return "survival_rate.png"


def _generate_goodput_plot(df, experiment_name, analysis_dir):
    """Generate goodput plot showing both sent and received throughput."""
    plt.figure(figsize=(12, 6))
    has_meaningful_goodput_data = False

    if not df.empty and "bytes_received" in df.columns and "bytes_sent" in df.columns and "epoch_duration_ms" in df.columns:
        goodput_data = df[(df["bytes_received"].notna()) & (df["bytes_sent"].notna()) & (df["epoch_duration_ms"].notna())].copy()
        if not goodput_data.empty and (goodput_data["bytes_received"].sum() > 0 or goodput_data["bytes_sent"].sum() > 0):
            has_meaningful_goodput_data = True
            # Calculate throughput in Mbps
            goodput_data["goodput_mbps"] = (goodput_data["bytes_received"] * 8 / 1e6) / (goodput_data["epoch_duration_ms"] / 1000)
            goodput_data["sent_mbps"] = (goodput_data["bytes_sent"] * 8 / 1e6) / (goodput_data["epoch_duration_ms"] / 1000)

            goodput_agg = goodput_data.groupby("epoch").agg({"goodput_mbps": "mean", "sent_mbps": "mean"}).reset_index()

            ax = plt.subplot(1, 1, 1)
            # Plot sent throughput (total)
            ax.plot(
                goodput_agg["epoch"],
                goodput_agg["sent_mbps"],
                color=COLORS["bar_secondary"],
                linewidth=2,
                marker="s",
                markersize=3,
                label="Sent Throughput",
                alpha=0.7,
            )
            # Plot received (goodput)
            ax.plot(
                goodput_agg["epoch"],
                goodput_agg["goodput_mbps"],
                color=COLORS["goodput"],
                linewidth=2.5,
                marker="o",
                markersize=4,
                label="Goodput (Received)",
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
            color="#666666",
            transform=ax.transAxes,
        )
        ax.axis("off")
        plt.title(f"Goodput (Effective Throughput) - {experiment_name}")
    plt.tight_layout()
    plt.savefig(analysis_dir / "goodput.png", dpi=150)
    plt.close()
    return "goodput.png"


def _generate_traffic_volume_plot(df, experiment_name, analysis_dir):
    """Generate traffic volume plot."""
    plt.figure(figsize=(12, 6))
    has_meaningful_traffic_data = False

    if not df.empty and "bytes_sent" in df.columns:
        traffic_data = df[df["bytes_sent"].notna()].copy()
        if not traffic_data.empty and traffic_data["bytes_sent"].sum() > 0:
            has_meaningful_traffic_data = True
            traffic_agg = traffic_data.groupby("epoch").agg({"bytes_sent": "sum"}).reset_index()
            traffic_agg["sent_mb"] = traffic_agg["bytes_sent"] / (1024 * 1024)

            ax = plt.subplot(1, 1, 1)
            ax.bar(
                traffic_agg["epoch"],
                traffic_agg["sent_mb"],
                color=COLORS["traffic"],
                alpha=0.8,
                edgecolor=COLORS["traffic"],
                linewidth=0.5,
            )
            ax.set_ylabel("Sent Data [MB]")
            ax.set_xlabel("Epoch")
            plt.title(f"Traffic Volume - {experiment_name}")

    if not has_meaningful_traffic_data:
        ax = plt.gca()
        ax.text(
            0.5,
            0.5,
            "No data transfer recorded\n(UDP not enabled or no model sharing)",
            ha="center",
            va="center",
            fontsize=14,
            color="#666666",
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
            ax1.bar(
                transfer_agg["epoch"],
                transfer_agg["epoch_duration_s"],
                color=COLORS["bar_primary"],
                alpha=0.8,
                label="Epoch Duration",
                edgecolor=COLORS["bar_primary"],
                linewidth=0.5,
            )
            ax1.set_ylabel("Epoch Duration [sec]")
            ax1.legend(loc="upper right")
            ax1.set_title(f"Total Transfer Time Breakdown - {experiment_name}")

            # Bottom: Compression Time (in ms for visibility)
            ax2.bar(
                transfer_agg["epoch"],
                transfer_agg["compression_ms"],
                color=COLORS["bar_secondary"],
                alpha=0.9,
                label="Compression Time (T_comp)",
                edgecolor=COLORS["bar_secondary"],
                linewidth=0.5,
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
        color="#666666",
        transform=ax.transAxes,
    )
    ax.axis("off")
    plt.title(f"Total Transfer Time (T_comm + T_comp) - {experiment_name}")
    plt.tight_layout()
    plt.savefig(analysis_dir / "total_transfer_time.png", dpi=150)
    plt.close()
    return "total_transfer_time.png"


def _generate_cpu_usage_plot(resources_df, experiment_name, analysis_dir):
    """Generate CPU usage plot."""
    plt.figure(figsize=(12, 6))
    if not resources_df.empty and "cpu_percent" in resources_df.columns:
        num_nodes = resources_df["node"].nunique()
        palette = NODE_PALETTE[:num_nodes] if num_nodes <= len(NODE_PALETTE) else "husl"

        ax = sns.lineplot(
            data=resources_df,
            x="timestamp",
            y="cpu_percent",
            hue="node",
            alpha=0.4,
            legend=False,
            palette=palette,
            estimator=None,
            linewidth=1.2,
        )
        mean_cpu = resources_df.groupby("timestamp")["cpu_percent"].mean().reset_index()
        sns.lineplot(
            data=mean_cpu,
            x="timestamp",
            y="cpu_percent",
            color=COLORS["mean_line"],
            linewidth=2.5,
            label="Mean",
            ax=ax,
        )
        ax.set_ylim(0, 100)
        plt.title(f"CPU Usage - {experiment_name}")
        plt.ylabel("CPU Usage [%]")
        plt.xlabel("Elapsed Time [sec]")
        plt.legend()
    else:
        plt.text(
            0.5,
            0.5,
            "No CPU usage data available",
            ha="center",
            va="center",
            fontsize=14,
            color="#666666",
        )
        plt.title(f"CPU Usage - {experiment_name}")
    plt.tight_layout()
    plt.savefig(analysis_dir / "cpu_usage.png", dpi=150)
    plt.close()
    return "cpu_usage.png"


def _generate_time_to_accuracy_plot(df, experiment_name, analysis_dir):
    """Generate time-to-accuracy plot."""
    if df.empty or "test_accuracy" not in df.columns or "timestamp" not in df.columns:
        return None

    acc_data = df[df["test_accuracy"].notna()].copy()
    if acc_data.empty:
        return None

    plt.figure(figsize=(12, 6))
    epoch_stats = acc_data.groupby("epoch").agg({"timestamp": "mean", "test_accuracy": ["mean", "std", "min", "max"]})
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

    plt.title(f"Time-to-Accuracy (Target: {TARGET_ACCURACY:.0%}) - {experiment_name}")
    plt.ylabel("Test Accuracy")
    plt.xlabel("Elapsed Time [sec]")
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


def analyze_results(experiment_id):
    """Analyze collected results and generate plots in parallel."""
    print(f"📊 Analyzing results for: {experiment_id}")

    exp_dir = RESULTS_DIR / experiment_id
    analysis_dir = exp_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    # Extract experiment name without timestamp
    timestamp_pattern = r"-\d{8}T\d{6}$"
    experiment_name = re.sub(timestamp_pattern, "", experiment_id)

    # Load data
    df, resources_df = _load_metrics_and_resources(exp_dir)

    if df.empty:
        print("⚠️  No metrics data found.")
        return

    # Set theme
    sns.set_theme(style="darkgrid")

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

        ax.axvline(
            x=wafl_start,
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
            wafl_start,
            y_pos,
            " Phase Switch",
            color=COLORS["phase_line"],
            va=va,
            fontsize=10,
            fontweight="bold",
        )

    # Define all plot generation tasks
    plot_tasks = [
        (
            "accuracy.png",
            lambda: _generate_accuracy_plot(df, experiment_name, analysis_dir, add_phase_line),
        ),
        (
            "epoch_duration.png",
            lambda: _generate_epoch_duration_plot(df, experiment_name, analysis_dir, add_phase_line),
        ),
        (
            "idle_time_ratio.png",
            lambda: _generate_idle_time_plot(df, experiment_name, analysis_dir),
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
            "cpu_usage.png",
            lambda: _generate_cpu_usage_plot(resources_df, experiment_name, analysis_dir),
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
    ]

    # Execute plot generation sequentially
    generated_plots = []
    for task_name, task_func in plot_tasks:
        try:
            result = task_func()
            if result:
                generated_plots.append(result)
                print(f"  ✅ Generated {result}")
        except Exception as e:
            print(f"  ❌ Failed to generate {task_name}: {e}")

    print(f"✨ Analysis complete. {len(generated_plots)} plots generated in: {analysis_dir}")


# =============================================================================
# Cross-Experiment Comparison Functions
# =============================================================================


def _get_experiment_groups():
    """Group experiments by 'Experiment X:' pattern."""
    if not RESULTS_DIR.exists():
        return {}

    groups = {}
    for d in RESULTS_DIR.iterdir():
        if d.is_dir() and not d.name.startswith("."):
            # Extract experiment group pattern like "Experiment 1:" or "Experiment 2:"
            match = re.match(r"^(Experiment \d+:)", d.name)
            if match:
                group_name = match.group(1).strip(":")
                if group_name not in groups:
                    groups[group_name] = []
                groups[group_name].append(d.name)

    # Sort experiments within each group
    for group in groups:
        groups[group].sort()

    return groups


def _load_experiment_data(experiment_id):
    """Load metrics data for an experiment."""
    exp_dir = RESULTS_DIR / experiment_id
    df, _ = _load_metrics_and_resources(exp_dir)
    return df


def _get_short_name(experiment_id):
    """Extract short name from experiment ID for legend."""
    # Remove "Experiment X: Title - " and timestamp
    match = re.match(r"^Experiment \d+: [^-]+ - (.+)-\d{8}T\d{6}$", experiment_id)
    if match:
        return match.group(1)
    # Fallback: remove timestamp
    return re.sub(r"-\d{8}T\d{6}$", "", experiment_id)


def _generate_accuracy_comparison(experiments_data, group_name, output_dir):
    """Generate accuracy comparison plot."""
    plt.figure(figsize=(14, 7))

    for i, (exp_id, df) in enumerate(experiments_data.items()):
        if df.empty or "test_accuracy" not in df.columns:
            continue

        plot_df = df[df["test_accuracy"].notna()].copy()
        if "is_ssp_interrupted" in plot_df.columns:
            plot_df = plot_df[~plot_df["is_ssp_interrupted"]]

        if plot_df.empty:
            continue

        acc_mean = plot_df.groupby("epoch")["test_accuracy"].mean().reset_index()

        color = NODE_PALETTE[i % len(NODE_PALETTE)]
        plt.plot(
            acc_mean["epoch"],
            acc_mean["test_accuracy"],
            color=color,
            linewidth=2,
            label=_get_short_name(exp_id),
            marker="o",
            markersize=3,
        )

    plt.title(f"Test Accuracy Comparison - {group_name}")
    plt.ylabel("Test Accuracy")
    plt.xlabel("Epoch")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_comparison.png", dpi=150)
    plt.close()
    return "accuracy_comparison.png"


def _generate_loss_comparison(experiments_data, group_name, output_dir):
    """Generate loss comparison plot."""
    plt.figure(figsize=(14, 7))

    for i, (exp_id, df) in enumerate(experiments_data.items()):
        if df.empty or "test_loss" not in df.columns:
            continue

        plot_df = df[df["test_loss"].notna()].copy()
        if "is_ssp_interrupted" in plot_df.columns:
            plot_df = plot_df[~plot_df["is_ssp_interrupted"]]

        if plot_df.empty:
            continue

        loss_mean = plot_df.groupby("epoch")["test_loss"].mean().reset_index()

        color = NODE_PALETTE[i % len(NODE_PALETTE)]
        plt.plot(
            loss_mean["epoch"],
            loss_mean["test_loss"],
            color=color,
            linewidth=2,
            label=_get_short_name(exp_id),
            marker="o",
            markersize=3,
        )

    plt.title(f"Test Loss Comparison - {group_name}")
    plt.ylabel("Test Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_comparison.png", dpi=150)
    plt.close()
    return "loss_comparison.png"


def _generate_duration_comparison(experiments_data, group_name, output_dir):
    """Generate epoch duration comparison plot."""
    plt.figure(figsize=(14, 7))

    for i, (exp_id, df) in enumerate(experiments_data.items()):
        if df.empty or "epoch_duration_ms" not in df.columns:
            continue

        plot_df = df[df["epoch_duration_ms"].notna()].copy()
        if plot_df.empty:
            continue

        dur_mean = plot_df.groupby("epoch")["epoch_duration_ms"].mean().reset_index()
        dur_mean["duration_s"] = dur_mean["epoch_duration_ms"] / 1000

        color = NODE_PALETTE[i % len(NODE_PALETTE)]
        plt.plot(
            dur_mean["epoch"],
            dur_mean["duration_s"],
            color=color,
            linewidth=2,
            label=_get_short_name(exp_id),
            marker="o",
            markersize=3,
        )

    plt.title(f"Epoch Duration Comparison - {group_name}")
    plt.ylabel("Duration [sec]")
    plt.xlabel("Epoch")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "epoch_duration_comparison.png", dpi=150)
    plt.close()
    return "epoch_duration_comparison.png"


def _generate_time_to_accuracy_comparison(experiments_data, group_name, output_dir):
    """Generate time-to-accuracy bar chart comparison."""
    tta_data = []

    for exp_id, df in experiments_data.items():
        if df.empty or "test_accuracy" not in df.columns or "timestamp" not in df.columns:
            continue

        plot_df = df[df["test_accuracy"].notna()].copy()
        if "is_ssp_interrupted" in plot_df.columns:
            plot_df = plot_df[~plot_df["is_ssp_interrupted"]]

        if plot_df.empty:
            continue

        epoch_stats = plot_df.groupby("epoch").agg({"test_accuracy": "mean", "timestamp": "max"}).reset_index()
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

    plt.title(f"Time to {TARGET_ACCURACY:.0%} Accuracy - {group_name}")
    plt.ylabel("Time [sec]")
    plt.xlabel("Experiment")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "time_to_accuracy_comparison.png", dpi=150)
    plt.close()
    return "time_to_accuracy_comparison.png"


def _generate_survival_rate_comparison(experiments_data, group_name, output_dir):
    """Generate survival rate comparison plot."""
    plt.figure(figsize=(14, 7))
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
            marker="o",
            markersize=3,
        )

    if not has_data:
        plt.close()
        return None

    plt.title(f"Survival Rate Comparison (UDP/FEC) - {group_name}")
    plt.ylabel("Survival Rate")
    plt.xlabel("Epoch")
    plt.ylim(0, 1.05)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "survival_rate_comparison.png", dpi=150)
    plt.close()
    return "survival_rate_comparison.png"


def _generate_throughput_comparison(experiments_data, group_name, output_dir):
    """Generate throughput comparison plot."""
    plt.figure(figsize=(14, 7))
    has_data = False

    for i, (exp_id, df) in enumerate(experiments_data.items()):
        if df.empty or "bytes_received" not in df.columns or "epoch_duration_ms" not in df.columns:
            continue

        plot_df = df[(df["bytes_received"].notna()) & (df["epoch_duration_ms"].notna())].copy()
        if plot_df.empty or plot_df["bytes_received"].sum() == 0:
            continue

        has_data = True
        plot_df["throughput_mbps"] = (plot_df["bytes_received"] * 8 / 1e6) / (plot_df["epoch_duration_ms"] / 1000)
        tp_mean = plot_df.groupby("epoch")["throughput_mbps"].mean().reset_index()

        color = NODE_PALETTE[i % len(NODE_PALETTE)]
        plt.plot(
            tp_mean["epoch"],
            tp_mean["throughput_mbps"],
            color=color,
            linewidth=2,
            label=_get_short_name(exp_id),
            marker="o",
            markersize=3,
        )

    if not has_data:
        plt.close()
        return None

    plt.title(f"Throughput Comparison - {group_name}")
    plt.ylabel("Throughput [Mbps]")
    plt.xlabel("Epoch")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "throughput_comparison.png", dpi=150)
    plt.close()
    return "throughput_comparison.png"


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
        ]

        generated = []
        for plot_name, plot_func in plots:
            try:
                result = plot_func(experiments_data, group_name, group_dir)
                if result:
                    generated.append(result)
                    print(f"  ✅ Generated {result}")
            except Exception as e:
                print(f"  ❌ Failed to generate {plot_name}: {e}")

        print(f"  📈 {len(generated)} comparison plots generated in: {group_dir}")

    print(f"\n🎉 Comparison complete! Results in: {comparison_dir}")


def main():
    parser = argparse.ArgumentParser(description="Collect and analyze WAFL results")
    parser.add_argument("--id", help="Experiment ID (default: latest)")
    parser.add_argument("--skip-collect", action="store_true", help="Skip collection, only analyze")
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
    args = parser.parse_args()

    config = load_config()

    if args.compare:
        # Cross-experiment comparison mode
        compare_experiments()
    elif args.all:
        # Find all experiments without analysis folder
        experiments = get_experiments_without_analysis()
        if not experiments:
            print("✅ All experiments already have analysis folders.")
            return

        print(f"📋 Found {len(experiments)} experiments without analysis folder:")
        for exp_id in experiments:
            print(f"   - {exp_id}")
        print()

        # Process each experiment
        for i, exp_id in enumerate(experiments):
            print(f"\n{'=' * 60}")
            print(f"[{i + 1}/{len(experiments)}] Processing: {exp_id}")
            print(f"{'=' * 60}")

            if not args.skip_collect:
                collect_results(exp_id, config)

            analyze_results(exp_id)

        print(f"\n🎉 All {len(experiments)} experiments processed!")
    else:
        # Single experiment mode
        exp_id = args.id or get_latest_experiment_id()
        if not exp_id:
            print("❌ No experiment ID found.")
            sys.exit(1)

        if not args.skip_collect:
            collect_results(exp_id, config)

        analyze_results(exp_id)


if __name__ == "__main__":
    main()

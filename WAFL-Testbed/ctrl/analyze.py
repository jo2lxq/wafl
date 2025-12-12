import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import paramiko
import seaborn as sns

# Japanese font configuration for matplotlib
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CTRL_DIR = PROJECT_ROOT / "ctrl"
CONFIG_FILE = CTRL_DIR / "execution_config.json"
RESULTS_DIR = PROJECT_ROOT / "results"

# Target accuracy for Time-to-Accuracy plot
TARGET_ACCURACY = 0.95


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


def analyze_results(experiment_id):
    """Analyze collected results and generate plots."""
    print(f"📊 Analyzing results for: {experiment_id}")

    exp_dir = RESULTS_DIR / experiment_id
    analysis_dir = exp_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    # Extract experiment name without timestamp (e.g., "exp1-20251211T144430" -> "exp1")
    # The timestamp format is "-YYYYMMDDTHHMMSS" at the end
    timestamp_pattern = r"-\d{8}T\d{6}$"
    experiment_name = re.sub(timestamp_pattern, "", experiment_id)

    # 1. Load Metrics
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

    if not metrics_dfs:
        print("⚠️  No metrics data found.")
        return

    df = pd.concat(metrics_dfs)

    # 2. Load Resources
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

    # Set theme
    sns.set_theme(style="darkgrid")

    # --- Helper to add phase switch line ---
    def add_phase_line(ax, x_col="epoch", text_position="top"):
        wafl_start = df[df["phase"] == "WAFL"][x_col].min()
        if not pd.isna(wafl_start):
            ax.axvline(x=wafl_start, color="firebrick", linestyle="--", alpha=0.7)
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
                color="firebrick",
                va=va,
                fontsize=10,
            )

    # ==========================================================================
    # 1. Accuracy
    # ==========================================================================
    if not df.empty and "test_accuracy" in df.columns:
        plt.figure(figsize=(12, 6))
        ax = sns.lineplot(
            data=df,
            x="epoch",
            y="test_accuracy",
            hue="node",
            alpha=0.3,
            legend=False,
            palette="viridis",
            estimator=None,
        )
        sns.lineplot(
            data=df,
            x="epoch",
            y="test_accuracy",
            color="navy",
            linewidth=2,
            label="Mean",
            errorbar=None,
            ax=ax,
        )
        add_phase_line(ax, "epoch", text_position="bottom")
        plt.title(f"Test Accuracy - {experiment_name}")
        plt.ylabel("Test Accuracy")
        plt.xlabel("Epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(analysis_dir / "accuracy.png", dpi=150)
        plt.close()
        print("  ✅ Generated accuracy.png")

    # ==========================================================================
    # 2. Wall-clock time per epoch
    # ==========================================================================
    if not df.empty and "epoch_duration_ms" in df.columns:
        epoch_dur = df[df["epoch_duration_ms"].notna()].copy()
        if not epoch_dur.empty:
            # Convert to seconds for readability
            epoch_dur["epoch_duration_s"] = epoch_dur["epoch_duration_ms"] / 1000

            plt.figure(figsize=(12, 6))
            ax = sns.lineplot(
                data=epoch_dur,
                x="epoch",
                y="epoch_duration_s",
                hue="node",
                alpha=0.3,
                legend=False,
                palette="viridis",
                estimator=None,
            )
            sns.lineplot(
                data=epoch_dur,
                x="epoch",
                y="epoch_duration_s",
                color="navy",
                linewidth=2,
                label="Mean",
                errorbar=None,
                ax=ax,
            )
            add_phase_line(ax, "epoch")
            plt.title(f"Wall-clock Time per Epoch - {experiment_name}")
            plt.ylabel("Duration [sec]")
            plt.xlabel("Epoch")
            plt.legend()
            plt.tight_layout()
            plt.savefig(analysis_dir / "epoch_duration.png", dpi=150)
            plt.close()
            print("  ✅ Generated epoch_duration.png")

    # ==========================================================================
    # 3. Idle Time Ratio
    # ==========================================================================
    # Calculate idle time as: max_epoch_duration - node_epoch_duration
    if not df.empty and "epoch_duration_ms" in df.columns:
        epoch_dur = df[df["epoch_duration_ms"].notna()].copy()
        if not epoch_dur.empty:
            # For each epoch, calculate max duration and idle time
            max_dur_per_epoch = epoch_dur.groupby("epoch")["epoch_duration_ms"].max().reset_index()
            max_dur_per_epoch.columns = ["epoch", "max_duration_ms"]
            epoch_dur = epoch_dur.merge(max_dur_per_epoch, on="epoch")
            epoch_dur["idle_time_ms"] = epoch_dur["max_duration_ms"] - epoch_dur["epoch_duration_ms"]
            epoch_dur["idle_time_ratio"] = epoch_dur["idle_time_ms"] / epoch_dur["max_duration_ms"]

            # Aggregate per epoch
            idle_agg = epoch_dur.groupby("epoch").agg({"idle_time_ratio": "mean", "idle_time_ms": "sum"}).reset_index()

            plt.figure(figsize=(12, 6))
            ax = plt.subplot(1, 1, 1)
            ax.bar(
                idle_agg["epoch"],
                idle_agg["idle_time_ratio"] * 100,
                color="steelblue",
                alpha=0.7,
            )
            ax.set_ylabel("Idle Time Ratio [%]", color="steelblue")
            ax.set_xlabel("Epoch")
            ax.tick_params(axis="y", labelcolor="steelblue")
            ax.set_ylim(0, 100)
            plt.title(f"Idle Time Ratio (Sync Wait Time) - {experiment_name}")
            plt.tight_layout()
            plt.savefig(analysis_dir / "idle_time_ratio.png", dpi=150)
            plt.close()
            print("  ✅ Generated idle_time_ratio.png")

    # ==========================================================================
    # 4. Wasted Computation
    # ==========================================================================
    # Only show wasted computation when SSP force-skips actually occurred
    has_wasted_data = False
    if not df.empty and "wasted_ms" in df.columns:
        # Filter only epochs where force-skip occurred (wasted_ms > 0)
        ssp_data = df[(df["wasted_ms"].notna()) & (df["wasted_ms"] > 0)].copy()
        if not ssp_data.empty:
            has_wasted_data = True
            wasted_per_epoch = ssp_data.groupby("epoch").agg({"wasted_ms": "sum", "batches_processed": "sum"}).reset_index()
            wasted_per_epoch["wasted_s"] = wasted_per_epoch["wasted_ms"] / 1000

            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax1.bar(
                wasted_per_epoch["epoch"],
                wasted_per_epoch["wasted_s"],
                color="coral",
                alpha=0.7,
                label="Wasted Time",
            )
            ax1.set_ylabel("Wasted Time [sec]", color="coral")
            ax1.tick_params(axis="y", labelcolor="coral")
            ax1.set_xlabel("Epoch")

            ax2 = ax1.twinx()
            ax2.plot(
                wasted_per_epoch["epoch"],
                wasted_per_epoch["batches_processed"],
                color="navy",
                linewidth=2,
                marker="o",
                markersize=3,
                label="Incomplete Batches",
            )
            ax2.set_ylabel("Incomplete Batches (before force-skip)", color="navy")
            ax2.tick_params(axis="y", labelcolor="navy")

            plt.title(f"Wasted Computation (SSP Force-Skip) - {experiment_name}")
            fig.tight_layout()
            plt.savefig(analysis_dir / "wasted_computation.png", dpi=150)
            plt.close()
            print("  ✅ Generated wasted_computation.png")

    if not has_wasted_data:
        # No wasted data available, create placeholder plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(
            0.5,
            0.5,
            "No wasted computation data\n(SSP not enabled or no force-skips occurred)",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        plt.title(f"Wasted Computation (SSP) - {experiment_name}")
        plt.tight_layout()
        plt.savefig(analysis_dir / "wasted_computation.png", dpi=150)
        plt.close()
        print("  ✅ Generated wasted_computation.png (no SSP data)")

    # ==========================================================================
    # 5. Survival Rate
    # ==========================================================================
    # Always generate this plot
    plt.figure(figsize=(12, 6))
    has_meaningful_survival_data = False
    if not df.empty and "survival_rate" in df.columns:
        udp_data = df[df["survival_rate"].notna()].copy()
        # Check if UDP was actually used (bytes_sent or bytes_received > 0)
        # or if survival_rate varies (indicating UDP with packet loss)
        udp_used = False
        if "bytes_sent" in df.columns and "bytes_received" in df.columns:
            udp_used = (df["bytes_sent"].sum() > 0) or (df["bytes_received"].sum() > 0)

        has_variation = udp_data["survival_rate"].std() > 0.001 if not udp_data.empty else False

        if not udp_data.empty and (has_variation or udp_used):
            has_meaningful_survival_data = True
            ax = sns.lineplot(
                data=udp_data,
                x="epoch",
                y="survival_rate",
                hue="node",
                alpha=0.3,
                legend=False,
                palette="viridis",
                estimator=None,
            )
            sns.lineplot(
                data=udp_data,
                x="epoch",
                y="survival_rate",
                color="navy",
                linewidth=2,
                label="Mean",
                errorbar=None,
                ax=ax,
            )
            ax.set_ylim(0, 1.05)
            add_phase_line(ax, "epoch")

            # Add informative subtitle if all values are 1.0
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
            transform=ax.transAxes,
        )
        ax.axis("off")
        plt.title(f"Survival Rate (UDP/FEC) - {experiment_name}")
    plt.tight_layout()
    plt.savefig(analysis_dir / "survival_rate.png", dpi=150)
    plt.close()
    print("  ✅ Generated survival_rate.png")

    # ==========================================================================
    # 6. Goodput
    # ==========================================================================
    # Always generate this plot
    plt.figure(figsize=(12, 6))
    has_meaningful_goodput_data = False
    if not df.empty and "bytes_received" in df.columns and "epoch_duration_ms" in df.columns:
        goodput_data = df[(df["bytes_received"].notna()) & (df["epoch_duration_ms"].notna())].copy()
        # Check if there's actual data (not all zeros)
        if not goodput_data.empty and goodput_data["bytes_received"].sum() > 0:
            has_meaningful_goodput_data = True
            goodput_data["goodput_mbps"] = (goodput_data["bytes_received"] * 8 / 1e6) / (goodput_data["epoch_duration_ms"] / 1000)
            goodput_agg = goodput_data.groupby("epoch")["goodput_mbps"].mean().reset_index()

            ax = plt.subplot(1, 1, 1)
            ax.plot(
                goodput_agg["epoch"],
                goodput_agg["goodput_mbps"],
                color="green",
                linewidth=2,
                marker="o",
                markersize=3,
            )
            ax.fill_between(
                goodput_agg["epoch"],
                goodput_agg["goodput_mbps"],
                alpha=0.3,
                color="green",
            )
            ax.set_ylabel("Goodput [Mbps]")
            ax.set_xlabel("Epoch")
            plt.title(f"Goodput (Effective Throughput) - {experiment_name}")

    if not has_meaningful_goodput_data:
        ax = plt.gca()
        ax.text(
            0.5,
            0.5,
            "No data transfer recorded\n(UDP not enabled or no model sharing)",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.axis("off")
        plt.title(f"Goodput (Effective Throughput) - {experiment_name}")
    plt.tight_layout()
    plt.savefig(analysis_dir / "goodput.png", dpi=150)
    plt.close()
    print("  ✅ Generated goodput.png")

    # ==========================================================================
    # 7. Traffic Volume
    # ==========================================================================
    # Always generate this plot
    plt.figure(figsize=(12, 6))
    has_meaningful_traffic_data = False
    if not df.empty and "bytes_sent" in df.columns:
        traffic_data = df[df["bytes_sent"].notna()].copy()
        # Check if there's actual data (not all zeros)
        if not traffic_data.empty and traffic_data["bytes_sent"].sum() > 0:
            has_meaningful_traffic_data = True
            # Aggregate total traffic per epoch
            traffic_agg = traffic_data.groupby("epoch").agg({"bytes_sent": "sum"}).reset_index()
            traffic_agg["sent_mb"] = traffic_agg["bytes_sent"] / (1024 * 1024)

            ax = plt.subplot(1, 1, 1)
            ax.bar(traffic_agg["epoch"], traffic_agg["sent_mb"], color="teal", alpha=0.7)
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
            transform=ax.transAxes,
        )
        ax.axis("off")
        plt.title(f"Traffic Volume - {experiment_name}")
    plt.tight_layout()
    plt.savefig(analysis_dir / "traffic_volume.png", dpi=150)
    plt.close()
    print("  ✅ Generated traffic_volume.png")

    # ==========================================================================
    # 8. Total Transfer Time (T_comm + T_comp)
    # ==========================================================================
    # Always generate this plot
    plt.figure(figsize=(12, 6))
    has_meaningful_compression_data = False
    if not df.empty and "compression_time_ms" in df.columns and "epoch_duration_ms" in df.columns:
        transfer_data = df[(df["compression_time_ms"].notna()) & (df["epoch_duration_ms"].notna())].copy()
        # Check if there's actual compression data (not all zeros)
        if not transfer_data.empty and transfer_data["compression_time_ms"].sum() > 0:
            has_meaningful_compression_data = True
            transfer_data["total_transfer_s"] = transfer_data["epoch_duration_ms"] / 1000
            transfer_data["compression_s"] = transfer_data["compression_time_ms"] / 1000

            transfer_agg = transfer_data.groupby("epoch").agg({"total_transfer_s": "mean", "compression_s": "mean"}).reset_index()

            ax = plt.subplot(1, 1, 1)
            ax.bar(
                transfer_agg["epoch"],
                transfer_agg["total_transfer_s"],
                color="steelblue",
                alpha=0.7,
                label="Epoch Duration",
            )
            ax.bar(
                transfer_agg["epoch"],
                transfer_agg["compression_s"],
                color="coral",
                alpha=0.9,
                label="Compression Time (T_comp)",
            )
            ax.set_ylabel("Time [sec]")
            ax.set_xlabel("Epoch")
            ax.legend()
            plt.title(f"Total Transfer Time (T_comm + T_comp) - {experiment_name}")

    if not has_meaningful_compression_data:
        ax = plt.gca()
        ax.text(
            0.5,
            0.5,
            "Compression not enabled\n(compression_time_ms = 0)",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.axis("off")
        plt.title(f"Total Transfer Time (T_comm + T_comp) - {experiment_name}")
    plt.tight_layout()
    plt.savefig(analysis_dir / "total_transfer_time.png", dpi=150)
    plt.close()
    print("  ✅ Generated total_transfer_time.png")

    # ==========================================================================
    # 9. CPU Usage
    # ==========================================================================
    plt.figure(figsize=(12, 6))
    if not resources_df.empty and "cpu_percent" in resources_df.columns:
        ax = sns.lineplot(
            data=resources_df,
            x="timestamp",
            y="cpu_percent",
            hue="node",
            alpha=0.3,
            legend=False,
            palette="viridis",
            estimator=None,
        )
        mean_cpu = resources_df.groupby("timestamp")["cpu_percent"].mean().reset_index()
        sns.lineplot(
            data=mean_cpu,
            x="timestamp",
            y="cpu_percent",
            color="navy",
            linewidth=2,
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
        )
        plt.title(f"CPU Usage - {experiment_name}")
    plt.tight_layout()
    plt.savefig(analysis_dir / "cpu_usage.png", dpi=150)
    plt.close()
    print("  ✅ Generated cpu_usage.png")

    # ==========================================================================
    # 10. Target Accuracy Reached Time (Time-to-Accuracy)
    # ==========================================================================
    if not df.empty and "test_accuracy" in df.columns and "timestamp" in df.columns:
        acc_data = df[df["test_accuracy"].notna()].copy()
        if not acc_data.empty:
            plt.figure(figsize=(12, 6))

            # Calculate statistics per epoch
            epoch_stats = acc_data.groupby("epoch").agg({"timestamp": "mean", "test_accuracy": ["mean", "std", "min", "max"]})
            epoch_stats.columns = ["timestamp", "mean", "std", "min", "max"]
            epoch_stats = epoch_stats.reset_index()
            epoch_stats["std"] = epoch_stats["std"].fillna(0)

            ax = plt.gca()

            # Plot confidence band (mean ± std)
            ax.fill_between(
                epoch_stats["timestamp"],
                epoch_stats["mean"] - epoch_stats["std"],
                epoch_stats["mean"] + epoch_stats["std"],
                alpha=0.3,
                color="steelblue",
                label="Mean ± SD",
            )

            # Plot mean line
            ax.plot(
                epoch_stats["timestamp"],
                epoch_stats["mean"],
                color="navy",
                linewidth=2.5,
                label="Mean",
                marker="o",
                markersize=3,
            )

            # Target line
            ax.axhline(
                y=TARGET_ACCURACY,
                color="green",
                linestyle="--",
                linewidth=2,
                label=f"Target ({TARGET_ACCURACY:.0%})",
            )

            # Find time to target (first epoch where mean exceeds target)
            reached_epochs = epoch_stats[epoch_stats["mean"] >= TARGET_ACCURACY]
            if not reached_epochs.empty:
                first_reach = reached_epochs["timestamp"].iloc[0]
                ax.axvline(
                    x=first_reach,
                    color="red",
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
            print("  ✅ Generated time_to_accuracy.png")

    # ==========================================================================
    # Mean accuracy/loss plots
    # ==========================================================================
    acc_df = df.melt(
        id_vars=["epoch", "phase", "node"],
        value_vars=["train_accuracy", "test_accuracy"],
        var_name="metric",
        value_name="value",
    )
    loss_df = df.melt(
        id_vars=["epoch", "phase", "node"],
        value_vars=["train_loss", "test_loss"],
        var_name="metric",
        value_name="value",
    )

    if not acc_df.empty:
        plt.figure(figsize=(12, 6))
        ax = sns.lineplot(
            data=acc_df,
            x="epoch",
            y="value",
            hue="metric",
            markers=False,
            dashes=False,
            errorbar="sd",
        )
        add_phase_line(ax, "epoch", text_position="bottom")
        plt.title(f"Accuracy over Epochs (Mean +/- SD) - {experiment_name}")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.tight_layout()
        plt.savefig(analysis_dir / "accuracy_mean.png", dpi=150)
        plt.close()
        print("  ✅ Generated accuracy_mean.png")

    if not loss_df.empty:
        plt.figure(figsize=(12, 6))
        ax = sns.lineplot(
            data=loss_df,
            x="epoch",
            y="value",
            hue="metric",
            markers=False,
            dashes=False,
            errorbar="sd",
        )
        add_phase_line(ax, "epoch")
        plt.title(f"Loss over Epochs (Mean +/- SD) - {experiment_name}")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.tight_layout()
        plt.savefig(analysis_dir / "loss_mean.png", dpi=150)
        plt.close()
        print("  ✅ Generated loss_mean.png")

    # Node-wise plots
    if not df.empty:
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=df,
            x="epoch",
            y="test_accuracy",
            hue="node",
            alpha=0.3,
            legend=False,
            palette="viridis",
            estimator=None,
        )
        ax = sns.lineplot(
            data=df,
            x="epoch",
            y="test_accuracy",
            color="navy",
            linewidth=2,
            label="Mean",
            errorbar=None,
        )
        add_phase_line(ax, "epoch", text_position="bottom")
        plt.title(f"Node-wise Test Accuracy - {experiment_name}")
        plt.ylabel("Test Accuracy")
        plt.xlabel("Epoch")
        plt.tight_layout()
        plt.savefig(analysis_dir / "accuracy_nodes.png", dpi=150)
        plt.close()
        print("  ✅ Generated accuracy_nodes.png")

        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=df,
            x="epoch",
            y="test_loss",
            hue="node",
            alpha=0.3,
            legend=False,
            palette="viridis",
            estimator=None,
        )
        ax = sns.lineplot(
            data=df,
            x="epoch",
            y="test_loss",
            color="navy",
            linewidth=2,
            label="Mean",
            errorbar=None,
        )
        add_phase_line(ax, "epoch")
        plt.title(f"Node-wise Test Loss - {experiment_name}")
        plt.ylabel("Test Loss")
        plt.xlabel("Epoch")
        plt.tight_layout()
        plt.savefig(analysis_dir / "loss_nodes.png", dpi=150)
        plt.close()
        print("  ✅ Generated loss_nodes.png")

    print(f"✨ Analysis complete. Results in: {analysis_dir}")


def main():
    parser = argparse.ArgumentParser(description="Collect and analyze WAFL results")
    parser.add_argument("--id", help="Experiment ID (default: latest)")
    parser.add_argument("--skip-collect", action="store_true", help="Skip collection, only analyze")
    args = parser.parse_args()

    config = load_config()

    exp_id = args.id or get_latest_experiment_id()
    if not exp_id:
        print("❌ No experiment ID found.")
        sys.exit(1)

    if not args.skip_collect:
        collect_results(exp_id, config)

    analyze_results(exp_id)


if __name__ == "__main__":
    main()

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import paramiko
import seaborn as sns

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CTRL_DIR = PROJECT_ROOT / "ctrl"
CONFIG_FILE = CTRL_DIR / "execution_config.json"
RESULTS_DIR = PROJECT_ROOT / "results"


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

    # 1. Load Metrics
    metrics_dfs = []
    for node_dir in exp_dir.iterdir():
        if node_dir.is_dir() and node_dir.name != "analysis":
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

    # 2. Plot Training/Test Accuracy & Loss
    sns.set_theme(style="darkgrid")

    # Melt dataframe for easier plotting with seaborn
    acc_df = df.melt(id_vars=["epoch", "phase", "node"], value_vars=["train_accuracy", "test_accuracy"], var_name="metric", value_name="value")
    loss_df = df.melt(id_vars=["epoch", "phase", "node"], value_vars=["train_loss", "test_loss"], var_name="metric", value_name="value")

    # --- Helper to add phase switch line ---
    def add_phase_line(ax, x_col="epoch"):
        # Find start of WAFL phase
        wafl_start = df[df["phase"] == "WAFL"][x_col].min()
        if not pd.isna(wafl_start):
            ax.axvline(x=wafl_start, color="firebrick", linestyle="--", alpha=0.7)
            # Place text slightly to the right of the line, at the top
            y_min, y_max = ax.get_ylim()
            ax.text(wafl_start, y_max * 0.15, " Phase Switch", color="firebrick", va="bottom")

    # --- Aggregated Plots (Mean + SD) ---
    # optimization: use errorbar='sd' (standard deviation) instead of bootstrap CI (slow)
    if not acc_df.empty:
        plt.figure(figsize=(12, 6))
        ax = sns.lineplot(data=acc_df, x="epoch", y="value", hue="metric", style="phase", markers=False, dashes=False, errorbar="sd")
        add_phase_line(ax, "epoch")
        plt.title(f"Accuracy over Epochs (Mean ± SD) - {experiment_id}")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.savefig(analysis_dir / "accuracy_mean.png")
        plt.close()
        print("  ✅ Generated accuracy_mean.png")

    if not loss_df.empty:
        plt.figure(figsize=(12, 6))
        ax = sns.lineplot(data=loss_df, x="epoch", y="value", hue="metric", style="phase", markers=False, dashes=False, errorbar="sd")
        add_phase_line(ax, "epoch")
        plt.title(f"Loss over Epochs (Mean ± SD) - {experiment_id}")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.savefig(analysis_dir / "loss_mean.png")
        plt.close()
        print("  ✅ Generated loss_mean.png")

    # --- Node-wise Plots (Spaghetti Plots) ---
    # Useful for identifying outliers or stragglers
    # optimization: estimator=None avoids expensive aggregation
    if not df.empty:
        # Node-wise Test Accuracy
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x="epoch", y="test_accuracy", hue="node", alpha=0.3, legend=False, palette="viridis", estimator=None)
        # Changed mean line color to navy (softer than black)
        ax = sns.lineplot(data=df, x="epoch", y="test_accuracy", color="navy", linewidth=2, label="Mean", errorbar=None)
        add_phase_line(ax, "epoch")
        plt.title(f"Node-wise Test Accuracy - {experiment_id}")
        plt.ylabel("Test Accuracy")
        plt.xlabel("Epoch")
        plt.savefig(analysis_dir / "accuracy_nodes.png")
        plt.close()
        print("  ✅ Generated accuracy_nodes.png")

        # Node-wise Test Loss
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x="epoch", y="test_loss", hue="node", alpha=0.3, legend=False, palette="viridis", estimator=None)
        # Changed mean line color to navy
        ax = sns.lineplot(data=df, x="epoch", y="test_loss", color="navy", linewidth=2, label="Mean", errorbar=None)
        add_phase_line(ax, "epoch")
        plt.title(f"Node-wise Test Loss - {experiment_id}")
        plt.ylabel("Test Loss")
        plt.xlabel("Epoch")
        plt.savefig(analysis_dir / "loss_nodes.png")
        plt.close()
        print("  ✅ Generated loss_nodes.png")

    # 3. Resource Usage Analysis
    resources_data = []
    for node_dir in exp_dir.iterdir():
        if node_dir.is_dir() and node_dir.name != "analysis":
            res_file = node_dir / "resources.csv"
            if res_file.exists():
                try:
                    rdf = pd.read_csv(res_file)
                    rdf["node"] = node_dir.name
                    # Normalize timestamp relative to start
                    if not rdf.empty:
                        # Timestamp is already relative in new implementation
                        resources_data.append(rdf)
                except Exception:
                    pass

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

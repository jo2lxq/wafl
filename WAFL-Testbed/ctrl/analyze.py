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
    """Find the latest experiment directory in results/."""
    if not RESULTS_DIR.exists():
        return None
    # List all directories in results/
    dirs = [d for d in RESULTS_DIR.iterdir() if d.is_dir()]
    if not dirs:
        return None
    # Return the one with the latest modification time
    return max(dirs, key=lambda d: d.stat().st_mtime).name


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
    # We want to plot train_accuracy and test_accuracy
    acc_df = df.melt(id_vars=["epoch", "phase", "node"], value_vars=["train_accuracy", "test_accuracy"], var_name="metric", value_name="value")
    loss_df = df.melt(id_vars=["epoch", "phase", "node"], value_vars=["train_loss", "test_loss"], var_name="metric", value_name="value")

    if not acc_df.empty:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=acc_df, x="epoch", y="value", hue="metric", style="phase", markers=True, dashes=False)
        plt.title(f"Accuracy over Epochs ({experiment_id})")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.savefig(analysis_dir / "accuracy_curve.png")
        plt.close()
        print("  ✅ Generated accuracy_curve.png")

    if not loss_df.empty:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=loss_df, x="epoch", y="value", hue="metric", style="phase", markers=True, dashes=False)
        plt.title(f"Loss over Epochs ({experiment_id})")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.savefig(analysis_dir / "loss_curve.png")
        plt.close()
        print("  ✅ Generated loss_curve.png")

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

    if resources_data:
        res_df = pd.concat(resources_data)

        # CPU Usage
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=res_df, x="timestamp", y="cpu_percent", hue="node", alpha=0.5, legend=False)
        # Add mean line
        sns.lineplot(data=res_df, x="timestamp", y="cpu_percent", color="red", linewidth=2, label="Mean")
        plt.title(f"CPU Usage over Time ({experiment_id})")
        plt.ylabel("CPU (%)")
        plt.xlabel("Time (s)")
        plt.savefig(analysis_dir / "cpu_usage.png")
        plt.close()
        print("  ✅ Generated cpu_usage.png")

        # Memory Usage
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=res_df, x="timestamp", y="memory_used_mb", hue="node", alpha=0.5, legend=False)
        sns.lineplot(data=res_df, x="timestamp", y="memory_used_mb", color="red", linewidth=2, label="Mean")
        plt.title(f"Memory Usage over Time ({experiment_id})")
        plt.ylabel("Memory (MB)")
        plt.xlabel("Time (s)")
        plt.savefig(analysis_dir / "memory_usage.png")
        plt.close()
        print("  ✅ Generated memory_usage.png")

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

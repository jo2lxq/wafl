import argparse
import os
import sys
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

"""
WAFL Emulation Testbed: Analysis and Visualization Program

This script aggregates and visualizes the learning results from all devices
for a specific experiment. It is designed to work with `learning_data.csv` files
containing the columns: `epoch`, `train_acc`, `train_loss`, `test_acc`, `test_loss`.

# Prerequisites:
1.  The results for the experiment have already been collected using `collect.py`.
2.  Python libraries `pandas`, `seaborn`, and `matplotlib` are installed.

# Usage:
# To analyze a specific experiment:
$ python ctrl/analyze.py [experiment_id]

# To analyze the latest experiment automatically:
$ python ctrl/analyze.py
"""


def find_latest_experiment(results_path: str) -> Optional[str]:
    """
    Finds the most recently modified directory (experiment) in the results folder.

    Args:
        results_path: The path to the 'results' directory.

    Returns:
        The name of the latest experiment directory, or None if not found.
    """
    if not os.path.isdir(results_path):
        return None

    all_dirs = [d for d in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, d))]

    if not all_dirs:
        return None

    latest_dir = max(all_dirs, key=lambda d: os.path.getmtime(os.path.join(results_path, d)))
    return latest_dir


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="📊 Analyze and visualize results from a WAFL experiment.", formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "experiment_id",
        nargs="?",
        default=None,
        help="ID of the experiment to analyze.\nIf omitted, the latest experiment will be used.",
    )
    parser.add_argument("--results_dir", default="./results", help="Path to the main results directory.")
    return parser.parse_args()


def load_and_transform_data(experiment_path: str) -> Optional[pd.DataFrame]:
    """
    Finds `learning_data.csv` files, loads them, and transforms them from
    wide format to long format for easier plotting.

    Args:
        experiment_path: Path to the specific experiment's result directory.

    Returns:
        A single pandas DataFrame with aggregated and transformed data.
    """
    all_data = []
    print(f"📂 Searching for device data in: {experiment_path}")

    if not os.path.isdir(experiment_path):
        print("❌ Error: Experiment directory not found.", file=sys.stderr)
        return None

    for device_id_str in os.listdir(experiment_path):
        device_path = os.path.join(experiment_path, device_id_str)
        if os.path.isdir(device_path) and device_id_str.isdigit():
            csv_path = os.path.join(device_path, "learning_data.csv")
            if os.path.exists(csv_path):
                try:
                    # Read the wide-format CSV
                    df_wide = pd.read_csv(csv_path)

                    # Transform train data into long format
                    df_train = df_wide[["epoch", "train_acc", "train_loss"]].copy()
                    df_train.rename(columns={"train_acc": "accuracy", "train_loss": "loss"}, inplace=True)
                    df_train["phase"] = "train"

                    # Transform test data into long format
                    df_test = df_wide[["epoch", "test_acc", "test_loss"]].copy()
                    df_test.rename(columns={"test_acc": "accuracy", "test_loss": "loss"}, inplace=True)
                    df_test["phase"] = "test"

                    # Combine into a single long-format DataFrame
                    df_long = pd.concat([df_train, df_test], ignore_index=True)
                    df_long["device_id"] = int(device_id_str)
                    all_data.append(df_long)
                    print(f"  - Loaded & reshaped data for Device #{device_id_str}")
                except Exception as e:
                    print(f"  - ⚠️ Warning: Could not process {csv_path}. Reason: {e}")

    if not all_data:
        print("❌ Error: No valid `learning_data.csv` files were found.", file=sys.stderr)
        return None

    return pd.concat(all_data, ignore_index=True)


def plot_learning_curves(df: pd.DataFrame, output_dir: str, experiment_id: str):
    """
    Generates and saves various insightful learning curve plots.
    """
    print("\n📊 Generating plots...")
    sns.set_theme(style="whitegrid")

    # === Plot 1 & 2: Average Learning Curves (Accuracy & Loss) ===
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    fig.suptitle(f"Average Learning Curves (Experiment: {experiment_id})", fontsize=16)

    sns.lineplot(ax=axes[0], data=df, x="epoch", y="accuracy", hue="phase", errorbar="sd")
    axes[0].set_title("Average Accuracy vs. Epoch")
    axes[0].set_ylabel("Accuracy")

    sns.lineplot(ax=axes[1], data=df, x="epoch", y="loss", hue="phase", errorbar="sd")
    axes[1].set_title("Average Loss vs. Epoch")
    axes[1].set_ylabel("Loss")

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    save_path = os.path.join(output_dir, "1_average_curves.png")
    plt.savefig(save_path)
    plt.close()
    print(f"  📈 Saved average learning curves to: {save_path}")

    # === Plot 3: Individual Device Test Accuracy ===
    df_test = df[df["phase"] == "test"]
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df_test, x="epoch", y="accuracy", hue="device_id", palette="viridis_r", legend="full")
    plt.title(f"Individual Device Test Accuracy vs. Epoch\n(Experiment: {experiment_id})")
    plt.ylabel("Test Accuracy")
    plt.legend(title="Device ID", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    save_path = os.path.join(output_dir, "2_individual_test_accuracy.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  🎨 Saved individual accuracy plot to: {save_path}")

    # === Plot 4: Final Test Accuracy Distribution (Box Plot) ===
    df_final_epoch = df_test.loc[df_test.groupby("device_id")["epoch"].idxmax()]
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df_final_epoch, y="accuracy", color="skyblue")
    sns.stripplot(data=df_final_epoch, y="accuracy", color="black", size=5, jitter=0.1)
    plt.title(f"Distribution of Final Test Accuracy Across Devices\n(Experiment: {experiment_id})")
    plt.ylabel("Final Test Accuracy")
    plt.xlabel("All Devices")
    plt.tight_layout()
    save_path = os.path.join(output_dir, "3_final_accuracy_distribution.png")
    plt.savefig(save_path)
    plt.close()
    print(f"  📊 Saved final accuracy distribution plot to: {save_path}")

    # === Plot 5: Learning Speed vs. Final Performance (Scatter Plot) ===
    # Epochs to reach 80% of max accuracy
    max_acc_overall = df_final_epoch["accuracy"].max()
    threshold = max_acc_overall * 0.90
    epochs_to_threshold = df_test[df_test["accuracy"] >= threshold].groupby("device_id")["epoch"].min().reset_index()
    epochs_to_threshold.rename(columns={"epoch": "epochs_to_reach_90pct_max"}, inplace=True)

    df_final_merged = pd.merge(df_final_epoch, epochs_to_threshold, on="device_id", how="left")

    plt.figure(figsize=(12, 7))
    sns.scatterplot(
        data=df_final_merged,
        x="epochs_to_reach_90pct_max",
        y="accuracy",
        hue="device_id",
        palette="viridis_r",
        s=100,
        legend=False,
    )
    plt.title(f"Learning Speed vs. Final Performance\n(Experiment: {experiment_id})")
    plt.xlabel(f"Epochs to Reach {threshold:.2f} Accuracy")
    plt.ylabel("Final Test Accuracy")
    plt.tight_layout()
    save_path = os.path.join(output_dir, "4_speed_vs_performance.png")
    plt.savefig(save_path)
    plt.close()
    print(f"  Scatter plot of learning speed vs performance saved to: {save_path}")


def main():
    """
    Main execution function.
    """
    args = parse_arguments()

    print("🚀 --- Starting Analysis Program ---")

    experiment_id = args.experiment_id
    if experiment_id is None:
        print("🔍 Experiment ID not specified. Searching for the latest...")
        experiment_id = find_latest_experiment(args.results_dir)
        if experiment_id is None:
            print(f"❌ Error: No experiment directories found in {args.results_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"✅ Found latest experiment: '{experiment_id}'")

    experiment_path = os.path.join(args.results_dir, experiment_id)

    aggregated_df = load_and_transform_data(experiment_path)

    if aggregated_df is None or aggregated_df.empty:
        print("\n❌ Analysis aborted due to missing data.")
        return

    output_summary_dir = os.path.join(experiment_path, "summary")
    os.makedirs(output_summary_dir, exist_ok=True)
    print(f"\n✅ Data loaded successfully! Found {len(aggregated_df)} total records.")
    print(f"🖼️  Plots will be saved to: {output_summary_dir}")

    plot_learning_curves(aggregated_df, output_summary_dir, experiment_id)

    print("\n🎉 --- Analysis complete! ---")


if __name__ == "__main__":
    main()

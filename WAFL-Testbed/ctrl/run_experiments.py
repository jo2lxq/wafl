#!/usr/bin/env python3
"""
Batch experiment runner for WAFL-Testbed.

This script runs experiments sequentially using parameter files from ctrl/parameters/.
For each parameter file:
1. Updates ctrl/parameters.json with the parameter file content
2. Updates ctrl/execution_config.json with the experiment_name from the parameter file
3. Executes ctrl/main.py as a subprocess

Usage:
    python ctrl/run_experiments.py                      # Run all experiments
    python ctrl/run_experiments.py --list               # List experiments without running
    python ctrl/run_experiments.py --select "exp1_*"    # Run only matching experiments
    python ctrl/run_experiments.py --dry-run            # Show what would be executed
"""

import argparse
import fnmatch
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def get_script_dir() -> Path:
    """Get the directory where this script is located."""
    return Path(__file__).parent


def get_parameter_files(parameters_dir: Path, pattern: Optional[str] = None) -> List[Path]:
    """
    Get all JSON parameter files from the parameters directory.

    Args:
        parameters_dir: Path to the parameters directory
        pattern: Optional glob pattern to filter files (e.g., "exp1_*")

    Returns:
        Sorted list of parameter file paths
    """
    if not parameters_dir.exists():
        print(f"❌ Parameters directory not found: {parameters_dir}")
        return []

    files = sorted(parameters_dir.glob("*.json"))

    if pattern:
        files = [f for f in files if fnmatch.fnmatch(f.name, pattern)]

    return files


def load_json(file_path: Path) -> dict:
    """Load JSON file and return as dictionary."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(file_path: Path, data: dict) -> None:
    """Save dictionary to JSON file with pretty formatting."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def update_execution_config(config_path: Path, experiment_name: str) -> None:
    """
    Update the experiment_name in execution_config.json.

    Args:
        config_path: Path to execution_config.json
        experiment_name: New experiment name to set
    """
    config = load_json(config_path)
    config["experiment_name"] = experiment_name
    save_json(config_path, config)


def run_experiment(
    param_file: Path,
    parameters_json: Path,
    execution_config: Path,
    dry_run: bool = False,
) -> bool:
    """
    Run a single experiment.

    Args:
        param_file: Path to the parameter file
        parameters_json: Path to ctrl/parameters.json
        execution_config: Path to ctrl/execution_config.json
        dry_run: If True, don't actually run the experiment

    Returns:
        True if successful, False otherwise
    """
    # Load parameter file
    try:
        params = load_json(param_file)
    except Exception as e:
        print(f"❌ Failed to load {param_file.name}: {e}")
        return False

    experiment_name = params.get("experiment_name", param_file.stem)

    print(f"\n{'=' * 60}")
    print(f"📋 Experiment: {param_file.name}")
    print(f"📝 Name: {experiment_name}")
    print(f"{'=' * 60}")

    if dry_run:
        print("🔸 [DRY-RUN] Would execute this experiment")
        return True

    try:
        # Step 1: Copy parameter file to parameters.json
        print(f"📄 Updating {parameters_json.name}...")
        shutil.copy2(param_file, parameters_json)

        # Step 2: Update execution_config.json with experiment_name
        print(f"⚙️  Updating experiment_name in {execution_config.name}...")
        update_execution_config(execution_config, experiment_name)

        # Step 3: Run main.py
        print("🚀 Starting experiment...")
        ctrl_dir = get_script_dir()
        project_root = ctrl_dir.parent

        result = subprocess.run(
            [sys.executable, "-u", str(ctrl_dir / "main.py")],
            cwd=str(project_root),
            check=False,
        )

        if result.returncode == 0:
            print(f"✅ Experiment completed successfully: {param_file.name}")
            return True
        else:
            print(f"❌ Experiment failed with return code {result.returncode}: {param_file.name}")
            return False

    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user")
        raise
    except Exception as e:
        print(f"❌ Error running experiment: {e}")
        return False


def list_experiments(files: List[Path]) -> None:
    """List all experiment files with their names."""
    print(f"\n📋 Available experiments ({len(files)} files):\n")

    for i, f in enumerate(files, 1):
        try:
            params = load_json(f)
            name = params.get("experiment_name", "N/A")
            epochs_self = params.get("epochs", {}).get("self", "?")
            epochs_wafl = params.get("epochs", {}).get("wafl", "?")
            print(f"  {i:2}. {f.name}")
            print(f"      └─ {name}")
            print(f"         (SELF: {epochs_self}, WAFL: {epochs_wafl} epochs)")
        except Exception as e:
            print(f"  {i:2}. {f.name} [Error: {e}]")


def main():
    parser = argparse.ArgumentParser(
        description="Run WAFL experiments sequentially from parameter files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiments without running them",
    )
    parser.add_argument(
        "--select",
        type=str,
        metavar="PATTERN",
        help="Run only experiments matching the pattern (e.g., 'exp1_*')",
    )
    parser.add_argument(
        "--skip",
        type=str,
        action="append",
        metavar="FILENAME",
        help="Skip specified experiment file(s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without actually running",
    )

    args = parser.parse_args()

    # Setup paths
    ctrl_dir = get_script_dir()
    parameters_dir = ctrl_dir / "parameters"
    parameters_json = ctrl_dir / "parameters.json"
    execution_config = ctrl_dir / "execution_config.json"

    # Verify required files exist
    if not parameters_json.exists():
        print(f"❌ Parameters file not found: {parameters_json}")
        sys.exit(1)
    if not execution_config.exists():
        print(f"❌ Execution config not found: {execution_config}")
        sys.exit(1)

    # Get parameter files
    files = get_parameter_files(parameters_dir, args.select)

    # Apply skip filter
    if args.skip:
        files = [f for f in files if f.name not in args.skip]

    if not files:
        print("❌ No parameter files found to run")
        sys.exit(1)

    # List mode
    if args.list:
        list_experiments(files)
        sys.exit(0)

    # Run experiments
    print("\n🔬 WAFL Batch Experiment Runner")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Found {len(files)} experiment(s) to run")

    if args.dry_run:
        print("🔸 Running in DRY-RUN mode (no actual execution)")

    results = {"success": [], "failed": []}

    try:
        for i, param_file in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] Processing {param_file.name}...")

            success = run_experiment(
                param_file,
                parameters_json,
                execution_config,
                dry_run=args.dry_run,
            )

            if success:
                results["success"].append(param_file.name)
            else:
                results["failed"].append(param_file.name)

    except KeyboardInterrupt:
        print("\n\n⚠️  Batch execution interrupted by user")

    # Summary
    print(f"\n{'=' * 60}")
    print("📊 Experiment Batch Summary")
    print(f"{'=' * 60}")
    print(f"✅ Successful: {len(results['success'])}")
    for name in results["success"]:
        print(f"   • {name}")
    print(f"❌ Failed: {len(results['failed'])}")
    for name in results["failed"]:
        print(f"   • {name}")
    print(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if results["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

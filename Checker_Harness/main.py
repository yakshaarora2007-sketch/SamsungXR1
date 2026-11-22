#!/usr/bin/env python3
"""
main.py - Reproducible Trajectory Degradation Experiment Harness CLI.
Samsung XR Hackathon Track 3: Path Finder.

Commands:
  python main.py --sequence MH01 --new-experiment
  python main.py --sequence MH01 --generate-segment-only
  python main.py --experiment <id> --run-degradation
  python main.py --experiment <id> --rerun-degradation
  python main.py --list-experiments
  python main.py --experiment <id> --info
"""

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

# Ensure current directory and Degrader are in sys.path
_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from config.settings import HarnessConfig
from dataset.loader import MissingDatasetError
from experiment.manager import ExperimentManager
from experiment.runner import ExperimentRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Reproducible Trajectory Degradation Experiment Harness (Samsung XR Hackathon Track 3)",
    )

    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--new-experiment",
        action="store_true",
        help="Stage A & B: Extract a new 15-second segment and execute all 30 degradation runs (10 T1, 10 T2, 10 T3)",
    )
    action_group.add_argument(
        "--generate-segment-only",
        action="store_true",
        help="Stage A only: Extract and save a new 15-second segment without executing degradation",
    )
    action_group.add_argument(
        "--run-degradation",
        action="store_true",
        help="Stage B only: Execute degradation runs on an existing saved segment",
    )
    action_group.add_argument(
        "--rerun-degradation",
        action="store_true",
        help="Rerun degradation on an existing experiment, creating a new batch version while preserving all prior runs",
    )
    action_group.add_argument(
        "--list-experiments",
        action="store_true",
        help="List all existing experiments and their execution status",
    )
    action_group.add_argument(
        "--info",
        action="store_true",
        help="Display detailed metadata, segment information, and run status for an experiment",
    )

    # Context options
    parser.add_argument(
        "--sequence",
        type=str,
        default="MH01",
        help="Sequence identifier (default: MH01, e.g. MH_01_easy, V1_01_easy)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        dest="experiment_id",
        help="Experiment ID or directory name (required for --run-degradation, --rerun-degradation, --info)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to EuRoC dataset root directory (defaults to config or dataset/euroc)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to root output directory for experiments (default: outputs)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Master random seed for reproducible segment selection and degradation",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config.yaml file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show full stack traces on error",
    )

    return parser


def handle_list_experiments(manager: ExperimentManager):
    """Lists all stored experiments."""
    experiments = manager.list_experiments()
    if not experiments:
        print("\nNo experiments found in outputs directory.")
        print("To create an experiment, run:")
        print("  python main.py --sequence MH01 --new-experiment\n")
        return

    print("\n" + "=" * 95)
    print(f"{'EXPERIMENT ID':<35} {'SEQUENCE':<15} {'STATUS':<20} {'RUNS (T1/T2/T3)':<15}")
    print("=" * 95)
    for exp in experiments:
        exp_id = exp.get("experiment_id", "")
        seq = exp.get("sequence", "N/A")
        status = exp.get("status", "N/A")
        runs_str = f"{exp.get('t1_runs', 0)}/{exp.get('t2_runs', 0)}/{exp.get('t3_runs', 0)}"
        print(f"{exp_id:<35} {seq:<15} {status:<20} {runs_str:<15}")
    print("=" * 95 + "\n")


def handle_experiment_info(manager: ExperimentManager, experiment_id: str):
    """Displays detailed information for a specific experiment."""
    if not experiment_id:
        print("Error: --experiment <experiment_id> is required for --info", file=sys.stderr)
        sys.exit(1)

    exp_dir = manager.get_experiment_dir(experiment_id)
    meta = manager.load_experiment_metadata(experiment_id)

    print("\n" + "=" * 70)
    print(f"EXPERIMENT DETAILS: {meta.get('experiment_id')}")
    print("=" * 70)
    print(f"Directory:       {exp_dir.resolve()}")
    print(f"Created At:      {meta.get('creation_timestamp')}")
    print(f"Status:          {meta.get('status')}")
    print(f"Sequence:        {meta.get('sequence_name')}")
    print(f"Dataset Path:    {meta.get('original_dataset_path')}")
    print(f"Segment Span:    {meta.get('segment_start_timestamp_ns')} -> {meta.get('segment_end_timestamp_ns')}")
    print(f"Nominal Dur:     {meta.get('segment_duration_s')}s (Actual: {meta.get('actual_duration_s')}s)")
    print(f"Selection Seed:  {meta.get('segment_selection_seed')}")

    targets = meta.get("runs_target", {})
    t1_runs = list((exp_dir / "T1").glob("run_*"))
    t2_runs = list((exp_dir / "T2").glob("run_*"))
    t3_runs = list((exp_dir / "T3").glob("run_*"))
    print("\nDegradation Progress:")
    print(f"  T1 (Checkpoints):     {len(t1_runs)} / {targets.get('T1', 10)}")
    print(f"  T2 (IMU Blackout):    {len(t2_runs)} / {targets.get('T2', 10)}")
    print(f"  T3 (Degradation):     {len(t3_runs)} / {targets.get('T3', 10)}")

    md_report = exp_dir / "experiment.md"
    if md_report.exists():
        print(f"\nMarkdown Report: {md_report.resolve()}")

    print("=" * 70 + "\n")


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Load configuration
    cfg_path = Path(args.config) if args.config else None
    config = HarnessConfig.load(cfg_path)

    # Default to Checker_Harness/outputs if not explicitly specified
    default_outputs = _current_dir / "outputs"
    outputs_dir = Path(args.output_dir) if args.output_dir else default_outputs

    # Default to dataset/euroc under Checker_Harness or project root
    if args.dataset_dir:
        dataset_root = Path(args.dataset_dir)
    else:
        candidate_ds = _current_dir / config.dataset.root_dir
        if not candidate_ds.exists() and (Path(config.dataset.root_dir)).exists():
            dataset_root = Path(config.dataset.root_dir)
        else:
            dataset_root = candidate_ds

    runner = ExperimentRunner(
        outputs_dir=outputs_dir,
        dataset_root=dataset_root,
        degrader_config=config.degrader,
        harness_version=config.experiment.harness_version,
        t1_count=config.experiment.t1_runs,
        t2_count=config.experiment.t2_runs,
        t3_count=config.experiment.t3_runs,
    )

    try:
        if args.list_experiments:
            handle_list_experiments(runner.manager)

        elif args.info:
            handle_experiment_info(runner.manager, args.experiment_id)

        elif args.new_experiment or args.generate_segment_only:
            runner.run_new_experiment(
                sequence_name=args.sequence,
                seed=args.seed,
                generate_segment_only=args.generate_segment_only,
            )

        elif args.run_degradation:
            if not args.experiment_id:
                print("Error: --experiment <experiment_id> is required for --run-degradation", file=sys.stderr)
                sys.exit(1)
            runner.run_existing_experiment(
                experiment_id=args.experiment_id,
                rerun=False,
                seed=args.seed,
            )

        elif args.rerun_degradation:
            if not args.experiment_id:
                print("Error: --experiment <experiment_id> is required for --rerun-degradation", file=sys.stderr)
                sys.exit(1)
            runner.run_existing_experiment(
                experiment_id=args.experiment_id,
                rerun=True,
                seed=args.seed,
            )

    except MissingDatasetError as ex:
        print(str(ex), file=sys.stderr)
        sys.exit(2)

    except Exception as ex:
        print(f"\n[ERROR] {type(ex).__name__}: {str(ex)}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

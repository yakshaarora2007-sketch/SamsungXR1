#!/usr/bin/env python3
"""
generate_samples.py - IMU Evaluation Sample Generator for the EuRoC MAV Dataset.

Commands:
    python generate_samples.py --init   # Selects a new 15s motion interval and generates run_001
    python generate_samples.py --rerun  # Reuses the active 15s interval to generate run_002, run_003, ...
"""

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from interval_selector import IntervalSelector
from state_manager import StateManager
from t1_generator import T1Generator
from t2_generator import T2Generator
from t3_generator import T3Generator
from graph_generator import GraphGenerator
from validation import RunValidator, ValidationError


def generate_run_case(
    run_num: int,
    staging_dir: Path,
    imu_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    seed: Optional[int] = None,
) -> dict:
    """Generates T1, T2, T3 and graphs into a staging directory."""
    rng = np.random.default_rng(seed)

    # Derive unique sub-seeds if a base seed is provided
    s1 = int(rng.integers(0, 2**31 - 1)) if seed is not None else None
    s2 = int(rng.integers(0, 2**31 - 1)) if seed is not None else None
    s3 = int(rng.integers(0, 2**31 - 1)) if seed is not None else None

    # T1: Checkpoints
    t1_gen = T1Generator(imu_df, gt_df)
    t1_res = t1_gen.generate(seed=s1)
    t1_gen.save(staging_dir / "T1", t1_res)

    # T2: IMU Blackout
    t2_gen = T2Generator(imu_df, gt_df)
    t2_res = t2_gen.generate(seed=s2)
    t2_gen.save(staging_dir / "T2", t2_res)

    # T2 Graph
    GraphGenerator.generate_t2_graph(
        original_imu_df=imu_df,
        t2_result=t2_res,
        output_path=staging_dir / "T2" / "graphs" / "imu_blackout.png",
    )

    # T3: Sensor Degradation
    t3_gen = T3Generator(imu_df, gt_df)
    t3_res = t3_gen.generate(seed=s3)
    t3_gen.save(staging_dir / "T3", t3_res)

    # T3 Graphs
    GraphGenerator.generate_t3_graphs(
        t3_result=t3_res,
        output_dir=staging_dir / "T3" / "graphs",
    )

    run_meta = {
        "seeds": {"t1": s1, "t2": s2, "t3": s3, "master": seed},
        "t1_checkpoints_count": t1_res["metadata"]["num_checkpoints"],
        "t2_blackout_onset_rel_s": t2_res["metadata"]["blackout_onset_relative_s"],
        "t3_degradation_start_rel_s": t3_res["metadata"]["degradation_start_time_s"],
        "t3_degradation_duration_s": t3_res["metadata"]["degradation_duration_s"],
    }
    return run_meta


def handle_init(args):
    """Initializes a new base interval and generates run_001."""
    state_mgr = StateManager(args.output_dir)

    # A new initialization replaces the previous interval and its runs.
    state_mgr.clear_previous_runs_if_exists()

    # Select new 15-second base interval
    print(f"[INIT] Selecting a new 15.0s continuous interval from: {args.dataset_dir} ...")
    selector = IntervalSelector(args.dataset_dir)
    interval_data = selector.select_interval(sequence_name=args.sequence, seed=args.seed)

    seq_name = interval_data["sequence_name"]
    meta = interval_data["metadata"]
    print(f"[INIT] Selected sequence: {seq_name}")
    print(f"[INIT] Interval duration: {meta['nominal_duration_s']}s "
          f"({meta['start_timestamp_ns']} -> {meta['end_timestamp_ns']})")
    print(f"[INIT] Available IMU samples: {meta['imu_sample_count']} (dt ~ {meta['imu_median_dt_ms']:.2f} ms)")

    # Save base datasets and state
    current_dir = state_mgr.init_interval(interval_data)
    print(f"[INIT] Saved base interval data to: {current_dir}")

    # Generate run_001 atomically
    run_num = 1
    staging_dir = state_mgr.get_staging_directory(run_num)
    try:
        print(f"[INIT] Generating run_{run_num:03d} (T1, T2, T3 scenarios and graphs)...")
        run_meta = generate_run_case(
            run_num=run_num,
            staging_dir=staging_dir,
            imu_df=interval_data["imu_df"],
            gt_df=interval_data["gt_df"],
            seed=args.seed,
        )

        # Validate before committing
        print(f"[INIT] Validating run_{run_num:03d} integrity...")
        RunValidator.validate_run(
            run_dir=staging_dir,
            original_imu_df=interval_data["imu_df"],
            base_start_ts=meta["start_timestamp_ns"],
            base_end_ts=meta["end_timestamp_ns"],
        )

        final_run_dir = state_mgr.commit_staged_run(staging_dir, run_num, run_meta)
        print(f"[SUCCESS] run_{run_num:03d} successfully committed to: {final_run_dir}")
        print(f"  - T1: {final_run_dir / 'T1' / 'checkpoints.csv'}")
        print(f"  - T2: Blackout onset @ {run_meta['t2_blackout_onset_rel_s']:.2f}s")
        print(f"  - T3: 2s Degradation @ [{run_meta['t3_degradation_start_rel_s']:.2f}s – "
              f"{run_meta['t3_degradation_start_rel_s'] + run_meta['t3_degradation_duration_s']:.2f}s]")
        print(f"  - Graphs generated in T2/graphs and T3/graphs")

    except Exception as e:
        if staging_dir.exists():
            import shutil
            shutil.rmtree(staging_dir)
        print(f"[ERROR] Failed to generate run_{run_num:03d}: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


def handle_rerun(args):
    """Reuses the currently active 15s base interval to generate the next evaluation run."""
    state_mgr = StateManager(args.output_dir)

    if not state_mgr.has_active_interval():
        print(
            "[ERROR] No active base interval found. Please run '--init' first:\n"
            "    python generate_samples.py --init",
            file=sys.stderr,
        )
        sys.exit(1)

    state = state_mgr.load_state()
    current_dir = state_mgr.current_interval_dir

    imu_path = current_dir / "original_imu.csv"
    gt_path = current_dir / "original_groundtruth.csv"

    # Verify base files exist and match state
    if not imu_path.exists() or not gt_path.exists():
        print(f"[ERROR] Base interval files missing in {current_dir}", file=sys.stderr)
        sys.exit(1)

    imu_df = pd.read_csv(imu_path)
    gt_df = pd.read_csv(gt_path)

    ts_col = imu_df.columns[0]
    start_ts = int(imu_df[ts_col].iloc[0])
    end_ts = int(imu_df[ts_col].iloc[-1])

    if start_ts != state["start_timestamp_ns"]:
        print(
            f"[ERROR] Base interval integrity mismatch! "
            f"File start {start_ts} != state {state['start_timestamp_ns']}",
            file=sys.stderr,
        )
        sys.exit(1)

    run_num = state_mgr.get_next_run_number()
    print(f"[RERUN] Reusing active interval '{state['interval_id']}' ({state['sequence_name']})")
    print(f"[RERUN] Interval timestamps: {start_ts} -> {end_ts} (15.0s)")
    print(f"[RERUN] Generating run_{run_num:03d} ...")

    staging_dir = state_mgr.get_staging_directory(run_num)
    try:
        run_meta = generate_run_case(
            run_num=run_num,
            staging_dir=staging_dir,
            imu_df=imu_df,
            gt_df=gt_df,
            seed=args.seed,
        )

        print(f"[RERUN] Validating run_{run_num:03d} integrity...")
        RunValidator.validate_run(
            run_dir=staging_dir,
            original_imu_df=imu_df,
            base_start_ts=state["start_timestamp_ns"],
            base_end_ts=state["end_timestamp_ns"],
        )

        final_run_dir = state_mgr.commit_staged_run(staging_dir, run_num, run_meta)
        print(f"[SUCCESS] run_{run_num:03d} successfully committed to: {final_run_dir}")
        print(f"  - T1: 10 new randomized checkpoints")
        print(f"  - T2: New blackout onset @ {run_meta['t2_blackout_onset_rel_s']:.2f}s")
        print(f"  - T3: New 2s Degradation @ [{run_meta['t3_degradation_start_rel_s']:.2f}s – "
              f"{run_meta['t3_degradation_start_rel_s'] + run_meta['t3_degradation_duration_s']:.2f}s]")
        print(f"  - New graphs generated in T2/graphs and T3/graphs")

    except Exception as e:
        if staging_dir.exists():
            import shutil
            shutil.rmtree(staging_dir)
        print(f"[ERROR] Failed to generate rerun run_{run_num:03d}: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="IMU Evaluation Sample Generator for EuRoC MAV Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--init",
        action="store_true",
        help="Initialize a new 15-second motion interval and generate run_001",
    )
    group.add_argument(
        "--rerun",
        action="store_true",
        help="Reuse the current 15-second interval and generate the next run",
    )

    parser.add_argument(
        "--dataset-dir",
        default="vicon_room1",
        help="Path to EuRoC dataset directory containing sequence folders (default: vicon_room1)",
    )
    parser.add_argument(
        "--output-dir",
        default="generated_samples",
        help="Path to root output directory for samples (default: generated_samples)",
    )
    parser.add_argument(
        "--sequence",
        default=None,
        help="Optional specific sequence name to use for --init (default: random selection)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility",
    )

    args = parser.parse_args()

    if args.init:
        handle_init(args)
    elif args.rerun:
        handle_rerun(args)


if __name__ == "__main__":
    main()

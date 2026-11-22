"""
experiment/runner.py - Orchestrates Stage A (15s extraction) and Stage B (30 degradation runs).
Guarantees common source segment across all runs and independent degrader execution.
Progressively updates experiment.md on every run.
"""

from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

from dataset.loader import EuRoCLoader, SourceSegment, MissingDatasetError
from degradation.adapter import DegradationAdapter
from evaluation.aggregation import EvaluationAggregator
from .manager import ExperimentManager
from .reporter import ExperimentReporter


class ExperimentRunner:
    """Coordinates end-to-end trajectory degradation experiments."""

    def __init__(
        self,
        outputs_dir: str = "outputs",
        dataset_root: str = "dataset/euroc",
        degrader_config: Optional[Dict[str, Any]] = None,
        harness_version: str = "1.0.0",
        t1_count: int = 10,
        t2_count: int = 10,
        t3_count: int = 10,
    ):
        self.outputs_dir = Path(outputs_dir)
        self.dataset_root = Path(dataset_root)
        self.degrader_config = degrader_config or {}
        self.harness_version = harness_version
        self.t1_count = t1_count
        self.t2_count = t2_count
        self.t3_count = t3_count

        self.manager = ExperimentManager(outputs_dir=str(self.outputs_dir))
        self.degrader = DegradationAdapter(degrader_config=self.degrader_config)

    def run_new_experiment(
        self,
        sequence_name: str = "MH01",
        seed: Optional[int] = None,
        generate_segment_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes a new experiment:
        1. Checks dataset & extracts one 15s segment.
        2. Creates unique experiment folder.
        3. Saves source segment.
        4. If not segment_only, runs 10 T1, 10 T2, 10 T3 degradation runs.
        """
        print(f"\n========================================================")
        print(f"[HARNESS] Starting New Experiment for Sequence: {sequence_name}")
        print(f"========================================================")

        # Stage A: Extract 15s segment
        loader = EuRoCLoader(root_dir=str(self.dataset_root))
        print(f"[STAGE A] Discovering sequence '{sequence_name}' in '{self.dataset_root}'...")
        segment = loader.extract_15s_segment(sequence_name=sequence_name, seed=seed)

        # Create unique experiment directory
        exp_id, exp_dir = self.manager.create_experiment_dir(segment.sequence_name)
        print(f"[STAGE A] Created unique experiment directory: {exp_dir.resolve()}")

        # Save source segment permanently
        source_paths = segment.save(exp_dir / "source")
        print(f"[STAGE A] Saved 15.0s source segment (IMU samples: {len(segment.imu_df)}, GT samples: {len(segment.gt_df)})")

        exp_info = {
            "experiment_id": exp_id,
            "created_at": datetime.now().isoformat(),
            "dataset_name": "EuRoC MAV",
            "sequence_name": segment.sequence_name,
            "dataset_path": segment.dataset_path,
            "status": "Segment Generated (Degradation Pending)" if generate_segment_only else "Running Degradation",
            "t1_target": self.t1_count,
            "t2_target": self.t2_count,
            "t3_target": self.t3_count,
            "degrader_configuration": self.degrader.config,
            "harness_version": self.harness_version,
        }

        # Setup reporter and write initial experiment.md
        reporter = ExperimentReporter(exp_dir, exp_info, segment.metadata)
        reporter.write_report()
        print(f"[DOCS] Initialized documentation at {exp_dir / 'experiment.md'}")

        # Save top-level metadata.json
        self.manager.save_experiment_metadata(
            experiment_dir=exp_dir,
            exp_id=exp_id,
            segment=segment,
            degrader_config=self.degrader.config,
            harness_version=self.harness_version,
            t1_target=self.t1_count,
            t2_target=self.t2_count,
            t3_target=self.t3_count,
            status=exp_info["status"],
        )

        if generate_segment_only:
            print(f"[COMPLETE] 15-second segment extraction finished. Degradation marked pending.")
            return {
                "experiment_id": exp_id,
                "experiment_dir": str(exp_dir),
                "status": "Segment Generated (Degradation Pending)",
                "segment_info": segment.metadata,
            }

        # Stage B: Run all 30 degradation experiments
        completed_runs = self._execute_degradation_batch(
            exp_dir=exp_dir,
            exp_id=exp_id,
            segment=segment,
            reporter=reporter,
            master_seed=seed,
            base_dir=exp_dir,
        )

        # Generate reports in reports/
        report_paths = EvaluationAggregator.generate_reports(
            runs=completed_runs,
            output_dir=exp_dir / "reports",
            experiment_id=exp_id,
        )

        reporter.set_status("COMPLETED")
        self.manager.save_experiment_metadata(
            experiment_dir=exp_dir,
            exp_id=exp_id,
            segment=segment,
            degrader_config=self.degrader.config,
            harness_version=self.harness_version,
            t1_target=self.t1_count,
            t2_target=self.t2_count,
            t3_target=self.t3_count,
            run_seeds={r["run_id"]: r["random_seed"] for r in completed_runs},
            status="COMPLETED",
        )

        print(f"\n[COMPLETE] All 30 degradation runs finished successfully.")
        print(f"[DOCS] Final Experiment Report: {exp_dir / 'experiment.md'}")
        print(f"[DOCS] Evaluation Summary: {report_paths['summary_md']}")

        return {
            "experiment_id": exp_id,
            "experiment_dir": str(exp_dir),
            "status": "COMPLETED",
            "runs": completed_runs,
            "reports": {k: str(v) for k, v in report_paths.items()},
        }

    def run_existing_experiment(
        self,
        experiment_id: str,
        rerun: bool = False,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Runs or reruns degradation on an existing saved 15-second segment.
        If rerun=True, creates a new batch directory (batch_02, etc.) to preserve previous runs.
        """
        exp_dir = self.manager.get_experiment_dir(experiment_id)
        source_dir = exp_dir / "source"
        if not source_dir.exists():
            raise FileNotFoundError(f"Source segment not found in {exp_dir}")

        segment = SourceSegment.load(source_dir)
        exp_meta = self.manager.load_experiment_metadata(experiment_id)

        target_dir = exp_dir
        batch_prefix = ""
        if rerun:
            batch_num, target_dir = self.manager.get_next_batch_dir(exp_dir)
            batch_prefix = f"batch_{batch_num:02d}/"
            print(f"[RERUN] Created new batch version to preserve previous results: {target_dir}")

        exp_info = {
            "experiment_id": experiment_id,
            "created_at": exp_meta.get("creation_timestamp", datetime.now().isoformat()),
            "dataset_name": exp_meta.get("dataset_name", "EuRoC MAV"),
            "sequence_name": segment.sequence_name,
            "dataset_path": segment.dataset_path,
            "status": f"Running Degradation ({batch_prefix[:-1] if batch_prefix else 'batch_01'})",
            "t1_target": self.t1_count,
            "t2_target": self.t2_count,
            "t3_target": self.t3_count,
            "degrader_configuration": self.degrader.config,
            "harness_version": self.harness_version,
        }

        reporter = ExperimentReporter(exp_dir, exp_info, segment.metadata)

        completed_runs = self._execute_degradation_batch(
            exp_dir=exp_dir,
            exp_id=experiment_id,
            segment=segment,
            reporter=reporter,
            master_seed=seed,
            base_dir=target_dir,
            batch_prefix=batch_prefix,
        )

        report_paths = EvaluationAggregator.generate_reports(
            runs=completed_runs,
            output_dir=target_dir / "reports",
            experiment_id=experiment_id,
        )

        reporter.set_status("COMPLETED")
        return {
            "experiment_id": experiment_id,
            "experiment_dir": str(exp_dir),
            "target_dir": str(target_dir),
            "status": "COMPLETED",
            "runs": completed_runs,
        }

    def _execute_degradation_batch(
        self,
        exp_dir: Path,
        exp_id: str,
        segment: SourceSegment,
        reporter: ExperimentReporter,
        master_seed: Optional[int] = None,
        base_dir: Optional[Path] = None,
        batch_prefix: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Executes 10 T1, 10 T2, 10 T3 degradation runs on the common source segment.
        Updates reporter immediately after each individual run.
        """
        base_dir = base_dir or exp_dir
        rng = np.random.default_rng(master_seed)

        completed_runs = []
        total_runs = self.t1_count + self.t2_count + self.t3_count
        run_counter = 0

        tier_schedule = [
            ("T1", self.t1_count),
            ("T2", self.t2_count),
            ("T3", self.t3_count),
        ]

        print(f"\n[STAGE B] Starting 30 Degradation Runs (T1: {self.t1_count}, T2: {self.t2_count}, T3: {self.t3_count})...")

        for tier, count in tier_schedule:
            for i in range(1, count + 1):
                run_counter += 1
                run_id = f"run_{i:03d}"
                run_dir = base_dir / tier / run_id

                # Derive independent sub-seed for reproducibility
                sub_seed = int(rng.integers(0, 2**31 - 1)) if master_seed is not None else int(np.random.randint(0, 2**31 - 1))

                print(f"  [{run_counter:02d}/{total_runs:02d}] Executing {tier} - {run_id} (seed={sub_seed})... ", end="", flush=True)

                run_meta = self.degrader.execute_run(
                    tier=tier,
                    run_id=run_id,
                    output_dir=run_dir,
                    source_imu_df=segment.imu_df,
                    source_gt_df=segment.gt_df,
                    parent_experiment_id=exp_id,
                    source_segment_path=str(exp_dir / "source"),
                    seed=sub_seed,
                )

                if run_meta["execution_status"] == "SUCCESS":
                    print(f"OK ({run_meta['duration_seconds']:.2f}s)")
                else:
                    print(f"FAILED: {run_meta['error_details']}")

                # Immediately update experiment.md on disk
                reporter.add_run(run_meta)
                completed_runs.append(run_meta)

        return completed_runs

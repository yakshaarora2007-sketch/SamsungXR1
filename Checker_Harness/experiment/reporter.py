"""
experiment/reporter.py - Progressive Experiment Markdown Report generator.
Strictly adheres to the required Markdown format and updates on every completed run.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import json


class ExperimentReporter:
    """Generates and progressively updates experiment.md as runs execute."""

    def __init__(self, experiment_dir: Path, exp_info: Dict[str, Any], segment_info: Dict[str, Any]):
        self.experiment_dir = Path(experiment_dir)
        self.md_path = self.experiment_dir / "experiment.md"
        self.exp_info = dict(exp_info)
        self.segment_info = dict(segment_info)
        self.runs: List[Dict[str, Any]] = []
        self.errors_and_warnings: List[str] = []

    def set_status(self, status: str):
        self.exp_info["status"] = status
        self.write_report()

    def add_run(self, run_meta: Dict[str, Any]):
        """Records a completed run and immediately updates the Markdown report on disk."""
        # Check if run_id already in list (update if exists)
        existing_idx = None
        for i, r in enumerate(self.runs):
            if r.get("run_id") == run_meta.get("run_id") and r.get("tier") == run_meta.get("tier"):
                existing_idx = i
                break

        if existing_idx is not None:
            self.runs[existing_idx] = run_meta
        else:
            self.runs.append(run_meta)

        if run_meta.get("execution_status") != "SUCCESS" and run_meta.get("error_details"):
            self.errors_and_warnings.append(
                f"- **{run_meta.get('run_id')} ({run_meta.get('tier')}):** {run_meta.get('error_details')}"
            )

        self.write_report()

    def write_report(self):
        """Renders and writes experiment.md."""
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        exp_id = self.exp_info.get("experiment_id", "Unknown")
        created_at = self.exp_info.get("created_at", "Unknown")
        dataset_name = self.exp_info.get("dataset_name", "EuRoC MAV")
        seq_name = self.exp_info.get("sequence_name", "MH_01_easy")
        dataset_path = self.exp_info.get("dataset_path", "Unknown")
        status = self.exp_info.get("status", "In Progress")

        # Segment info
        seg_dur = self.segment_info.get("segment_duration_s", 15.0)
        start_ts = self.segment_info.get("start_timestamp_ns", "N/A")
        end_ts = self.segment_info.get("end_timestamp_ns", "N/A")
        rel_start_s = self.segment_info.get("start_time_relative_to_sequence_s", "N/A")
        rel_end_s = self.segment_info.get("end_time_relative_to_sequence_s", "N/A")
        orig_seq_dur = self.segment_info.get("original_sequence_duration_s", "N/A")
        imu_samples = self.segment_info.get("imu_sample_count", "N/A")
        gt_samples = self.segment_info.get("gt_sample_count", "N/A")

        if isinstance(rel_start_s, (int, float)):
            rel_start_str = f"{rel_start_s:.2f}s"
        else:
            rel_start_str = str(rel_start_s)

        if isinstance(rel_end_s, (int, float)):
            rel_end_str = f"{rel_end_s:.2f}s"
        else:
            rel_end_str = str(rel_end_s)

        if isinstance(orig_seq_dur, (int, float)):
            orig_seq_dur_str = f"{orig_seq_dur:.2f} seconds"
        else:
            orig_seq_dur_str = str(orig_seq_dur)

        # Degradation Summary Table
        t1_runs = [r for r in self.runs if r.get("tier") == "T1"]
        t2_runs = [r for r in self.runs if r.get("tier") == "T2"]
        t3_runs = [r for r in self.runs if r.get("tier") == "T3"]

        t1_succ = sum(1 for r in t1_runs if r.get("execution_status") == "SUCCESS")
        t2_succ = sum(1 for r in t2_runs if r.get("execution_status") == "SUCCESS")
        t3_succ = sum(1 for r in t3_runs if r.get("execution_status") == "SUCCESS")

        t1_fail = len(t1_runs) - t1_succ
        t2_fail = len(t2_runs) - t2_succ
        t3_fail = len(t3_runs) - t3_succ

        total_runs = len(self.runs)
        total_succ = t1_succ + t2_succ + t3_succ
        total_fail = t1_fail + t2_fail + t3_fail

        # Target run counts
        t1_target = self.exp_info.get("t1_target", 10)
        t2_target = self.exp_info.get("t2_target", 10)
        t3_target = self.exp_info.get("t3_target", 10)
        total_target = t1_target + t2_target + t3_target

        # Reproducibility
        sel_seed = self.segment_info.get("selection_seed", "None")
        deg_cfg_str = json.dumps(self.exp_info.get("degrader_configuration", {}), indent=2)
        harness_ver = self.exp_info.get("harness_version", "1.0.0")

        # Run Results table
        run_rows = []
        for r in self.runs:
            run_id = r.get("run_id", "")
            tier = r.get("tier", "")
            r_status = r.get("execution_status", "")
            seed_val = r.get("random_seed")
            seed_str = str(seed_val) if seed_val is not None else "None"
            out_rel = f"{tier}/{run_id}"
            run_rows.append(f"| {run_id} | {tier} | {r_status} | {seed_str} | {out_rel} |")

        if not run_rows:
            run_table_body = "| *No runs executed yet* | | | | |"
        else:
            run_table_body = "\n".join(run_rows)

        # Errors list
        if not self.errors_and_warnings:
            errors_section = "None. All attempted runs completed successfully without errors."
        else:
            errors_section = "\n".join(self.errors_and_warnings)

        # Individual seeds string
        seeds_summary = ", ".join(
            f"{r.get('run_id')}: {r.get('random_seed')}" for r in self.runs
        ) if self.runs else "None"

        # Memory Constraint Analysis
        mem_profiles = [r.get("memory") for r in self.runs if r.get("memory")]
        if mem_profiles:
            max_alg_mem = max(p.get("algorithm_peak_bytes", 0) for p in mem_profiles)
            max_alg_mb = max_alg_mem / (1024 * 1024)
            base_mem = mem_profiles[0].get("interpreter_baseline_bytes", 0)
            base_mb = base_mem / (1024 * 1024)
            rss_peak_mb = max(p.get("process_rss_peak_mb", 0.0) for p in mem_profiles)
            any_violation = any(p.get("status") == "VIOLATION" for p in mem_profiles)
            all_full = all(p.get("coverage") == "FULL" for p in mem_profiles)
            mem_status = "VIOLATION" if any_violation else "PASS"
            coverage_str = "FULL" if all_full else "PARTIALLY_VERIFIED"
            methodology_str = mem_profiles[0].get("methodology", "tracemalloc + native buffer tracking + process delta isolation")
        else:
            max_alg_mem = 0
            max_alg_mb = 0.0
            base_mem = 0
            base_mb = 0.0
            rss_peak_mb = 0.0
            mem_status = "PENDING"
            coverage_str = "UNVERIFIED"
            methodology_str = "tracemalloc + native buffer tracking + process delta isolation"

        memory_section = f"""## Memory Constraint Report (Algorithm / Heap Level)

| Metric | Requirement | Status |
|---|---|---|
| Algorithm-level peak memory | <= 8 MB | {max_alg_mb:.3f} MB ({max_alg_mem:,} bytes) |
| Python interpreter baseline | Excluded | {base_mb:.2f} MB (Exempt from limit) |
| OS-level memory | Excluded | Exempt from limit |
| Native algorithm buffers | Included | Tracked in Algorithm Memory |
| Temporary algorithm allocations | Included | Tracked in Algorithm Memory |

### Memory Verification Details

- **Constraint Status:** **{mem_status}**
- **Algorithm-Level Peak Memory:** {max_alg_mem:,} bytes ({max_alg_mb:.3f} MB) / 8,388,608 bytes (8.000 MB)
- **Python Interpreter Baseline:** {base_mem:,} bytes ({base_mb:.2f} MB) [Excluded]
- **Process RSS Peak (Diagnostic):** {rss_peak_mb:.2f} MB
- **Measurement Methodology:** {methodology_str}
- **Measurement Coverage:** {coverage_str}
"""

        content = f"""# Experiment Report

## Experiment Information

- Experiment ID: {exp_id}
- Creation timestamp: {created_at}
- Dataset: {dataset_name}
- Sequence: {seq_name}
- Original dataset path: {dataset_path}
- Experiment status: {status}

## Selected 15-Second Segment

- Segment duration: {seg_dur} seconds
- Start timestamp: {start_ts}
- End timestamp: {end_ts}
- Start time relative to original sequence: {rel_start_str}
- End time relative to original sequence: {rel_end_str}
- Original sequence duration: {orig_seq_dur_str}
- Number of IMU samples: {imu_samples}
- Number of ground-truth samples: {gt_samples}

## Degradation Summary

| Level | Total Runs | Successful | Failed |
|-------|------------|------------|--------|
| T1    | {t1_target}         | {t1_succ}          | {t1_fail}      |
| T2    | {t2_target}         | {t2_succ}          | {t2_fail}      |
| T3    | {t3_target}         | {t3_succ}          | {t3_fail}      |
| Total | {total_target}         | {total_succ}          | {total_fail}      |

## Individual Run Results

| Run ID | Level | Status | Seed | Output Path |
|--------|-------|--------|------|-------------|
{run_table_body}

{memory_section}

## Solver Evaluation

Status: Pending solver implementation.

## Errors and Warnings

{errors_section}

## Reproducibility

- Segment selection seed: {sel_seed}
- Individual degradation seeds: {seeds_summary}
- Degrader configuration:
```json
{deg_cfg_str}
```
- Harness version: {harness_ver}
"""

        with open(self.md_path, "w", encoding="utf-8") as f:
            f.write(content)

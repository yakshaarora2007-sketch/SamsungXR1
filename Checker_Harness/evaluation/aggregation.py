"""
evaluation/aggregation.py - Aggregates run results, generates summary statistics and reports.
"""

from pathlib import Path
from typing import Dict, List, Any
import json
import pandas as pd


class EvaluationAggregator:
    """Aggregates run metadata and evaluation metrics across degradation runs."""

    @staticmethod
    def generate_reports(
        runs: List[Dict[str, Any]],
        output_dir: Path,
        experiment_id: str,
    ) -> Dict[str, Path]:
        """
        Generates results.csv, results.json, and summary.md inside reports/ directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "results.csv"
        json_path = output_dir / "results.json"
        md_path = output_dir / "summary.md"

        # Tabulate run summary
        rows = []
        mem_profiles = []
        for r in runs:
            mem = r.get("memory") or {}
            if mem:
                mem_profiles.append(mem)
            rows.append({
                "run_id": r.get("run_id"),
                "tier": r.get("tier"),
                "status": r.get("execution_status"),
                "seed": r.get("random_seed"),
                "duration_seconds": round(r.get("duration_seconds", 0.0), 3),
                "algorithm_peak_mb": round(mem.get("algorithm_peak_mb", 0.0), 3),
                "memory_status": mem.get("status", "UNVERIFIED"),
                "error": r.get("error_details") or "None",
            })

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

        max_alg_mem = max((m.get("algorithm_peak_bytes", 0) for m in mem_profiles), default=0)
        max_alg_mb = max_alg_mem / (1024 * 1024)
        base_mem = mem_profiles[0].get("interpreter_baseline_bytes", 0) if mem_profiles else 0
        base_mb = base_mem / (1024 * 1024)
        any_mem_violation = any(m.get("status") == "VIOLATION" for m in mem_profiles)
        mem_overall_status = "VIOLATION" if any_mem_violation else "PASS"

        # JSON summary
        summary_data = {
            "experiment_id": experiment_id,
            "total_runs": len(runs),
            "successful_runs": sum(1 for r in runs if r.get("execution_status") == "SUCCESS"),
            "failed_runs": sum(1 for r in runs if r.get("execution_status") != "SUCCESS"),
            "memory_constraint": {
                "limit_bytes": 8 * 1024 * 1024,
                "limit_mb": 8.0,
                "max_algorithm_peak_bytes": max_alg_mem,
                "max_algorithm_peak_mb": round(max_alg_mb, 3),
                "interpreter_baseline_mb": round(base_mb, 3),
                "status": mem_overall_status,
                "all_runs_compliant": not any_mem_violation,
            },
            "runs": runs,
            "solver_evaluation": "Pending solver implementation",
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        # Markdown summary
        md_content = [
            f"# Experiment Evaluation Summary: {experiment_id}",
            "",
            "## Execution Overview",
            f"- **Total Runs Executed:** {len(runs)}",
            f"- **Successful Runs:** {summary_data['successful_runs']}",
            f"- **Failed Runs:** {summary_data['failed_runs']}",
            f"- **Memory Constraint Status (<= 8 MB):** **{mem_overall_status}** (Peak: {max_alg_mb:.3f} MB)",
            f"- **Solver Status:** Pending solver implementation",
            "",
            "## Memory Constraint Report (Algorithm / Heap Level)",
            "| Metric | Requirement | Observed Status |",
            "|---|---|---|",
            f"| Algorithm-level peak memory | <= 8 MB | {max_alg_mb:.3f} MB ({max_alg_mem:,} bytes) |",
            f"| Python interpreter baseline | Excluded | {base_mb:.2f} MB (Exempt from limit) |",
            "| OS-level memory | Excluded | Exempt from limit |",
            "| Native algorithm buffers | Included | Tracked in Algorithm Memory |",
            "| Temporary algorithm allocations | Included | Tracked in Algorithm Memory |",
            "",
            "## Breakdown by Degradation Tier",
            "| Tier | Total Runs | Successful | Failed |",
            "|---|---:|---:|---:|",
        ]

        for tier in ["T1", "T2", "T3"]:
            tier_runs = [r for r in runs if r.get("tier") == tier]
            success = sum(1 for r in tier_runs if r.get("execution_status") == "SUCCESS")
            failed = len(tier_runs) - success
            md_content.append(f"| {tier} | {len(tier_runs)} | {success} | {failed} |")

        md_content.extend([
            "",
            "## Trajectory Solver Evaluation",
            "Solver evaluation is pending implementation. Once a trajectory solver adapter is connected,",
            "this section will report Absolute Trajectory Error (ATE) and Relative Pose Error (RPE) across tiers.",
            "",
        ])

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_content))

        return {
            "csv": csv_path,
            "json": json_path,
            "summary_md": md_path,
        }

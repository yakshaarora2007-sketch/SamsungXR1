"""
degradation/adapter.py - Adapter for the existing degrader modules in Degrader/.
Reuses T1Generator, T2Generator, T3Generator, GraphGenerator, and RunValidator directly.
Does not rewrite any degradation logic.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

# Set safe MPLCONFIGDIR before importing matplotlib through degrader modules
mpl_cache_dir = Path(tempfile.gettempdir()) / "mpl_cache_harness"
mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

# Add Degrader directory to sys.path to load existing modules
_current_dir = Path(__file__).resolve().parent
_possible_degrader_paths = [
    _current_dir.parent.parent / "Degrader",
    Path("Degrader").resolve(),
    Path("../Degrader").resolve(),
    Path("e:/SamsungXR/Degrader"),
]

degrader_dir = None
for p in _possible_degrader_paths:
    if p.exists() and (p / "t1_generator.py").exists():
        degrader_dir = p.resolve()
        break

if degrader_dir and str(degrader_dir) not in sys.path:
    sys.path.insert(0, str(degrader_dir))

# Import existing degrader modules
try:
    from t1_generator import T1Generator
    from t2_generator import T2Generator
    from t3_generator import T3Generator
    from graph_generator import GraphGenerator
    from groundtruth_handler import GroundTruthHandler
    from validation import RunValidator, ValidationError
    from degradation import DEFAULT_DEGRADATION_CONFIG, identify_imu_columns
    from memory.monitor import AlgorithmMemoryMonitor, MemoryConstraintStatus
except ImportError as e:
    raise ImportError(
        f"Failed to import existing degrader modules from {degrader_dir}: {e}. "
        f"Ensure 'Degrader/' exists alongside 'Checker_Harness/'."
    )


class DegradationAdapter:
    """Interfaces with existing degrader without altering its logic."""

    def __init__(self, degrader_config: Optional[Dict[str, Any]] = None):
        self.config = dict(DEFAULT_DEGRADATION_CONFIG)
        if degrader_config:
            self.config.update(degrader_config)

    def execute_run(
        self,
        tier: str,
        run_id: str,
        output_dir: Path,
        source_imu_df: pd.DataFrame,
        source_gt_df: pd.DataFrame,
        parent_experiment_id: str,
        source_segment_path: str,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes an independent degradation run for the specified tier (T1, T2, or T3).
        Saves all outputs and per-run metadata into output_dir.
        """
        tier = tier.upper()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        start_time_iso = datetime.now().isoformat()
        start_ts = time.time()
        status = "SUCCESS"
        error_msg = None
        tier_meta = {}

        # Instrument algorithm memory monitoring (excludes interpreter baseline)
        memory_monitor = AlgorithmMemoryMonitor(operation_name=f"{tier}_{run_id}", strict=False)
        memory_monitor.start()
        memory_monitor.track_buffer("source_imu_df", source_imu_df)
        memory_monitor.track_buffer("source_gt_df", source_gt_df)

        try:
            # 1. Execute algorithm under memory monitoring (evaluating <= 8 MB constraint)
            if tier == "T1":
                generator = T1Generator(source_imu_df, source_gt_df)
                result = generator.generate(seed=seed)
                tier_meta = result["metadata"]
                memory_monitor.track_buffer("t1_checkpoints", result.get("checkpoints_df"))

            elif tier == "T2":
                generator = T2Generator(source_imu_df, source_gt_df)
                result = generator.generate(seed=seed)
                tier_meta = result["metadata"]
                memory_monitor.track_buffer("t2_imu_input", result.get("imu_input_df"))

            elif tier == "T3":
                generator = T3Generator(source_imu_df, source_gt_df)
                result = generator.generate(seed=seed, config=self.config)
                tier_meta = result["metadata"]
                memory_monitor.track_buffer("t3_imu_input", result.get("imu_input_df"))

            else:
                raise ValueError(f"Unknown tier: {tier}. Must be T1, T2, or T3.")

            # Conclude algorithm memory profile
            mem_profile = memory_monitor.stop()

            # 2. Save tier artifacts and generate visualization plots (post-algorithm reporting)
            generator.save(output_dir, result)

            if tier == "T1":
                ts_col = source_imu_df.columns[0]
                base_start = int(source_imu_df[ts_col].iloc[0])
                base_end = int(source_imu_df[ts_col].iloc[-1])
                RunValidator.validate_t1(output_dir, base_start, base_end)

            elif tier == "T2":
                GraphGenerator.generate_t2_graph(
                    original_imu_df=source_imu_df,
                    t2_result=result,
                    output_path=output_dir / "graphs" / "imu_blackout.png",
                )
                RunValidator.validate_t2(output_dir, source_imu_df)

            elif tier == "T3":
                GraphGenerator.generate_t3_graphs(
                    t3_result=result,
                    output_dir=output_dir / "graphs",
                )
                RunValidator.validate_t3(output_dir, source_imu_df)

        except Exception as ex:
            if memory_monitor._running:
                mem_profile = memory_monitor.stop()
            status = "FAILED"
            error_msg = f"{type(ex).__name__}: {str(ex)}"

        if mem_profile.status == MemoryConstraintStatus.VIOLATION:
            if status == "SUCCESS":
                status = "FAILED (MEMORY VIOLATION)"
            error_msg = (
                f"{error_msg + ' | ' if error_msg else ''}"
                f"Peak algorithm memory ({mem_profile.algorithm_peak_mb:.3f} MB) exceeded 8 MB limit!"
            )

        end_time_iso = datetime.now().isoformat()
        elapsed_s = time.time() - start_ts

        # Save comprehensive run metadata
        run_metadata = {
            "run_id": run_id,
            "tier": tier,
            "random_seed": seed,
            "parent_experiment_id": parent_experiment_id,
            "source_segment_path": str(source_segment_path),
            "execution_status": status,
            "start_time": start_time_iso,
            "end_time": end_time_iso,
            "duration_seconds": elapsed_s,
            "error_details": error_msg,
            "tier_metadata": tier_meta,
            "degrader_configuration": self.config,
            "memory": mem_profile.to_dict(),
        }

        with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
            json.dump(run_metadata, f, indent=2)

        return run_metadata

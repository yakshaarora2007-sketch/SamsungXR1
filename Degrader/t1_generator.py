"""
t1_generator.py - Generates Tier 1 (T1) Checkpoint Evaluation data.
Randomly selects 10 distinct IMU timestamps and extracts their exact ground-truth poses.
No interpolation or SLERP is applied.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from groundtruth_handler import GroundTruthHandler


class T1Generator:
    """Generates T1 checkpoint evaluation dataset."""

    NUM_CHECKPOINTS: int = 10

    def __init__(self, imu_df: pd.DataFrame, gt_df: pd.DataFrame):
        self.imu_df = imu_df.copy()
        self.gt_handler = GroundTruthHandler(gt_df)
        self.ts_col = self.imu_df.columns[0]
        self.imu_timestamps = self.imu_df[self.ts_col].to_numpy(dtype=np.int64)

    def generate(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Randomly picks 10 distinct timestamps from IMU frames having exact ground truth matches.
        """
        rng = np.random.default_rng(seed)

        # Candidate timestamps with exact ground truth entries
        exact_common = self.gt_handler.get_exact_common_timestamps(self.imu_timestamps.tolist())

        if len(exact_common) < self.NUM_CHECKPOINTS:
            # Fallback to any IMU frame if exact matches are somehow fewer than 10
            candidates = self.imu_timestamps.tolist()
        else:
            candidates = exact_common

        if len(candidates) < self.NUM_CHECKPOINTS:
            raise ValueError(f"Insufficient IMU frames ({len(candidates)}) to select {self.NUM_CHECKPOINTS} checkpoints.")

        # Choose 10 distinct timestamps
        selected_ts = sorted(rng.choice(candidates, size=self.NUM_CHECKPOINTS, replace=False).tolist())
        selected_ts = [int(t) for t in selected_ts]

        # Format checkpoints DataFrame
        checkpoints_df = self.gt_handler.format_checkpoints(selected_ts)

        t0 = int(self.imu_timestamps[0])
        rel_times_s = [(ts - t0) * 1e-9 for ts in selected_ts]

        metadata = {
            "tier": "T1",
            "description": "10 distinct IMU timestamps and corresponding exact ground-truth poses",
            "num_checkpoints": self.NUM_CHECKPOINTS,
            "random_seed": seed,
            "interval_start_ns": t0,
            "interval_end_ns": int(self.imu_timestamps[-1]),
            "checkpoints": [
                {"timestamp": ts, "relative_time_s": float(rel_t)}
                for ts, rel_t in zip(selected_ts, rel_times_s)
            ]
        }

        return {
            "checkpoints_df": checkpoints_df,
            "metadata": metadata,
        }

    def save(self, output_dir: Path, result: Dict[str, Any]):
        """Saves checkpoints.csv and metadata.json into T1 directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "checkpoints.csv"
        meta_path = output_dir / "metadata.json"

        result["checkpoints_df"].to_csv(csv_path, index=False)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(result["metadata"], f, indent=2)

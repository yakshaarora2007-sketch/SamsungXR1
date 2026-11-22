"""
t2_generator.py - Generates Tier 2 (T2) Complete IMU Blackout Evaluation data.
Randomly cuts off IMU sensor observations between 5 and 12 seconds.
Preserves full 15-second ground truth trajectory in groundtruth.csv.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd


class T2Generator:
    """Generates T2 complete IMU blackout scenario."""

    BLIND_PREDICTION_DURATION_S: float = 5.0

    def __init__(self, imu_df: pd.DataFrame, gt_df: pd.DataFrame):
        self.imu_df = imu_df.copy()
        self.gt_df = gt_df.copy()
        self.ts_col = self.imu_df.columns[0]
        self.timestamps = self.imu_df[self.ts_col].to_numpy(dtype=np.int64)

    def generate(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Randomly selects a blackout onset between 5.0 and 12.0 seconds.
        Removes all IMU observations from that point until the end of the interval.
        """
        rng = np.random.default_rng(seed)

        t0 = int(self.timestamps[0])
        t_end = int(self.timestamps[-1])

        # Random onset between 5.0 and 12.0 seconds
        onset_offset_s = float(rng.uniform(5.0, 12.0))
        target_onset_ns = t0 + int(round(onset_offset_s * 1e9))

        # Find closest actual IMU frame >= target_onset_ns
        candidate_idx = np.searchsorted(self.timestamps, target_onset_ns)
        if candidate_idx >= len(self.timestamps):
            candidate_idx = len(self.timestamps) - 1

        blackout_onset_ts = int(self.timestamps[candidate_idx])
        actual_onset_rel_s = (blackout_onset_ts - t0) * 1e-9

        # Filter IMU input strictly before blackout onset
        imu_input_df = self.imu_df.iloc[:candidate_idx].copy().reset_index(drop=True)

        # 5-second blind prediction boundary
        blind_end_ns = blackout_onset_ts + int(round(self.BLIND_PREDICTION_DURATION_S * 1e9))
        blind_end_rel_s = (blind_end_ns - t0) * 1e-9

        n_total = len(self.imu_df)
        n_available = len(imu_input_df)
        n_excluded = n_total - n_available

        metadata = {
            "tier": "T2",
            "description": "Complete IMU blackout between 7-10s with 5s official blind prediction period",
            "blackout_onset_timestamp_ns": blackout_onset_ts,
            "blackout_onset_relative_s": float(actual_onset_rel_s),
            "blind_prediction_duration_s": self.BLIND_PREDICTION_DURATION_S,
            "blind_prediction_end_timestamp_ns": blind_end_ns,
            "blind_prediction_end_relative_s": float(blind_end_rel_s),
            "interval_end_timestamp_ns": t_end,
            "interval_end_relative_s": float((t_end - t0) * 1e-9),
            "total_original_imu_observations": int(n_total),
            "available_imu_observations": int(n_available),
            "excluded_imu_observations": int(n_excluded),
            "random_seed": seed,
        }

        return {
            "imu_input_df": imu_input_df,
            "groundtruth_df": self.gt_df,
            "metadata": metadata,
            "onset_rel_s": float(actual_onset_rel_s),
            "blind_end_rel_s": float(blind_end_rel_s),
        }

    def save(self, output_dir: Path, result: Dict[str, Any]):
        """Saves imu_input.csv, groundtruth.csv, and metadata.json into T2 directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        imu_csv = output_dir / "imu_input.csv"
        gt_csv = output_dir / "groundtruth.csv"
        meta_json = output_dir / "metadata.json"

        result["imu_input_df"].to_csv(imu_csv, index=False)
        result["groundtruth_df"].to_csv(gt_csv, index=False)
        with open(meta_json, "w", encoding="utf-8") as f:
            json.dump(result["metadata"], f, indent=2)

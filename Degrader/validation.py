"""
validation.py - Comprehensive validation checks for generated base intervals and runs.
Ensures all non-negotiable requirements for T1, T2, and T3 are rigorously met before committing a run.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from degradation import identify_imu_columns


class ValidationError(Exception):
    """Raised when validation of an evaluation tier or run fails."""
    pass


class RunValidator:
    """Validates data integrity and compliance of generated runs."""

    @staticmethod
    def validate_t1(t1_dir: Path, base_start_ts: int, base_end_ts: int) -> bool:
        """Validates T1 checkpoints."""
        csv_path = t1_dir / "checkpoints.csv"
        meta_path = t1_dir / "metadata.json"

        if not csv_path.exists():
            raise ValidationError(f"T1 checkpoints.csv missing at {csv_path}")
        if not meta_path.exists():
            raise ValidationError(f"T1 metadata.json missing at {meta_path}")

        df = pd.read_csv(csv_path)

        # Requirement: exactly 10 checkpoints
        if len(df) != 10:
            raise ValidationError(f"T1 must have exactly 10 checkpoints, found {len(df)}")

        # Required columns
        required_cols = [
            "timestamp", "position_x", "position_y", "position_z",
            "quaternion_x", "quaternion_y", "quaternion_z", "quaternion_w"
        ]
        for col in required_cols:
            if col not in df.columns:
                raise ValidationError(f"T1 checkpoints.csv missing column '{col}'")

        # Distinct timestamps
        ts = df["timestamp"].to_numpy(dtype=np.int64)
        if len(np.unique(ts)) != 10:
            raise ValidationError("T1 checkpoints contain duplicate timestamps")

        # Timestamps within base interval
        if np.any(ts < base_start_ts) or np.any(ts > base_end_ts):
            raise ValidationError("T1 timestamps out of base interval bounds")

        # No nulls/NaNs
        if df.isna().any().any():
            raise ValidationError("T1 checkpoints contain NaN values")

        return True

    @staticmethod
    def validate_t2(t2_dir: Path, original_imu_df: pd.DataFrame) -> bool:
        """Validates T2 complete IMU blackout scenario."""
        imu_csv = t2_dir / "imu_input.csv"
        gt_csv = t2_dir / "groundtruth.csv"
        meta_json = t2_dir / "metadata.json"
        graph_png = t2_dir / "graphs" / "imu_blackout.png"

        if not imu_csv.exists():
            raise ValidationError(f"T2 imu_input.csv missing at {imu_csv}")
        if not gt_csv.exists():
            raise ValidationError(f"T2 groundtruth.csv missing at {gt_csv}")
        if not meta_json.exists():
            raise ValidationError(f"T2 metadata.json missing at {meta_json}")
        if not graph_png.exists() or graph_png.stat().st_size == 0:
            raise ValidationError(f"T2 graph missing or empty at {graph_png}")

        with open(meta_json, "r", encoding="utf-8") as f:
            meta = json.load(f)

        onset_s = meta.get("blackout_onset_relative_s", 0.0)
        if not (4.99 <= onset_s <= 12.01):
            raise ValidationError(f"T2 blackout onset {onset_s:.2f}s is not within [5.0, 12.0] seconds")

        onset_ts = meta["blackout_onset_timestamp_ns"]

        # Check that no observations exist at or after onset_ts
        imu_df = pd.read_csv(imu_csv)
        ts_col = imu_df.columns[0]
        timestamps = imu_df[ts_col].to_numpy(dtype=np.int64)

        if len(timestamps) > 0 and np.any(timestamps >= onset_ts):
            raise ValidationError("T2 imu_input.csv contains observations at or after blackout onset")

        # Ground truth should preserve full 15 seconds
        gt_df = pd.read_csv(gt_csv)
        if len(gt_df) == 0:
            raise ValidationError("T2 groundtruth.csv is empty")

        return True

    @staticmethod
    def validate_t3(t3_dir: Path, original_imu_df: pd.DataFrame) -> bool:
        """Validates T3 sensor degradation scenario and clean region invariance."""
        imu_csv = t3_dir / "imu_input.csv"
        gt_csv = t3_dir / "groundtruth.csv"
        meta_json = t3_dir / "metadata.json"
        acc_png = t3_dir / "graphs" / "accelerometer_degradation.png"
        gyr_png = t3_dir / "graphs" / "gyroscope_degradation.png"

        for path in [imu_csv, gt_csv, meta_json, acc_png, gyr_png]:
            if not path.exists() or path.stat().st_size == 0:
                raise ValidationError(f"T3 output missing or empty: {path}")

        with open(meta_json, "r", encoding="utf-8") as f:
            meta = json.load(f)

        start_s = meta.get("degradation_start_time_s", 0.0)
        dur_s = meta.get("degradation_duration_s", 0.0)

        if not (4.99 <= start_s <= 7.01):
            raise ValidationError(f"T3 degradation start time {start_s:.2f}s is not within [5.0, 7.0] seconds")
        if not (1.99 <= dur_s <= 2.01):
            raise ValidationError(f"T3 degradation duration {dur_s:.2f}s is not 2.0 seconds")

        degraded_df = pd.read_csv(imu_csv)
        ts_col, orig_acc, orig_gyr = identify_imu_columns(original_imu_df)

        orig_ts = original_imu_df[ts_col].to_numpy(dtype=np.int64)
        deg_start_ts = meta["start_timestamp_ns"]
        deg_end_ts = meta["end_timestamp_ns"]

        # Check clean pre-region
        clean_pre_orig = original_imu_df[original_imu_df[ts_col] < deg_start_ts].reset_index(drop=True)
        clean_pre_deg = degraded_df[degraded_df[ts_col] < deg_start_ts].reset_index(drop=True)

        if len(clean_pre_orig) != len(clean_pre_deg):
            raise ValidationError(
                f"T3 clean pre-region row count mismatch: orig={len(clean_pre_orig)}, degraded={len(clean_pre_deg)}"
            )

        if not clean_pre_orig.equals(clean_pre_deg):
            # Check numerical equivalence
            diff = np.max(np.abs(clean_pre_orig.iloc[:, 1:].to_numpy() - clean_pre_deg.iloc[:, 1:].to_numpy()))
            if diff > 1e-6:
                raise ValidationError(f"T3 clean pre-region was modified! Max difference: {diff}")

        # Check clean post-region
        clean_post_orig = original_imu_df[original_imu_df[ts_col] > deg_end_ts].reset_index(drop=True)
        clean_post_deg = degraded_df[degraded_df[ts_col] > deg_end_ts].reset_index(drop=True)

        if len(clean_post_orig) != len(clean_post_deg):
            raise ValidationError(
                f"T3 clean post-region row count mismatch: orig={len(clean_post_orig)}, degraded={len(clean_post_deg)}"
            )

        if not clean_post_orig.equals(clean_post_deg):
            diff = np.max(np.abs(clean_post_orig.iloc[:, 1:].to_numpy() - clean_post_deg.iloc[:, 1:].to_numpy()))
            if diff > 1e-6:
                raise ValidationError(f"T3 clean post-region was modified! Max difference: {diff}")

        return True

    @classmethod
    def validate_run(
        cls,
        run_dir: Path,
        original_imu_df: pd.DataFrame,
        base_start_ts: int,
        base_end_ts: int,
    ) -> bool:
        """Validates all tiers of a generated run."""
        cls.validate_t1(run_dir / "T1", base_start_ts, base_end_ts)
        cls.validate_t2(run_dir / "T2", original_imu_df)
        cls.validate_t3(run_dir / "T3", original_imu_df)
        return True

"""
interval_selector.py - Discovers EuRoC sequences and extracts continuous 15.0-second base intervals.
Preserves int64 nanosecond timestamps and exact original sensor columns.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


class IntervalSelector:
    """Discovers EuRoC sequences and selects continuous 15-second intervals."""

    DURATION_NS: int = 15_000_000_000  # Exactly 15.0 seconds in nanoseconds

    def __init__(self, dataset_dir: str = "vicon_room1"):
        self.dataset_dir = Path(dataset_dir)
        self.sequences = self._discover_sequences()

    def _discover_sequences(self) -> Dict[str, Dict[str, Path]]:
        """Finds all valid EuRoC sequence directories containing imu0 and groundtruth data."""
        discovered = {}
        if not self.dataset_dir.exists():
            return discovered

        for item in sorted(self.dataset_dir.iterdir()):
            if not item.is_dir():
                continue
            seq_name = item.name
            imu_csv = item / "mav0" / "imu0" / "data.csv"
            # Prefer state_groundtruth_estimate0, fallback to vicon0 if needed
            gt_csv = item / "mav0" / "state_groundtruth_estimate0" / "data.csv"
            if not gt_csv.exists():
                gt_csv = item / "mav0" / "vicon0" / "data.csv"

            if imu_csv.exists() and gt_csv.exists():
                discovered[seq_name] = {
                    "root": item,
                    "imu": imu_csv,
                    "gt": gt_csv,
                }
        return discovered

    def get_sequence_names(self) -> List[str]:
        return list(self.sequences.keys())

    def select_interval(
        self,
        sequence_name: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Randomly selects a valid sequence (or uses sequence_name) and extracts a continuous 15-second interval.
        """
        rng = np.random.default_rng(seed)

        if not self.sequences:
            raise FileNotFoundError(
                f"No valid EuRoC sequences with imu0/data.csv and groundtruth found in '{self.dataset_dir}'"
            )

        if sequence_name:
            if sequence_name not in self.sequences:
                raise ValueError(f"Sequence '{sequence_name}' not found. Available: {list(self.sequences.keys())}")
            selected_seq = sequence_name
        else:
            selected_seq = rng.choice(list(self.sequences.keys()))

        seq_info = self.sequences[selected_seq]

        # Load full datasets
        imu_df = pd.read_csv(seq_info["imu"])
        gt_df = pd.read_csv(seq_info["gt"])

        # Timestamp columns
        imu_ts_col = imu_df.columns[0]
        gt_ts_col = gt_df.columns[0]

        imu_timestamps = imu_df[imu_ts_col].to_numpy(dtype=np.int64)
        gt_timestamps = gt_df[gt_ts_col].to_numpy(dtype=np.int64)

        # Find overlapping range
        t_overlap_start = max(imu_timestamps[0], gt_timestamps[0])
        t_overlap_end = min(imu_timestamps[-1], gt_timestamps[-1])

        available_span_ns = t_overlap_end - t_overlap_start
        if available_span_ns < self.DURATION_NS:
            raise ValueError(
                f"Sequence '{selected_seq}' overlapping duration ({available_span_ns * 1e-9:.2f}s) "
                f"is less than required 15.0 seconds."
            )

        # Max valid start timestamp
        max_start_ts = t_overlap_end - self.DURATION_NS

        # Valid candidate IMU timestamps within [t_overlap_start, max_start_ts]
        valid_imu_starts = imu_timestamps[
            (imu_timestamps >= t_overlap_start) & (imu_timestamps <= max_start_ts)
        ]

        if len(valid_imu_starts) == 0:
            raise ValueError(f"No valid starting IMU timestamp found for 15s interval in '{selected_seq}'")

        # Randomly choose start timestamp from candidate IMU frames
        start_ts = int(rng.choice(valid_imu_starts))
        end_ts = int(start_ts + self.DURATION_NS)

        # Slice IMU and GT data
        imu_slice = imu_df[(imu_df[imu_ts_col] >= start_ts) & (imu_df[imu_ts_col] <= end_ts)].copy()
        gt_slice = gt_df[(gt_df[gt_ts_col] >= start_ts) & (gt_df[gt_ts_col] <= end_ts)].copy()

        # Calculate relative duration and median dt
        actual_duration_s = (imu_slice[imu_ts_col].iloc[-1] - imu_slice[imu_ts_col].iloc[0]) * 1e-9
        imu_dt_s = np.diff(imu_slice[imu_ts_col].to_numpy()) * 1e-9
        median_dt_ms = float(np.median(imu_dt_s) * 1000.0) if len(imu_dt_s) > 0 else 0.0

        metadata = {
            "sequence_name": selected_seq,
            "start_timestamp_ns": start_ts,
            "end_timestamp_ns": end_ts,
            "nominal_duration_s": 15.0,
            "actual_duration_s": float(actual_duration_s),
            "imu_sample_count": int(len(imu_slice)),
            "gt_sample_count": int(len(gt_slice)),
            "imu_median_dt_ms": median_dt_ms,
            "random_seed": seed,
        }

        return {
            "sequence_name": selected_seq,
            "imu_df": imu_slice,
            "gt_df": gt_slice,
            "metadata": metadata,
        }

"""
groundtruth_handler.py - Ground truth pose extraction and formatting for EuRoC MAV dataset.
Interpolation and SLERP are completely removed as per user instruction; poses are matched
directly to exact ground truth records without approximation.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd


class GroundTruthHandler:
    """Handles parsing, slicing, and pose lookup for EuRoC ground truth files."""

    def __init__(self, gt_path_or_df: Any):
        if isinstance(gt_path_or_df, (str, bytes)):
            self.df = self.load_groundtruth(gt_path_or_df)
        elif isinstance(gt_path_or_df, pd.DataFrame):
            self.df = gt_path_or_df.copy()
            self._standardize_columns()
        else:
            raise TypeError("Expected file path or pandas DataFrame")

        # Map timestamps to row index for O(1) exact lookups
        self._ts_to_idx = {int(ts): idx for idx, ts in enumerate(self.df["timestamp"].to_numpy())}
        self._timestamps = self.df["timestamp"].to_numpy(dtype=np.int64)

    @staticmethod
    def load_groundtruth(file_path: str) -> pd.DataFrame:
        """Loads EuRoC state_groundtruth_estimate0 CSV file."""
        df = pd.read_csv(file_path)
        # Strip whitespace from column names
        df.columns = [col.strip() for col in df.columns]
        # Standardize timestamp column name
        if df.columns[0].startswith("#"):
            df = df.rename(columns={df.columns[0]: "timestamp"})
        df["timestamp"] = df["timestamp"].astype(np.int64)
        return df

    def _standardize_columns(self):
        self.df.columns = [col.strip() for col in self.df.columns]
        if self.df.columns[0].startswith("#"):
            self.df = self.df.rename(columns={self.df.columns[0]: "timestamp"})
        self.df["timestamp"] = self.df["timestamp"].astype(np.int64)

    def slice_interval(self, start_ts: int, end_ts: int) -> pd.DataFrame:
        """Extracts ground truth observations within [start_ts, end_ts]."""
        mask = (self.df["timestamp"] >= start_ts) & (self.df["timestamp"] <= end_ts)
        return self.df[mask].copy().reset_index(drop=True)

    def get_exact_common_timestamps(self, imu_timestamps: List[int]) -> List[int]:
        """Finds all timestamps that exist exactly in both IMU and Ground Truth."""
        gt_ts_set = set(self._ts_to_idx.keys())
        common = [int(ts) for ts in imu_timestamps if int(ts) in gt_ts_set]
        return common

    def get_pose_exact(self, timestamp: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves the exact ground truth pose for a timestamp without interpolation.
        If timestamp is not in GT, finds the nearest exact frame without SLERP.
        """
        ts = int(timestamp)
        if ts in self._ts_to_idx:
            idx = self._ts_to_idx[ts]
        else:
            # Nearest exact frame without interpolation
            idx = int(np.argmin(np.abs(self._timestamps - ts)))

        row = self.df.iloc[idx]
        actual_ts = int(row["timestamp"])

        # Detect position columns
        p_cols = [c for c in self.df.columns if "p_RS_R" in c or "p_x" in c or c in ["x", "px"]]
        if len(p_cols) >= 3:
            px, py, pz = float(row[p_cols[0]]), float(row[p_cols[1]]), float(row[p_cols[2]])
        else:
            px, py, pz = float(row.iloc[1]), float(row.iloc[2]), float(row.iloc[3])

        # Detect quaternion columns (EuRoC has q_RS_w, q_RS_x, q_RS_y, q_RS_z)
        qw_col = [c for c in self.df.columns if "q_RS_w" in c or c in ["qw", "q_w"]]
        qx_col = [c for c in self.df.columns if "q_RS_x" in c or c in ["qx", "q_x"]]
        qy_col = [c for c in self.df.columns if "q_RS_y" in c or c in ["qy", "q_y"]]
        qz_col = [c for c in self.df.columns if "q_RS_z" in c or c in ["qz", "q_z"]]

        if qw_col and qx_col and qy_col and qz_col:
            qw = float(row[qw_col[0]])
            qx = float(row[qx_col[0]])
            qy = float(row[qy_col[0]])
            qz = float(row[qz_col[0]])
        else:
            # Fallback by standard column indices (4, 5, 6, 7)
            qw = float(row.iloc[4])
            qx = float(row.iloc[5])
            qy = float(row.iloc[6])
            qz = float(row.iloc[7])

        return {
            "timestamp": int(ts),
            "matched_gt_timestamp": actual_ts,
            "position_x": px,
            "position_y": py,
            "position_z": pz,
            "quaternion_x": qx,
            "quaternion_y": qy,
            "quaternion_z": qz,
            "quaternion_w": qw,
        }

    def format_checkpoints(self, timestamps: List[int]) -> pd.DataFrame:
        """
        Creates the standardized T1 checkpoints DataFrame for the requested timestamps.
        Columns: timestamp, position_x, position_y, position_z,
                 quaternion_x, quaternion_y, quaternion_z, quaternion_w
        """
        records = []
        for ts in timestamps:
            pose = self.get_pose_exact(ts)
            records.append({
                "timestamp": pose["timestamp"],
                "position_x": pose["position_x"],
                "position_y": pose["position_y"],
                "position_z": pose["position_z"],
                "quaternion_x": pose["quaternion_x"],
                "quaternion_y": pose["quaternion_y"],
                "quaternion_z": pose["quaternion_z"],
                "quaternion_w": pose["quaternion_w"],
            })

        cols = [
            "timestamp",
            "position_x", "position_y", "position_z",
            "quaternion_x", "quaternion_y", "quaternion_z", "quaternion_w"
        ]
        return pd.DataFrame(records, columns=cols)

"""
evaluation/trajectory.py - Trajectory representations and timestamp association.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import pandas as pd


@dataclass
class Trajectory:
    """Represents a timestamped 6-DOF trajectory."""
    timestamps: np.ndarray       # int64 nanoseconds (N,)
    positions: np.ndarray        # float64 [x, y, z] (N, 3)
    orientations: np.ndarray     # float64 quaternion [x, y, z, w] (N, 4)

    def __len__(self) -> int:
        return len(self.timestamps)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "Trajectory":
        """
        Parses trajectory from EuRoC groundtruth or checkpoint DataFrame.
        Expected columns:
          - timestamp
          - p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m] (or position_x, position_y, position_z)
          - q_RS_w, q_RS_x, q_RS_y, q_RS_z (or quaternion_x, quaternion_y, quaternion_z, quaternion_w)
        """
        cols = [c.strip() for c in df.columns]
        ts_col = df.columns[0]
        timestamps = df[ts_col].to_numpy(dtype=np.int64)

        # Position columns
        pos_cols = [c for c in df.columns if any(k in c.lower() for k in ["p_rs_r", "position", "pos"])]
        if len(pos_cols) >= 3:
            positions = df[pos_cols[:3]].to_numpy(dtype=float)
        elif len(df.columns) >= 4:
            positions = df.iloc[:, 1:4].to_numpy(dtype=float)
        else:
            raise ValueError("Could not find 3 position columns in DataFrame")

        # Quaternion columns
        # EuRoC native groundtruth has: q_RS_w, q_RS_x, q_RS_y, q_RS_z
        # T1 checkpoints has: quaternion_x, quaternion_y, quaternion_z, quaternion_w
        quat_cols = [c for c in df.columns if any(k in c.lower() for k in ["q_rs", "quaternion", "quat"])]
        if len(quat_cols) >= 4:
            q_data = df[quat_cols[:4]].to_numpy(dtype=float)
            # Check if first quat column is 'w'
            if "w" in quat_cols[0].lower():
                # [w, x, y, z] -> [x, y, z, w]
                orientations = np.column_stack([q_data[:, 1], q_data[:, 2], q_data[:, 3], q_data[:, 0]])
            else:
                orientations = q_data
        elif len(df.columns) >= 8:
            orientations = df.iloc[:, 4:8].to_numpy(dtype=float)
        else:
            # Default identity quaternions if not present
            orientations = np.tile([0.0, 0.0, 0.0, 1.0], (len(timestamps), 1))

        return cls(timestamps=timestamps, positions=positions, orientations=orientations)


def associate_trajectories(
    traj_ref: Trajectory,
    traj_est: Trajectory,
    max_diff_ns: int = 20_000_000,  # 20ms maximum time difference
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Associates timestamps between reference (ground truth) and estimated trajectories.
    Returns indices (idx_ref, idx_est) of matched frames.
    """
    ref_ts = traj_ref.timestamps
    est_ts = traj_est.timestamps

    matched_ref = []
    matched_est = []

    for i, t_est in enumerate(est_ts):
        # Find closest reference frame
        diffs = np.abs(ref_ts - t_est)
        min_idx = int(np.argmin(diffs))
        if diffs[min_idx] <= max_diff_ns:
            matched_ref.append(min_idx)
            matched_est.append(i)

    return np.array(matched_ref, dtype=int), np.array(matched_est, dtype=int)

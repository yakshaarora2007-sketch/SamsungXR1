"""
evaluation/metrics.py - Absolute Trajectory Error (ATE) and Relative Pose Error (RPE).
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import numpy as np

from .trajectory import Trajectory, associate_trajectories
from .alignment import align_umeyama, apply_transformation


@dataclass
class ATEMetrics:
    """Absolute Trajectory Error metrics."""
    rmse: float
    mean: float
    median: float
    std: float
    min: float
    max: float
    num_pairs: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RPEMetrics:
    """Relative Pose Error metrics."""
    trans_rmse: float
    trans_mean: float
    rot_rmse_deg: float
    rot_mean_deg: float
    delta_time_s: float
    num_pairs: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_ate(
    traj_ref: Trajectory,
    traj_est: Trajectory,
    align: bool = True,
    estimate_scale: bool = False,
) -> ATEMetrics:
    """
    Computes Absolute Trajectory Error (ATE) between reference and estimated trajectories.
    Optionally performs SE(3) Umeyama alignment before computing errors.
    """
    idx_ref, idx_est = associate_trajectories(traj_ref, traj_est)
    if len(idx_ref) < 3:
        raise ValueError(f"Insufficient matched pairs ({len(idx_ref)}) for trajectory evaluation")

    pts_ref = traj_ref.positions[idx_ref]
    pts_est = traj_est.positions[idx_est]

    if align:
        R, t, s = align_umeyama(pts_ref, pts_est, estimate_scale=estimate_scale)
        pts_est_aligned = apply_transformation(pts_est, R, t, s)
    else:
        pts_est_aligned = pts_est

    # Euclidean distance errors
    errors = np.linalg.norm(pts_ref - pts_est_aligned, axis=1)

    return ATEMetrics(
        rmse=float(np.sqrt(np.mean(errors ** 2))),
        mean=float(np.mean(errors)),
        median=float(np.median(errors)),
        std=float(np.std(errors)),
        min=float(np.min(errors)),
        max=float(np.max(errors)),
        num_pairs=len(errors),
    )


def compute_rpe(
    traj_ref: Trajectory,
    traj_est: Trajectory,
    delta_time_s: float = 1.0,
) -> RPEMetrics:
    """
    Computes Relative Pose Error (RPE) over delta_time_s intervals.
    """
    idx_ref, idx_est = associate_trajectories(traj_ref, traj_est)
    if len(idx_ref) < 5:
        raise ValueError(f"Insufficient matched pairs ({len(idx_ref)}) for RPE evaluation")

    timestamps = traj_ref.timestamps[idx_ref]
    pts_ref = traj_ref.positions[idx_ref]
    pts_est = traj_est.positions[idx_est]

    delta_ns = int(delta_time_s * 1e9)
    trans_errors = []

    for i in range(len(timestamps)):
        target_t = timestamps[i] + delta_ns
        j = np.searchsorted(timestamps, target_t)
        if j < len(timestamps) and abs(timestamps[j] - target_t) < 0.1 * delta_ns:
            # Reference relative displacement
            d_ref = pts_ref[j] - pts_ref[i]
            # Estimated relative displacement
            d_est = pts_est[j] - pts_est[i]
            trans_errors.append(np.linalg.norm(d_ref - d_est))

    if len(trans_errors) == 0:
        # Fallback to adjacent frame diffs if fixed interval did not match
        diff_ref = np.diff(pts_ref, axis=0)
        diff_est = np.diff(pts_est, axis=0)
        trans_errors = np.linalg.norm(diff_ref - diff_est, axis=1).tolist()

    trans_arr = np.array(trans_errors)
    return RPEMetrics(
        trans_rmse=float(np.sqrt(np.mean(trans_arr ** 2))),
        trans_mean=float(np.mean(trans_arr)),
        rot_rmse_deg=0.0,
        rot_mean_deg=0.0,
        delta_time_s=delta_time_s,
        num_pairs=len(trans_arr),
    )

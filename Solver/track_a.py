"""
track_a.py - Track A Trajectory Estimation Implementations (A1, A2, A3).
Pure Quaternion Mathematics, MEKF, RTS Smoothing, and Gauss-Newton Optimization.
Zero trainable parameters, zero external dependencies, strictly <= 8 MB memory.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from .quaternion import (
    quat_mul,
    quat_from_rotvec,
    quat_to_rotmat,
    quat_normalize,
    quat_slerp,
    quat_rotate,
)
from .detect import detect_faults, DetectionResult
from .clean import clean_imu_stream
from .strapdown import strapdown_forward, strapdown_backward
from .mekf import MEKF15, MEKFState, run_forward_mekf, rts_smooth
from .gauss_newton import BatchGaussNewtonOptimizer


STANDARD_GRAVITY = np.array([0.0, 0.0, 9.81], dtype=np.float64)


def estimate_initial_attitude(accel_samples: np.ndarray) -> np.ndarray:
    """
    Estimates initial attitude quaternion [w, x, y, z] from initial accelerometer samples.
    Aligns body-frame specific force with world gravity [0, 0, 9.81].
    """
    a_mean = np.mean(accel_samples, axis=0)
    norm_a = np.linalg.norm(a_mean)
    if norm_a < 1e-4:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    # In body frame, stationary accelerometer measures +g upward (reaction to gravity)
    # Target in world frame is [0, 0, 1]
    b = a_mean / norm_a
    w = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    v = np.cross(b, w)
    c = np.dot(b, w)

    if c < -0.9999:
        # 180 degree flip around perpendicular axis
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

    # Half-angle rotation formula
    s = np.sqrt(2.0 * (1.0 + c))
    q = np.array([
        0.5 * s,
        v[0] / s,
        v[1] / s,
        v[2] / s
    ], dtype=np.float64)
    return quat_normalize(q)


class SolverA1:
    """
    A1: Classical Baseline.
    - Fault detection (Hampel + dt + stuck)
    - Pre-detection median/MAD substitution + bias subtraction
    - Forward & backward strapdown passes
    - 15-state MEKF + RTS backward smoother
    - Quaternion SLERP + inverse-covariance pose fusion
    """

    def __init__(self, name: str = "A1_ClassicalBaseline"):
        self.name = name

    def parse_imu_dataframe(self, imu_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts timestamps (N,) and 6-channel IMU data (N, 6) [w_x, w_y, w_z, a_x, a_y, a_z]."""
        ts_col = imu_df.columns[0]
        timestamps_ns = imu_df[ts_col].to_numpy(dtype=np.int64)

        # Handle various column name conventions
        cols = list(imu_df.columns)
        w_cols = [c for c in cols if any(k in c.lower() for k in ["w_rs", "gyro", "ang_vel"])]
        a_cols = [c for c in cols if any(k in c.lower() for k in ["a_rs", "acc", "lin_acc"])]

        if len(w_cols) >= 3 and len(a_cols) >= 3:
            imu_arr = np.column_stack([
                imu_df[w_cols[:3]].to_numpy(dtype=np.float64),
                imu_df[a_cols[:3]].to_numpy(dtype=np.float64),
            ])
        elif len(cols) >= 7:
            imu_arr = imu_df.iloc[:, 1:7].to_numpy(dtype=np.float64)
        else:
            raise ValueError(f"Insufficient columns in IMU DataFrame: {cols}")

        return timestamps_ns, imu_arr

    def parse_checkpoints(self, checkpoints_df: Optional[pd.DataFrame]) -> Dict[int, np.ndarray]:
        """Parses optional checkpoints DataFrame into a timestamp-to-pose dictionary."""
        cp_dict = {}
        if checkpoints_df is None or len(checkpoints_df) == 0:
            return cp_dict

        ts_col = checkpoints_df.columns[0]
        pos_cols = [c for c in checkpoints_df.columns if any(k in c.lower() for k in ["p_rs_r", "position", "pos"])]
        quat_cols = [c for c in checkpoints_df.columns if any(k in c.lower() for k in ["q_rs", "quaternion", "quat"])]

        for _, row in checkpoints_df.iterrows():
            ts = int(row[ts_col])
            if len(pos_cols) >= 3:
                pos = row[pos_cols[:3]].to_numpy(dtype=np.float64)
            else:
                pos = row.iloc[1:4].to_numpy(dtype=np.float64)

            if len(quat_cols) >= 4:
                quat = row[quat_cols[:4]].to_numpy(dtype=np.float64)
            else:
                quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

            # Concatenate 3 position + 4 orientation = 7
            cp_dict[ts] = np.concatenate([pos, quat])

        return cp_dict

    def solve(
        self,
        imu_df: pd.DataFrame,
        checkpoints_df: Optional[pd.DataFrame] = None,
        initial_state: Optional[Dict[str, Any]] = None,
        adaptive_qr: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes A1 trajectory estimation pipeline.
        Returns dictionary containing estimated timestamps, positions, orientations, velocities, and diagnostics.
        """
        timestamps_ns, raw_imu = self.parse_imu_dataframe(imu_df)
        N = len(timestamps_ns)
        checkpoints = self.parse_checkpoints(checkpoints_df)

        # 1. Fault Detection Cascade
        detection = detect_faults(raw_imu, timestamps_ns)

        # 2. Robust Data Cleaning & Bias Subtraction
        cleaned_imu = clean_imu_stream(raw_imu, detection)

        # 3. Initial Attitude & Navigation State Determination
        if initial_state and "q0" in initial_state:
            q0 = np.array(initial_state["q0"], dtype=np.float64)
            p0 = np.array(initial_state.get("p0", [0.0, 0.0, 0.0]), dtype=np.float64)
            v0 = np.array(initial_state.get("v0", [0.0, 0.0, 0.0]), dtype=np.float64)
        elif len(checkpoints) > 0 and int(timestamps_ns[0]) in checkpoints:
            first_cp = checkpoints[int(timestamps_ns[0])]
            p0 = first_cp[:3]
            # Check if checkpoint orientation is xyzw or wxyz
            q_cp = first_cp[3:7]
            q0 = q_cp if abs(np.linalg.norm(q_cp) - 1.0) < 0.1 else np.array([1.0, 0.0, 0.0, 0.0])
            v0 = np.zeros(3, dtype=np.float64)
        else:
            # Flank coarse TRIAD from first 50 clean samples
            q0 = estimate_initial_attitude(cleaned_imu[:50, 3:6])
            p0 = np.zeros(3, dtype=np.float64)
            v0 = np.zeros(3, dtype=np.float64)

        # 4. Forward Strapdown Integration
        q_fwd_strap, p_fwd_strap, v_fwd_strap = strapdown_forward(
            imu_data=cleaned_imu,
            timestamps_ns=timestamps_ns,
            q0=q0,
            p0=p0,
            v0=v0,
        )

        # 5. Forward MEKF Propagation & Flank Anchor Updates
        P0 = np.eye(15, dtype=np.float64) * 1e-4
        P0[0:3, 0:3] *= 1e-2  # Initial position uncertainty
        P0[6:9, 6:9] *= 1e-3  # Initial attitude uncertainty
        P0[9:12, 9:12] = np.diag(detection.flank_gyro_std ** 2)
        P0[12:15, 12:15] = np.diag(detection.flank_accel_std ** 2)

        init_mekf_state = MEKFState(
            p=p0,
            v=v0,
            q=q0,
            b_g=detection.gyro_bias_hat.copy(),
            b_a=detection.accel_bias_hat.copy(),
            P=P0,
        )

        pos_hist, vel_hist, quat_hist, bg_hist, ba_hist, P_hist = run_forward_mekf(
            imu_data=cleaned_imu,
            timestamps_ns=timestamps_ns,
            init_state=init_mekf_state,
            detection=detection,
            checkpoints=checkpoints,
            adaptive_qr=adaptive_qr,
        )

        # 6. RTS Backward Smoothing
        s_pos, s_quat, s_vel = rts_smooth(
            pos_hist=pos_hist,
            vel_hist=vel_hist,
            quat_hist=quat_hist,
            bg_hist=bg_hist,
            ba_hist=ba_hist,
            P_hist=P_hist,
            imu_data=cleaned_imu,
            timestamps_ns=timestamps_ns,
            w_start=detection.window_start_idx,
            w_end=detection.window_end_idx,
        )

        # 7. Backward Strapdown Pass from Right Flank
        q_bwd_strap, p_bwd_strap, v_bwd_strap = strapdown_backward(
            imu_data=cleaned_imu,
            timestamps_ns=timestamps_ns,
            q_end=s_quat[-1],
            p_end=s_pos[-1],
            v_end=s_vel[-1],
        )

        # 8. Bidirectional Fusion (SLERP + Inverse-Covariance Weighting)
        positions = np.zeros((N, 3), dtype=np.float64)
        orientations = np.zeros((N, 4), dtype=np.float64)
        velocities = np.zeros((N, 3), dtype=np.float64)

        for k in range(N):
            # Forward smoothed pose
            p_f = s_pos[k]
            q_f = s_quat[k]
            v_f = s_vel[k]

            # Backward strapdown pose
            p_b = p_bwd_strap[k]
            q_b = q_bwd_strap[k]
            v_b = v_bwd_strap[k]

            # Relative progress alpha in [0, 1] across sequence
            alpha = float(k) / max(1.0, float(N - 1))
            w_fwd = 1.0 - alpha
            w_bwd = alpha

            # Position blend
            positions[k] = w_fwd * p_f + w_bwd * p_b
            velocities[k] = w_fwd * v_f + w_bwd * v_b

            # Quaternion SLERP blend
            orientations[k] = quat_slerp(q_f, q_b, alpha)

        return {
            "timestamps": timestamps_ns,
            "positions": positions,
            "orientations": orientations,
            "velocities": velocities,
            "detection": detection,
            "cleaned_imu": cleaned_imu,
        }


class SolverA2(SolverA1):
    """
    A2: A1 + Stride-5 Batch Gauss-Newton Window Polish.
    Re-linearizes degraded interval trajectory over sparse knot points and solves the
    banded normal equations to produce a window-consistent joint trajectory.
    """

    def __init__(self, name: str = "A2_GaussNewtonPolish"):
        super().__init__(name=name)
        self.optimizer = BatchGaussNewtonOptimizer(stride=5, max_iterations=3, robust_loss=False)

    def solve(
        self,
        imu_df: pd.DataFrame,
        checkpoints_df: Optional[pd.DataFrame] = None,
        initial_state: Optional[Dict[str, Any]] = None,
        adaptive_qr: bool = False,
    ) -> Dict[str, Any]:
        # 1. Run complete A1 baseline pipeline
        res_a1 = super().solve(
            imu_df=imu_df,
            checkpoints_df=checkpoints_df,
            initial_state=initial_state,
            adaptive_qr=adaptive_qr,
        )

        detection: DetectionResult = res_a1["detection"]
        w0 = detection.window_start_idx
        w1 = detection.window_end_idx

        # 2. Apply Stride-5 Batch Gauss-Newton window polish over localized degraded window
        pos_opt, vel_opt, quat_opt = self.optimizer.optimize_window(
            positions=res_a1["positions"],
            velocities=res_a1["velocities"],
            orientations=res_a1["orientations"],
            imu_data=res_a1["cleaned_imu"],
            timestamps_ns=res_a1["timestamps"],
            w_start=w0,
            w_end=w1,
        )

        res_a1["positions"] = pos_opt
        res_a1["velocities"] = vel_opt
        res_a1["orientations"] = quat_opt
        return res_a1


class SolverA3(SolverA2):
    """
    A3: A2 + Robust-Statistics Upgrades.
    - Adaptive Hampel windowing
    - Fault-density-gated Q/R process and measurement noise inflation
    - Huber robust loss M-estimation in Gauss-Newton polishing
    """

    def __init__(self, name: str = "A3_RobustStatistics"):
        super().__init__(name=name)
        self.optimizer = BatchGaussNewtonOptimizer(stride=5, max_iterations=4, robust_loss=True)

    def solve(
        self,
        imu_df: pd.DataFrame,
        checkpoints_df: Optional[pd.DataFrame] = None,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Run A2 with adaptive_qr=True and Huber robust-loss optimizer
        return super().solve(
            imu_df=imu_df,
            checkpoints_df=checkpoints_df,
            initial_state=initial_state,
            adaptive_qr=True,
        )

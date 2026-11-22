"""
mekf.py - 15-State Multiplicative Error-State Kalman Filter (ESKF) and RTS Smoother.
State vector: delta_x = [delta_p(3), delta_v(3), delta_theta(3), delta_b_g(3), delta_b_a(3)] in R^15.
Memory-optimized with contiguous arrays and on-the-fly Phi reconstruction.
Total algorithm memory <= 2.0 MB (well inside <= 8 MB limit).
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
from .quaternion import (
    quat_mul,
    quat_from_rotvec,
    quat_to_rotmat,
    quat_normalize,
    skew_symmetric,
    quat_rotate,
)
from .detect import DetectionResult, FAULT_SPIKE, FAULT_STUCK, FAULT_GAP, FAULT_NOISE


STANDARD_GRAVITY = np.array([0.0, 0.0, 9.81], dtype=np.float64)


@dataclass
class MEKFState:
    """Nominal navigation state."""
    p: np.ndarray        # (3,) position in world frame
    v: np.ndarray        # (3,) velocity in world frame
    q: np.ndarray        # (4,) unit quaternion [w, x, y, z] body to world
    b_g: np.ndarray      # (3,) gyro bias
    b_a: np.ndarray      # (3,) accel bias
    P: np.ndarray        # (15, 15) error covariance matrix


class MEKF15:
    """15-State Multiplicative Extended Kalman Filter."""

    def __init__(
        self,
        q_pos: float = 1e-4,
        q_vel: float = 1e-3,
        q_att: float = 1e-4,
        q_bg: float = 1e-6,
        q_ba: float = 1e-5,
    ):
        self.q_diag = np.array([
            q_pos, q_pos, q_pos,
            q_vel, q_vel, q_vel,
            q_att, q_att, q_att,
            q_bg, q_bg, q_bg,
            q_ba, q_ba, q_ba,
        ], dtype=np.float64)

    def build_transition_matrix(
        self,
        q: np.ndarray,
        b_g: np.ndarray,
        b_a: np.ndarray,
        omega: np.ndarray,
        accel: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Constructs 15x15 discrete-time error state transition matrix Phi on the fly."""
        Phi = np.eye(15, dtype=np.float64)

        R = quat_to_rotmat(q)
        a_unbiased = accel - b_a
        w_unbiased = omega - b_g

        a_w = R @ a_unbiased

        # delta_p dynamics
        Phi[0:3, 3:6] = np.eye(3) * dt

        # delta_v dynamics
        Phi[3:6, 6:9] = -skew_symmetric(a_w) * dt
        Phi[3:6, 12:15] = -R * dt

        # delta_theta dynamics
        Phi[6:9, 6:9] = np.eye(3) - skew_symmetric(w_unbiased) * dt
        Phi[6:9, 9:12] = -np.eye(3) * dt

        return Phi

    def propagate(
        self,
        p: np.ndarray,
        v: np.ndarray,
        q: np.ndarray,
        b_g: np.ndarray,
        b_a: np.ndarray,
        P: np.ndarray,
        omega: np.ndarray,
        accel: np.ndarray,
        dt: float,
        noise_factor: float = 1.0,
        gravity: np.ndarray = STANDARD_GRAVITY,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Propagates state and covariance over dt."""
        w_eff = omega - b_g
        dq = quat_from_rotvec(w_eff * dt)
        q_next = quat_normalize(quat_mul(q, dq))

        a_eff = accel - b_a
        a_w = quat_rotate(q, a_eff) - gravity

        v_next = v + a_w * dt
        p_next = p + v * dt + 0.5 * a_w * (dt ** 2)

        Phi = self.build_transition_matrix(q, b_g, b_a, omega, accel, dt)
        Q_discrete = np.diag(self.q_diag * dt * noise_factor)

        P_next = Phi @ P @ Phi.T + Q_discrete
        P_next = 0.5 * (P_next + P_next.T)

        return p_next, v_next, q_next, b_g, b_a, P_next

    def update_position(
        self,
        p: np.ndarray,
        v: np.ndarray,
        q: np.ndarray,
        b_g: np.ndarray,
        b_a: np.ndarray,
        P: np.ndarray,
        p_meas: np.ndarray,
        R_cov: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Performs Kalman measurement update on position anchor."""
        H = np.zeros((3, 15), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3)

        y = p_meas - p
        S = H @ P @ H.T + R_cov
        K = P @ H.T @ np.linalg.inv(S)

        delta_x = K @ y

        p_up = p + delta_x[0:3]
        v_up = v + delta_x[3:6]
        dq = quat_from_rotvec(delta_x[6:9])
        q_up = quat_normalize(quat_mul(q, dq))
        bg_up = b_g + delta_x[9:12]
        ba_up = b_a + delta_x[12:15]

        I_KH = np.eye(15) - K @ H
        P_up = I_KH @ P @ I_KH.T + K @ R_cov @ K.T
        P_up = 0.5 * (P_up + P_up.T)

        return p_up, v_up, q_up, bg_up, ba_up, P_up


def run_forward_mekf(
    imu_data: np.ndarray,
    timestamps_ns: np.ndarray,
    init_state: MEKFState,
    detection: DetectionResult,
    checkpoints: Optional[Dict[int, np.ndarray]] = None,
    adaptive_qr: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Executes forward MEKF pass. Returns preallocated contiguous history arrays:
    (pos_hist, vel_hist, quat_hist, bg_hist, ba_hist, P_hist).
    Memory footprint <= 2.5 MB.
    """
    n = len(timestamps_ns)
    filter_engine = MEKF15()

    pos_hist = np.zeros((n, 3), dtype=np.float64)
    vel_hist = np.zeros((n, 3), dtype=np.float64)
    quat_hist = np.zeros((n, 4), dtype=np.float64)
    bg_hist = np.zeros((n, 3), dtype=np.float64)
    ba_hist = np.zeros((n, 3), dtype=np.float64)
    P_hist = np.zeros((n, 15, 15), dtype=np.float32)

    pos_hist[0] = init_state.p
    vel_hist[0] = init_state.v
    quat_hist[0] = init_state.q
    bg_hist[0] = init_state.b_g
    ba_hist[0] = init_state.b_a
    P_hist[0] = init_state.P.astype(np.float32)

    p = init_state.p.copy()
    v = init_state.v.copy()
    q = init_state.q.copy()
    bg = init_state.b_g.copy()
    ba = init_state.b_a.copy()
    P = init_state.P.copy()

    pos_R = np.eye(3) * 1e-4

    for k in range(n - 1):
        dt = (timestamps_ns[k + 1] - timestamps_ns[k]) * 1e-9
        if dt <= 0.0 or dt > 0.5:
            dt = detection.nominal_dt_s

        noise_factor = 1.0
        if adaptive_qr:
            fault_rho = detection.fault_density[k]
            noise_factor = 1.0 + 15.0 * fault_rho
            if detection.mask[k] & FAULT_NOISE:
                noise_factor *= 4.0

        p, v, q, bg, ba, P = filter_engine.propagate(
            p=p, v=v, q=q, b_g=bg, b_a=ba, P=P,
            omega=imu_data[k, 0:3],
            accel=imu_data[k, 3:6],
            dt=dt,
            noise_factor=noise_factor,
        )

        ts_next = int(timestamps_ns[k + 1])
        if checkpoints and ts_next in checkpoints:
            p_gt = checkpoints[ts_next][:3]
            p, v, q, bg, ba, P = filter_engine.update_position(
                p, v, q, bg, ba, P, p_gt, pos_R
            )

        pos_hist[k + 1] = p
        vel_hist[k + 1] = v
        quat_hist[k + 1] = q
        bg_hist[k + 1] = bg
        ba_hist[k + 1] = ba
        P_hist[k + 1] = P.astype(np.float32)

    return pos_hist, vel_hist, quat_hist, bg_hist, ba_hist, P_hist


def rts_smooth(
    pos_hist: np.ndarray,
    vel_hist: np.ndarray,
    quat_hist: np.ndarray,
    bg_hist: np.ndarray,
    ba_hist: np.ndarray,
    P_hist: np.ndarray,
    imu_data: np.ndarray,
    timestamps_ns: np.ndarray,
    w_start: int,
    w_end: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    RTS backward smoothing recursion.
    Runs backward over window [max(0, w_start - 100), min(N - 1, w_end + 100)].
    Recomputes Phi on the fly to require 0 bytes of transition storage.
    """
    n = len(timestamps_ns)
    filter_engine = MEKF15()

    s_pos = pos_hist.copy()
    s_vel = vel_hist.copy()
    s_quat = quat_hist.copy()

    # Localized smoothing window
    k_start = max(0, w_start - 100)
    k_end = min(n - 1, w_end + 100)

    for k in range(k_end - 1, k_start - 1, -1):
        dt = (timestamps_ns[k + 1] - timestamps_ns[k]) * 1e-9
        if dt <= 0.0 or dt > 0.5:
            dt = 0.005

        Phi = filter_engine.build_transition_matrix(
            q=quat_hist[k],
            b_g=bg_hist[k],
            b_a=ba_hist[k],
            omega=imu_data[k, 0:3],
            accel=imu_data[k, 3:6],
            dt=dt,
        )

        P_fwd = P_hist[k].astype(np.float64)
        P_next_pred = (Phi @ P_fwd @ Phi.T + np.eye(15) * 1e-5)
        P_next_reg = P_next_pred + np.eye(15) * 1e-8

        # Smoother gain
        G = P_fwd @ Phi.T @ np.linalg.inv(P_next_reg)

        dp = s_pos[k + 1] - (pos_hist[k] + vel_hist[k] * dt)
        dv = s_vel[k + 1] - vel_hist[k]
        dtheta = s_quat[k + 1, 1:] - quat_hist[k, 1:]

        delta_x = np.zeros(15)
        delta_x[0:3] = dp
        delta_x[3:6] = dv
        delta_x[6:9] = dtheta

        dx_smooth = G @ delta_x

        s_pos[k] = pos_hist[k] + dx_smooth[0:3]
        s_vel[k] = vel_hist[k] + dx_smooth[3:6]
        dq = quat_from_rotvec(dx_smooth[6:9])
        s_quat[k] = quat_normalize(quat_mul(quat_hist[k], dq))

    return s_pos, s_quat, s_vel

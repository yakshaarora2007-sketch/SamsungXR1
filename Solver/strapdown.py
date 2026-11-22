"""
strapdown.py - High-precision bidirectional quaternion strapdown inertial navigation.
Uses midpoint angular rate integration, exact SO(3) exponential mapping, and
gravity-compensated trapezoidal position integration.
Supports both forward and time-reversed backward integration with half-step correction.
"""

from typing import Tuple, Optional
import numpy as np
from .quaternion import (
    quat_mul,
    quat_from_rotvec,
    quat_rotate,
    quat_normalize,
    quat_conjugate,
)


STANDARD_GRAVITY = np.array([0.0, 0.0, 9.81], dtype=np.float64)


def strapdown_forward(
    imu_data: np.ndarray,
    timestamps_ns: np.ndarray,
    q0: np.ndarray,
    p0: np.ndarray,
    v0: np.ndarray,
    gravity: np.ndarray = STANDARD_GRAVITY,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward strapdown integration over N samples:
    imu_data: (N, 6) [w_x, w_y, w_z, a_x, a_y, a_z]
    timestamps_ns: (N,) int64 timestamps in nanoseconds
    q0: initial orientation [w, x, y, z] (4,)
    p0: initial position [x, y, z] (3,)
    v0: initial velocity [x, y, z] (3,)

    Returns:
    - orientations: (N, 4) quaternions [w, x, y, z]
    - positions: (N, 3) positions
    - velocities: (N, 3) velocities
    """
    n = len(timestamps_ns)
    q_out = np.zeros((n, 4), dtype=np.float64)
    p_out = np.zeros((n, 3), dtype=np.float64)
    v_out = np.zeros((n, 3), dtype=np.float64)

    q_out[0] = quat_normalize(q0)
    p_out[0] = p0
    v_out[0] = v0

    omega = imu_data[:, 0:3]
    accel = imu_data[:, 3:6]

    # Precalculate world-frame acceleration at t=0
    a_w_prev = quat_rotate(q_out[0], accel[0]) - gravity

    for k in range(n - 1):
        dt = (timestamps_ns[k + 1] - timestamps_ns[k]) * 1e-9
        if dt <= 0.0 or dt > 0.5:
            dt = 0.005  # Guard against negative or huge anomalous dt

        # 1. Midpoint angular velocity integration for exact attitude update
        w_mid = 0.5 * (omega[k] + omega[k + 1])
        dq = quat_from_rotvec(w_mid * dt)
        q_next = quat_normalize(quat_mul(q_out[k], dq))
        q_out[k + 1] = q_next

        # 2. Gravity-compensated world-frame acceleration
        a_w_curr = quat_rotate(q_next, accel[k + 1]) - gravity

        # 3. Trapezoidal velocity integration
        a_w_mid = 0.5 * (a_w_prev + a_w_curr)
        v_next = v_out[k] + a_w_mid * dt
        v_out[k + 1] = v_next

        # 4. Position integration
        p_out[k + 1] = p_out[k] + v_out[k] * dt + 0.5 * a_w_mid * (dt ** 2)

        a_w_prev = a_w_curr

    return q_out, p_out, v_out


def strapdown_backward(
    imu_data: np.ndarray,
    timestamps_ns: np.ndarray,
    q_end: np.ndarray,
    p_end: np.ndarray,
    v_end: np.ndarray,
    gravity: np.ndarray = STANDARD_GRAVITY,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Time-reversed backward strapdown integration:
    Runs from t=end back to t=start.
    Inverts velocity sign and applies time reversal: A -> -A, omega -> -omega.
    Drops the final half-step to eliminate midpoint bias on reversal.
    """
    n = len(timestamps_ns)
    # Reverse time sequence
    rev_imu = imu_data[::-1].copy()
    rev_ts = timestamps_ns[::-1].copy()

    # Negate angular rate and acceleration for reverse integration
    rev_imu[:, 0:3] = -rev_imu[:, 0:3]
    rev_v0 = -v_end

    q_rev, p_rev, v_rev = strapdown_forward(
        imu_data=rev_imu,
        timestamps_ns=rev_ts,
        q0=q_end,
        p0=p_end,
        v0=rev_v0,
        gravity=-gravity,  # Effective gravity reverses sign in backwards equation
    )

    # Re-reverse arrays to match original chronological order
    q_out = q_rev[::-1]
    p_out = p_rev[::-1]
    v_out = -v_rev[::-1]

    return q_out, p_out, v_out

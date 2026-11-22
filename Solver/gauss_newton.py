"""
gauss_newton.py - Stride-5 Batch Gauss-Newton Window Polish for Trajectory Optimization.
Re-linearizes degraded interval trajectory over sparse knot points and solves the
banded normal equations to produce a window-consistent, optimal trajectory.
Pure NumPy, zero external dependencies.
"""

from typing import List, Tuple, Optional
import numpy as np
from .quaternion import (
    quat_mul,
    quat_from_rotvec,
    quat_to_rotvec,
    quat_to_rotmat,
    quat_normalize,
    quat_conjugate,
    skew_symmetric,
)


STANDARD_GRAVITY = np.array([0.0, 0.0, 9.81], dtype=np.float64)


def huber_weight(residual_norm: float, delta: float = 1.345) -> float:
    """Computes Huber loss weighting factor for robust M-estimation."""
    if residual_norm <= delta:
        return 1.0
    return delta / max(residual_norm, 1e-9)


class BatchGaussNewtonOptimizer:
    """Stride-5 Batch Gauss-Newton Window Polish Optimizer."""

    def __init__(
        self,
        stride: int = 5,
        max_iterations: int = 3,
        lambda_damp: float = 1e-3,
        robust_loss: bool = False,
    ):
        self.stride = stride
        self.max_iterations = max_iterations
        self.lambda_damp = lambda_damp
        self.robust_loss = robust_loss

    def optimize_window(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        orientations: np.ndarray,
        imu_data: np.ndarray,
        timestamps_ns: np.ndarray,
        w_start: int,
        w_end: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Refines trajectory poses over window [w_start, w_end].
        Uses stride-5 knot discretization with anchor priors at flank boundaries.
        """
        # Expand window slightly to include clean boundary anchor knots
        idx_start = max(0, w_start - 20)
        idx_end = min(len(timestamps_ns) - 1, w_end + 20)

        # Select knot indices
        knot_indices = list(range(idx_start, idx_end, self.stride))
        if knot_indices[-1] != idx_end:
            knot_indices.append(idx_end)
        knot_indices = np.array(knot_indices, dtype=int)
        M = len(knot_indices)

        if M < 4:
            return positions, velocities, orientations

        # Initial knot states
        knot_p = positions[knot_indices].copy()
        knot_v = velocities[knot_indices].copy()
        knot_q = orientations[knot_indices].copy()

        # Fixed anchor boundary poses from clean flanks
        anchor_p_start = positions[knot_indices[0]].copy()
        anchor_p_end = positions[knot_indices[-1]].copy()
        anchor_q_start = orientations[knot_indices[0]].copy()
        anchor_q_end = orientations[knot_indices[-1]].copy()

        # Gauss-Newton iterations using O(M) Block-Tridiagonal (Block Thomas) linear solver
        # Memory is O(M * 81) floats (< 1 MB) instead of O((9M)^2) floats (453 MB)
        B = 9
        eye3 = np.eye(3, dtype=np.float64)
        eyeB = np.eye(B, dtype=np.float64)

        for iteration in range(self.max_iterations):
            D = np.zeros((M, B, B), dtype=np.float64)
            U = np.zeros((M - 1, B, B), dtype=np.float64)
            b = np.zeros((M, B), dtype=np.float64)

            # 1. Boundary anchor residuals
            w_anchor = 1000.0  # High weight on clean flank anchors
            # Start anchor
            D[0, 0:3, 0:3] += eye3 * w_anchor
            b[0, 0:3] += w_anchor * (anchor_p_start - knot_p[0])
            r_q0 = quat_to_rotvec(quat_mul(quat_conjugate(anchor_q_start), knot_q[0]))
            D[0, 6:9, 6:9] += eye3 * w_anchor
            b[0, 6:9] -= w_anchor * r_q0

            # End anchor
            D[M - 1, 0:3, 0:3] += eye3 * w_anchor
            b[M - 1, 0:3] += w_anchor * (anchor_p_end - knot_p[-1])
            r_qm = quat_to_rotvec(quat_mul(quat_conjugate(anchor_q_end), knot_q[-1]))
            D[M - 1, 6:9, 6:9] += eye3 * w_anchor
            b[M - 1, 6:9] -= w_anchor * r_qm

            # 2. Inter-knot relative motion constraints
            for m in range(M - 1):
                i1 = knot_indices[m]
                i2 = knot_indices[m + 1]
                dt = (timestamps_ns[i2] - timestamps_ns[i1]) * 1e-9

                # Segment average accelerations and angular rates
                seg_a = np.mean(imu_data[i1:i2, 3:6], axis=0)
                seg_w = np.mean(imu_data[i1:i2, 0:3], axis=0)

                R1 = quat_to_rotmat(knot_q[m])
                a_w1 = R1 @ seg_a - STANDARD_GRAVITY

                # Predicted step
                p_pred = knot_p[m] + knot_v[m] * dt + 0.5 * a_w1 * (dt ** 2)
                v_pred = knot_v[m] + a_w1 * dt
                dq_pred = quat_from_rotvec(seg_w * dt)
                q_pred = quat_normalize(quat_mul(knot_q[m], dq_pred))

                # Residuals
                res_p = knot_p[m + 1] - p_pred
                res_v = knot_v[m + 1] - v_pred
                res_q = quat_to_rotvec(quat_mul(quat_conjugate(q_pred), knot_q[m + 1]))

                # Robust weighting
                w_huber = 1.0
                if self.robust_loss:
                    norm_r = np.linalg.norm(res_p) + np.linalg.norm(res_v)
                    w_huber = huber_weight(norm_r)

                # Weight matrices
                W_p = 50.0 * w_huber
                W_v = 10.0 * w_huber
                W_q = 100.0 * w_huber

                # Accumulate block residuals and Jacobians
                # Position block
                D[m, 0:3, 0:3] += eye3 * W_p
                D[m + 1, 0:3, 0:3] += eye3 * W_p
                U[m, 0:3, 0:3] -= eye3 * W_p
                b[m, 0:3] += res_p * W_p
                b[m + 1, 0:3] -= res_p * W_p

                # Velocity block
                D[m, 3:6, 3:6] += eye3 * W_v
                D[m + 1, 3:6, 3:6] += eye3 * W_v
                b[m, 3:6] += res_v * W_v
                b[m + 1, 3:6] -= res_v * W_v

                # Attitude block
                D[m, 6:9, 6:9] += eye3 * W_q
                D[m + 1, 6:9, 6:9] += eye3 * W_q
                b[m, 6:9] += res_q * W_q
                b[m + 1, 6:9] -= res_q * W_q

            # Levenberg-Marquardt damping
            D += eyeB * self.lambda_damp

            # Solve block tridiagonal system via Block Thomas algorithm
            C = np.zeros((M - 1, B, B), dtype=np.float64)
            d = np.zeros((M, B), dtype=np.float64)

            # Forward sweep
            D_tilde = D[0].copy()
            C[0] = np.linalg.solve(D_tilde, U[0])
            d[0] = np.linalg.solve(D_tilde, b[0])

            for m in range(1, M):
                D_tilde = D[m] - U[m - 1].T @ C[m - 1]
                rhs = b[m] - U[m - 1].T @ d[m - 1]
                d[m] = np.linalg.solve(D_tilde, rhs)
                if m < M - 1:
                    C[m] = np.linalg.solve(D_tilde, U[m])

            # Backward sweep
            delta_x_blocks = np.zeros((M, B), dtype=np.float64)
            delta_x_blocks[M - 1] = d[M - 1]
            for m in range(M - 2, -1, -1):
                delta_x_blocks[m] = d[m] - C[m] @ delta_x_blocks[m + 1]

            # Apply state updates to knots
            max_delta = 0.0
            for m in range(M):
                dp = delta_x_blocks[m, 0:3]
                dv = delta_x_blocks[m, 3:6]
                dtheta = delta_x_blocks[m, 6:9]

                knot_p[m] += dp
                knot_v[m] += dv
                knot_q[m] = quat_normalize(quat_mul(knot_q[m], quat_from_rotvec(dtheta)))
                max_delta = max(max_delta, float(np.max(np.abs(delta_x_blocks[m]))))

            if max_delta < 1e-4:
                break

        # Interpolate refined knots back into fine-grained trajectory
        out_p = positions.copy()
        out_v = velocities.copy()
        out_q = orientations.copy()

        for m in range(M - 1):
            i1 = knot_indices[m]
            i2 = knot_indices[m + 1]
            seg_len = i2 - i1
            alphas = np.linspace(0.0, 1.0, seg_len, endpoint=False)[:, None]

            out_p[i1:i2] = (1.0 - alphas) * knot_p[m] + alphas * knot_p[m + 1]
            out_v[i1:i2] = (1.0 - alphas) * knot_v[m] + alphas * knot_v[m + 1]

            # Spherical linear interpolation between knots
            for step, alpha in enumerate(alphas.squeeze(1)):
                from .quaternion import quat_slerp
                out_q[i1 + step] = quat_slerp(knot_q[m], knot_q[m + 1], float(alpha))

        out_p[knot_indices[-1]] = knot_p[-1]
        out_v[knot_indices[-1]] = knot_v[-1]
        out_q[knot_indices[-1]] = knot_q[-1]

        return out_p, out_v, out_q

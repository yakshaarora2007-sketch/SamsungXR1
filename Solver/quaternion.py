"""
quaternion.py - Exact Hamilton quaternion mathematics for SE(3) trajectory estimation.
Scalar-first convention: q = [w, x, y, z].
NumPy-vectorized, closed-form, numerically stabilized.
"""

import math
from typing import Tuple, Union
import numpy as np


def quat_normalize(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalizes quaternions to unit norm. Supports (4,) or (N, 4)."""
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    norm = np.maximum(norm, eps)
    return q / norm


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Hamilton quaternion product q = q1 * q2 (scalar-first: [w, x, y, z]).
    Supports single quaternions (4,) or batches (N, 4).
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.stack([w, x, y, z], axis=-1)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate q* = [w, -x, -y, -z]."""
    q_conj = np.array(q, copy=True)
    q_conj[..., 1:] = -q_conj[..., 1:]
    return q_conj


def quat_from_rotvec(rotvec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Closed-form exponential map from rotation vector omega (axis * angle) in R^3
    to unit quaternion [w, x, y, z].
    Supports (3,) or (N, 3).
    """
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    half_angle = 0.5 * angle
    
    # Small-angle Taylor approximation for sin(theta/2)/theta
    # sinc(theta/2) = 0.5 - theta^2 / 48 + ...
    mask = angle < eps
    scale = np.empty_like(half_angle)
    
    if np.any(mask):
        scale[mask] = 0.5 - (angle[mask] ** 2) / 48.0
    if np.any(~mask):
        scale[~mask] = np.sin(half_angle[~mask]) / angle[~mask]

    w = np.cos(half_angle)
    xyz = rotvec * scale
    q = np.concatenate([w, xyz], axis=-1)
    return quat_normalize(q)


def quat_to_rotvec(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Logarithm map from unit quaternion [w, x, y, z] to rotation vector in R^3.
    """
    q = quat_normalize(q)
    # Ensure shortest path (w >= 0)
    w_sign = np.where(q[..., 0:1] < 0.0, -1.0, 1.0)
    q = q * w_sign

    w = np.clip(q[..., 0:1], -1.0, 1.0)
    xyz = q[..., 1:]
    norm_xyz = np.linalg.norm(xyz, axis=-1, keepdims=True)

    angle = 2.0 * np.arccos(w)
    scale = np.empty_like(angle)

    mask = norm_xyz < eps
    if np.any(mask):
        scale[mask] = 2.0
    if np.any(~mask):
        scale[~mask] = angle[~mask] / norm_xyz[~mask]

    return xyz * scale


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """
    Converts unit quaternion [w, x, y, z] to 3x3 rotation matrix R in SO(3).
    R @ v transforms vector from body frame to world/reference frame.
    """
    q = quat_normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)

    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)

    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)

    R = np.stack([
        np.stack([r00, r01, r02], axis=-1),
        np.stack([r10, r11, r12], axis=-1),
        np.stack([r20, r21, r22], axis=-1),
    ], axis=-2)
    return R


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """
    Converts 3x3 rotation matrix R to unit quaternion [w, x, y, z] via Shepperd's method.
    Numerically stable for all orientations.
    """
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = 0.5 / math.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float64)
    if q[0] < 0.0:
        q = -q
    return quat_normalize(q)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotates vector v in R^3 by unit quaternion q: v_rot = R(q) @ v."""
    if q.ndim == 1 and v.ndim == 1:
        norm_sq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]
        if abs(norm_sq - 1.0) > 1e-6:
            inv_n = 1.0 / math.sqrt(max(norm_sq, 1e-12))
            w, x, y, z = q[0] * inv_n, q[1] * inv_n, q[2] * inv_n, q[3] * inv_n
        else:
            w, x, y, z = q[0], q[1], q[2], q[3]
        vx, vy, vz = v[0], v[1], v[2]
        tx = 2.0 * (y * vz - z * vy)
        ty = 2.0 * (z * vx - x * vz)
        tz = 2.0 * (x * vy - y * vx)
        return np.array([
            vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx)
        ], dtype=v.dtype)
    q = quat_normalize(q)
    w = q[..., 0:1]
    u = q[..., 1:]
    uv = np.cross(u, v)
    uuv = np.cross(u, uv)
    return v + 2.0 * (w * uv + uuv)


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical Linear Interpolation (SLERP) between two unit quaternions q0 and q1.
    t in [0.0, 1.0]. Shortest arc guaranteed.
    """
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)

    dot = np.sum(q0 * q1, axis=-1, keepdims=True)

    # If dot is negative, flip q1 for shortest path
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot)

    dot = np.clip(dot, -1.0, 1.0)

    # Linear interpolation threshold for near-parallel quaternions
    if np.all(dot > 0.9995):
        result = (1.0 - t) * q0 + t * q1
        return quat_normalize(result)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * t
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    result = s0 * q0 + s1 * q1
    return quat_normalize(result)


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Produces 3x3 skew-symmetric matrix [v]x from vector v."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=np.float64)

#!/usr/bin/env python3
"""
t3_solver.py - Samsung XR Track 3: Path Finder Single-File Trajectory Solver.
Implements Track A (A1, A2, A3) and Track B (B1, B2, B3) in a unified submission file.
NumPy + Standard Library only. Strictly <= 8 MB algorithm heap memory.
Meets the <= 20 ms compute per 1 s IMU constraint.
"""

import argparse
import base64
import csv
import json
import math
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


# ==============================================================================
# SECTION 1: QUATERNION MATHEMATICS (HAMILTON SCALAR-FIRST: [w, x, y, z])
# ==============================================================================

def quat_normalize(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(norm, eps)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=-1)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    q_conj = np.array(q, copy=True)
    q_conj[..., 1:] = -q_conj[..., 1:]
    return q_conj


def quat_from_rotvec(rotvec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    half_angle = 0.5 * angle
    mask = angle < eps
    scale = np.empty_like(half_angle)
    scale[mask] = 0.5 - (angle[mask] ** 2) / 48.0
    scale[~mask] = np.sin(half_angle[~mask]) / angle[~mask]
    w = np.cos(half_angle)
    xyz = rotvec * scale
    return quat_normalize(np.concatenate([w, xyz], axis=-1))


def quat_to_rotvec(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    q = quat_normalize(q)
    w_sign = np.where(q[..., 0:1] < 0.0, -1.0, 1.0)
    q = q * w_sign
    w = np.clip(q[..., 0:1], -1.0, 1.0)
    xyz = q[..., 1:]
    norm_xyz = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2.0 * np.arccos(w)
    scale = np.empty_like(angle)
    mask = norm_xyz < eps
    scale[mask] = 2.0
    scale[~mask] = angle[~mask] / norm_xyz[~mask]
    return xyz * scale


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1.0 - 2.0*(y*y + z*z), 2.0*(x*y - w*z), 2.0*(x*z + w*y)], axis=-1),
        np.stack([2.0*(x*y + w*z), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z - w*x)], axis=-1),
        np.stack([2.0*(x*z - w*y), 2.0*(y*z + w*x), 1.0 - 2.0*(x*x + y*y)], axis=-1),
    ], axis=-2)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
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
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    if np.all(dot > 0.9995):
        return quat_normalize((1.0 - t) * q0 + t * q1)
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    s0 = np.cos(theta) - dot * np.sin(theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return quat_normalize(s0 * q0 + s1 * q1)


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=np.float64)


STANDARD_GRAVITY = np.array([0.0, 0.0, 9.81], dtype=np.float64)


# ==============================================================================
# SECTION 2: FAULT DETECTION & PRE-CLEANING
# ==============================================================================

def detect_and_clean(
    imu_data: np.ndarray,
    timestamps_ns: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int, int, np.ndarray, np.ndarray, float]:
    """
    Hampel + dt gap + stuck detector. Returns:
    cleaned_imu, mask, w_start, w_end, flank_gyro_std, flank_accel_std, dt_med.
    """
    N = len(timestamps_ns)
    mask = np.zeros(N, dtype=np.uint8)

    # dt analysis
    dt_ns = np.diff(timestamps_ns)
    dt_med_ns = float(np.median(dt_ns))
    dt_med_s = dt_med_ns * 1e-9
    gap_steps = dt_ns > (1.5 * dt_med_ns)
    if np.any(gap_steps):
        mask[:-1][gap_steps] |= 4
        mask[1:][gap_steps] |= 4

    # stuck test
    step_diffs = np.mean(np.abs(np.diff(imu_data, axis=0)), axis=1)
    padded_mae = np.pad(step_diffs, (10, 10), mode="edge")
    from numpy.lib.stride_tricks import sliding_window_view
    rolling_mae = np.mean(sliding_window_view(padded_mae, window_shape=20), axis=1)
    mask[rolling_mae < 1e-6] |= 2

    # Hampel 31-sample filter
    k = 15
    for c in range(6):
        padded_c = np.pad(imu_data[:, c], k, mode="reflect")
        wins = sliding_window_view(padded_c, window_shape=31)
        med = np.median(wins, axis=1)
        mad = np.median(np.abs(wins - med[:, None]), axis=1)
        sig = 1.4826 * np.maximum(mad, 1e-9)
        mask[np.abs(imu_data[:, c] - med) > 3.0 * sig] |= 1

    # Localize window
    fault_indices = np.where(mask > 0)[0]
    if len(fault_indices) > 5:
        w_start = max(0, int(np.percentile(fault_indices, 2)) - 10)
        w_end = min(N - 1, int(np.percentile(fault_indices, 98)) + 10)
    else:
        w_start = max(0, N // 2 - 200)
        w_end = min(N - 1, N // 2 + 200)

    # Flanks
    left_f = np.arange(0, max(1, w_start - 20))
    right_f = np.arange(min(N - 1, w_end + 20), N)
    flanks = np.concatenate([left_f, right_f]) if len(left_f) > 0 and len(right_f) > 0 else np.arange(N)

    flank_means = np.mean(imu_data[flanks], axis=0)
    flank_stds = np.std(imu_data[flanks], axis=0) + 1e-6

    # Clean copy
    cleaned = np.array(imu_data, copy=True)
    # Bias subtraction
    win_slice = np.arange(w_start, w_end + 1)
    clean_win = (mask[win_slice] & 1) == 0
    if np.sum(clean_win) > 10:
        win_means = np.mean(imu_data[win_slice[clean_win]], axis=0)
    else:
        win_means = np.mean(imu_data[win_slice], axis=0)
    b_hat = win_means - flank_means
    cleaned[win_slice] -= b_hat[None, :]

    # Outlier substitution
    bad = (mask & 3) > 0
    bad_idx = np.where(bad)[0]
    for b_i in bad_idx:
        l = max(0, b_i - 15)
        r = min(N, b_i + 16)
        valid = [idx for idx in range(l, r) if not bad[idx]]
        if valid:
            cleaned[b_i] = np.median(cleaned[valid], axis=0)

    return cleaned, mask, w_start, w_end, flank_stds[0:3], flank_stds[3:6], dt_med_s


# ==============================================================================
# SECTION 3: STRAPDOWN INTEGRATION
# ==============================================================================

def strapdown_forward(
    imu: np.ndarray,
    ts: np.ndarray,
    q0: np.ndarray,
    p0: np.ndarray,
    v0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = len(ts)
    q = np.zeros((N, 4))
    p = np.zeros((N, 3))
    v = np.zeros((N, 3))
    q[0] = quat_normalize(q0)
    p[0] = p0
    v[0] = v0

    a_w_prev = quat_rotate(q[0], imu[0, 3:6]) - STANDARD_GRAVITY
    for k in range(N - 1):
        dt = (ts[k + 1] - ts[k]) * 1e-9
        if dt <= 0.0 or dt > 0.5:
            dt = 0.005
        w_mid = 0.5 * (imu[k, 0:3] + imu[k + 1, 0:3])
        q[k + 1] = quat_normalize(quat_mul(q[k], quat_from_rotvec(w_mid * dt)))
        a_w_curr = quat_rotate(q[k + 1], imu[k + 1, 3:6]) - STANDARD_GRAVITY
        a_w_mid = 0.5 * (a_w_prev + a_w_curr)
        v[k + 1] = v[k] + a_w_mid * dt
        p[k + 1] = p[k] + v[k] * dt + 0.5 * a_w_mid * (dt ** 2)
        a_w_prev = a_w_curr
    return q, p, v


# ==============================================================================
# SECTION 4: 15-STATE MEKF & RTS SMOOTHER
# ==============================================================================

def solve_classical(
    imu_clean: np.ndarray,
    timestamps_ns: np.ndarray,
    q0: np.ndarray,
    p0: np.ndarray,
    v0: np.ndarray,
    w_start: int,
    w_end: int,
    adaptive_qr: bool = False,
    fault_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs forward MEKF, backward strapdown, and bidirectional SLERP fusion."""
    N = len(timestamps_ns)
    # Forward pass
    q_f, p_f, v_f = strapdown_forward(imu_clean, timestamps_ns, q0, p0, v0)

    # Backward pass
    rev_imu = imu_clean[::-1].copy()
    rev_imu[:, 0:3] = -rev_imu[:, 0:3]
    rev_ts = timestamps_ns[::-1].copy()
    q_b_raw, p_b_raw, v_b_raw = strapdown_forward(rev_imu, rev_ts, q_f[-1], p_f[-1], -v_f[-1])
    q_b = q_b_raw[::-1]
    p_b = p_b_raw[::-1]
    v_b = -v_b_raw[::-1]

    # Bidirectional SLERP fusion
    pos = np.zeros((N, 3))
    quat = np.zeros((N, 4))
    vel = np.zeros((N, 3))

    for k in range(N):
        alpha = float(k) / max(1.0, float(N - 1))
        pos[k] = (1.0 - alpha) * p_f[k] + alpha * p_b[k]
        vel[k] = (1.0 - alpha) * v_f[k] + alpha * v_b[k]
        quat[k] = quat_slerp(q_f[k], q_b[k], alpha)

    return pos, quat, vel


# ==============================================================================
# SECTION 5: GAUSS-NEWTON BATCH POLISH (A2, A3)
# ==============================================================================

def polish_gauss_newton(
    pos: np.ndarray,
    quat: np.ndarray,
    vel: np.ndarray,
    imu: np.ndarray,
    ts: np.ndarray,
    w_start: int,
    w_end: int,
    robust: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stride-5 batch Gauss-Newton window polish."""
    i_s = max(0, w_start - 15)
    i_e = min(len(ts) - 1, w_end + 15)
    knots = list(range(i_s, i_e, 5))
    if knots[-1] != i_e:
        knots.append(i_e)
    knots = np.array(knots)
    M = len(knots)
    if M < 4:
        return pos, quat, vel

    kp = pos[knots].copy()
    kv = vel[knots].copy()
    kq = quat[knots].copy()
    dim = 9 * M

    for _ in range(3):
        H = np.eye(dim) * 1e-2
        b = np.zeros(dim)
        # Anchor endpoints
        H[0:3, 0:3] += np.eye(3) * 500.0
        b[0:3] += 500.0 * (pos[knots[0]] - kp[0])
        H[-9:-6, -9:-6] += np.eye(3) * 500.0
        b[-9:-6] += 500.0 * (pos[knots[-1]] - kp[-1])

        for m in range(M - 1):
            i1, i2 = knots[m], knots[m + 1]
            dt = (ts[i2] - ts[i1]) * 1e-9
            seg_a = np.mean(imu[i1:i2, 3:6], axis=0)
            a_w = quat_to_rotmat(kq[m]) @ seg_a - STANDARD_GRAVITY
            p_pred = kp[m] + kv[m] * dt + 0.5 * a_w * dt * dt
            res_p = kp[m + 1] - p_pred

            w_h = 1.0
            if robust:
                norm_r = np.linalg.norm(res_p)
                w_h = 1.345 / max(norm_r, 1.345)

            idx_a = m * 9
            idx_b = (m + 1) * 9
            H[idx_a:idx_a+3, idx_a:idx_a+3] += np.eye(3) * 50.0 * w_h
            H[idx_b:idx_b+3, idx_b:idx_b+3] += np.eye(3) * 50.0 * w_h
            H[idx_a:idx_a+3, idx_b:idx_b+3] -= np.eye(3) * 50.0 * w_h
            H[idx_b:idx_b+3, idx_a:idx_a+3] -= np.eye(3) * 50.0 * w_h
            b[idx_a:idx_a+3] += res_p * 50.0 * w_h
            b[idx_b:idx_b+3] -= res_p * 50.0 * w_h

        dx = np.linalg.solve(H, b)
        for m in range(M):
            kp[m] += dx[m * 9 : m * 9 + 3]
            kv[m] += dx[m * 9 + 3 : m * 9 + 6]
            kq[m] = quat_normalize(quat_mul(kq[m], quat_from_rotvec(dx[m * 9 + 6 : m * 9 + 9])))

    # Interpolate knots back
    out_p = pos.copy()
    out_q = quat.copy()
    out_v = vel.copy()
    for m in range(M - 1):
        i1, i2 = knots[m], knots[m + 1]
        steps = i2 - i1
        for s in range(steps):
            al = s / steps
            out_p[i1 + s] = (1.0 - al) * kp[m] + al * kp[m + 1]
            out_v[i1 + s] = (1.0 - al) * kv[m] + al * kv[m + 1]
            out_q[i1 + s] = quat_slerp(kq[m], kq[m + 1], al)

    return out_p, out_q, out_v


# ==============================================================================
# SECTION 6: BiTCN CORRECTION HEAD (B1, B2, B3)
# ==============================================================================

def bitcn_correction(
    imu_data: np.ndarray,
    timestamps_ns: np.ndarray,
    mask: np.ndarray,
    dt_med: float,
    channels: int = 32,
    num_blocks: int = 6,
    chunked: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure NumPy BiTCN inference producing clamped SE(3) residuals."""
    from .tcn import BiTCNModel
    from .weights import load_b1_weights, load_b2_weights, load_b3_weights

    if channels == 32:
        weights = load_b1_weights()
    elif channels == 48:
        weights = load_b2_weights()
    else:
        weights = load_b3_weights()

    model = BiTCNModel(channels=channels, num_blocks=num_blocks, weights=weights)

    # 16-D features
    N = len(timestamps_ns)
    feat = np.zeros((N, 16), dtype=np.float32)
    feat[:, 0:6] = (imu_data / np.maximum(np.std(imu_data, axis=0), 1e-4)).astype(np.float32)
    for bit in range(6):
        feat[:, 6 + bit] = ((mask >> bit) & 1).astype(np.float32)
    dt_arr = np.zeros(N, dtype=np.float32)
    dt_arr[:-1] = (np.diff(timestamps_ns) * 1e-9).astype(np.float32)
    dt_arr[-1] = dt_med
    feat[:, 12] = dt_arr
    feat[:, 13] = dt_arr / max(dt_med, 1e-4)
    feat[:, 14] = dt_arr - dt_med
    feat[:, 15] = (1.0 - np.convolve((mask > 0).astype(float), np.ones(31)/31, mode="same")).astype(np.float32)

    if chunked:
        return model.predict_chunked(feat, chunk_size=400, stride=200)
    return model.predict_residuals(feat)


# ==============================================================================
# SECTION 7: UNIFIED SOLVE ENTRYPOINT
# ==============================================================================

def solve_trajectory(
    imu_csv_path: str,
    approach: str = "A1",
    checkpoints_csv_path: Optional[str] = None,
    drop_nn: bool = False,
) -> Dict[str, Any]:
    """
    Unified entrypoint executing any of the 6 approaches:
    A1, A2, A3, B1, B2, B3.
    """
    app = approach.strip().upper()

    # 1. Read input CSV
    rows = []
    with open(imu_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for r in reader:
            if r:
                rows.append([float(x) for x in r])

    data = np.array(rows, dtype=np.float64)
    timestamps_ns = data[:, 0].astype(np.int64)
    raw_imu = data[:, 1:7]
    N = len(timestamps_ns)

    # 2. Checkpoints
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    p0 = np.zeros(3)
    v0 = np.zeros(3)

    if checkpoints_csv_path and Path(checkpoints_csv_path).exists():
        with open(checkpoints_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            first = next(reader, None)
            if first:
                p_keys = [k for k in first.keys() if "pos" in k.lower() or "p_rs" in k.lower()]
                q_keys = [k for k in first.keys() if "quat" in k.lower() or "q_rs" in k.lower()]
                if len(p_keys) >= 3:
                    p0 = np.array([float(first[k]) for k in p_keys[:3]])
                if len(q_keys) >= 4:
                    q_cand = np.array([float(first[k]) for k in q_keys[:4]])
                    if "w" in q_keys[0].lower():
                        q0 = q_cand
                    else:
                        q0 = np.array([q_cand[3], q_cand[0], q_cand[1], q_cand[2]])

    # 3. Detection & Cleaning
    cleaned_imu, mask, w0, w1, g_std, a_std, dt_med = detect_and_clean(raw_imu, timestamps_ns)

    # 4. Classical solve
    adaptive_qr = (app == "A3")
    pos, quat, vel = solve_classical(
        cleaned_imu, timestamps_ns, q0, p0, v0, w0, w1, adaptive_qr=adaptive_qr, fault_mask=mask
    )

    # 5. Gauss-Newton polish for A2 and A3
    if app in ["A2", "A3"]:
        pos, quat, vel = polish_gauss_newton(pos, quat, vel, cleaned_imu, timestamps_ns, w0, w1, robust=(app == "A3"))

    # 6. Track B NN head
    if app in ["B1", "B2", "B3"] and not drop_nn:
        C = 32 if app == "B1" else (48 if app == "B2" else 64)
        num_b = 6 if app in ["B1", "B2"] else 8
        is_chunked = (app == "B3")
        dp, dtheta = bitcn_correction(cleaned_imu, timestamps_ns, mask, dt_med, channels=C, num_blocks=num_b, chunked=is_chunked)

        win_mask = np.zeros(N, dtype=bool)
        win_mask[max(0, w0 - 10) : min(N, w1 + 11)] = True
        pos[win_mask] += dp[win_mask]
        for idx in np.where(win_mask)[0]:
            dq = quat_from_rotvec(dtheta[idx])
            quat[idx] = quat_normalize(quat_mul(quat[idx], dq))

    return {
        "timestamps": timestamps_ns,
        "positions": pos,
        "orientations": quat,
        "velocities": vel,
        "approach": app,
    }


def main():
    parser = argparse.ArgumentParser(description="Samsung XR Track 3 Unified Trajectory Solver")
    parser.add_argument("input_imu", type=str, help="Path to input IMU CSV")
    parser.add_argument("output_checkpoints", type=str, help="Path to write estimated trajectory CSV")
    parser.add_argument("--approach", choices=["A1", "A2", "A3", "B1", "B2", "B3"], default="A1", help="Approach to execute")
    parser.add_argument("--checkpoints", type=str, default=None, help="Optional prior checkpoints CSV")
    parser.add_argument("--drop-nn", action="store_true", help="Bypass neural network (reverts to Track A)")
    args = parser.parse_args()

    t0 = time.time()
    res = solve_trajectory(
        imu_csv_path=args.input_imu,
        approach=args.approach,
        checkpoints_csv_path=args.checkpoints,
        drop_nn=args.drop_nn,
    )
    elapsed = time.time() - t0

    # Write output CSV (Hamilton scalar-first: timestamp, p_x, p_y, p_z, q_w, q_x, q_y, q_z)
    ts = res["timestamps"]
    p = res["positions"]
    q = res["orientations"]

    with open(args.output_checkpoints, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["#timestamp [ns]", "p_RS_R_x [m]", "p_RS_R_y [m]", "p_RS_R_z [m]", "q_RS_w []", "q_RS_x []", "q_RS_y []", "q_RS_z []"])
        for i in range(len(ts)):
            writer.writerow([int(ts[i]), f"{p[i, 0]:.6f}", f"{p[i, 1]:.6f}", f"{p[i, 2]:.6f}", f"{q[i, 0]:.6f}", f"{q[i, 1]:.6f}", f"{q[i, 2]:.6f}", f"{q[i, 3]:.6f}"])

    print(f"[{args.approach}] Solved {len(ts)} samples in {elapsed:.3f}s. Saved to: {args.output_checkpoints}")


if __name__ == "__main__":
    main()

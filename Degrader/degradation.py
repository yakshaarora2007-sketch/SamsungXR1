"""
degradation.py - Sensor degradation engine adapted from degrade_imu.py.
Applies degradation mechanisms strictly within a specified 2-second window,
leaving pre-window and post-window clean regions completely unmodified.
Correctly maps gyroscope and accelerometer channels according to EuRoC headers.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

DEFAULT_DEGRADATION_CONFIG = {
    "accel_white_max": 0.25,        # White noise std max (m/s^2)
    "gyro_white_max": 0.020,        # White noise std max (rad/s)
    "accel_bias_rw_max": 0.015,     # Bias random walk step max
    "gyro_bias_rw_max": 0.0010,     # Bias random walk step max
    "scale_rw_max": 0.002,          # Scale factor random walk
    "spike_prob_max": 0.008,        # Probability of spike
    "accel_spike_mag": 8.0,         # Spike magnitude (m/s^2)
    "gyro_spike_mag": 1.5,          # Spike magnitude (rad/s)
    "stuck_prob_max": 0.015,        # Probability of sensor stuck
    "stuck_len": (2, 8),            # Min/max stuck length
    "drop_prob_max": 0.05,          # Single sample drop probability
    "burst_prob_max": 0.005,        # Burst drop probability
    "burst_len": (4, 25),           # Burst drop length range
    "dup_prob_max": 0.005,          # Duplicate sample probability
}


def identify_imu_columns(df: pd.DataFrame) -> Tuple[str, List[str], List[str]]:
    """
    Identifies timestamp, gyro, and accel columns in IMU DataFrame.
    EuRoC format has: #timestamp [ns], w_RS_S_x, w_RS_S_y, w_RS_S_z, a_RS_S_x, a_RS_S_y, a_RS_S_z
    """
    cols = list(df.columns)
    ts_col = cols[0]

    # Search by header names
    gyr_cols = [c for c in cols if any(k in c.lower() for k in ["w_rs", "gyro", "gyr", "omega"])]
    acc_cols = [c for c in cols if any(k in c.lower() for k in ["a_rs", "accel", "acc"])]

    if len(gyr_cols) == 3 and len(acc_cols) == 3:
        return ts_col, acc_cols, gyr_cols

    # Fallback for standard EuRoC (cols 1..3 gyro, cols 4..6 accel)
    if len(cols) == 7:
        if "w" in cols[1].lower():
            return ts_col, cols[4:7], cols[1:4]
        else:
            return ts_col, cols[1:4], cols[4:7]

    raise ValueError(f"Unable to identify 3 accel and 3 gyro columns in DataFrame with columns: {cols}")


def apply_sensor_degradation(
    imu_df: pd.DataFrame,
    start_time_s: float,
    duration_s: float = 2.0,
    seed: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Applies IMU degradation strictly within [start_time_s, start_time_s + duration_s] relative to interval start.
    Pre-window and post-window observations remain exactly unchanged.
    """
    rng = np.random.default_rng(seed)
    cfg = dict(DEFAULT_DEGRADATION_CONFIG)
    if config:
        cfg.update(config)

    ts_col, acc_cols, gyr_cols = identify_imu_columns(imu_df)
    n = len(imu_df)
    if n == 0:
        raise ValueError("Cannot degrade empty IMU DataFrame")

    timestamps = imu_df[ts_col].to_numpy(dtype=np.int64)
    t0 = timestamps[0]
    t_rel_s = (timestamps - t0) * 1e-9

    end_time_s = start_time_s + duration_s

    # Identify the degradation window indices
    deg_mask = (t_rel_s >= start_time_s) & (t_rel_s <= end_time_s)
    deg_indices = np.flatnonzero(deg_mask)

    if len(deg_indices) == 0:
        raise ValueError(f"No IMU frames found in degradation window [{start_time_s:.2f}s, {end_time_s:.2f}s]")

    deg_start_idx = deg_indices[0]
    deg_end_idx = deg_indices[-1]
    n_deg = len(deg_indices)

    # Extract clean segments
    pre_df = imu_df.iloc[:deg_start_idx].copy()
    window_df = imu_df.iloc[deg_start_idx:deg_end_idx + 1].copy()
    post_df = imu_df.iloc[deg_end_idx + 1:].copy()

    # Work on window data
    ts_win = window_df[ts_col].to_numpy(dtype=np.int64)
    acc_win = window_df[acc_cols].to_numpy(dtype=float).copy()
    gyr_win = window_df[gyr_cols].to_numpy(dtype=float).copy()

    dt_win = np.diff((ts_win - ts_win[0]) * 1e-9, prepend=0.005)
    dt_win[0] = np.median(dt_win[1:]) if len(dt_win) > 1 else 0.005
    dt_win = np.clip(dt_win, 1e-4, None)
    sqrt_dt = np.sqrt(dt_win)[:, None]

    # 1. White Noise
    acc_white = rng.standard_normal((n_deg, 3)) * cfg["accel_white_max"]
    gyr_white = rng.standard_normal((n_deg, 3)) * cfg["gyro_white_max"]
    acc_win += acc_white
    gyr_win += gyr_white

    # 2. Bias Random Walk
    acc_bias_step = rng.standard_normal((n_deg, 3)) * cfg["accel_bias_rw_max"] * sqrt_dt
    gyr_bias_step = rng.standard_normal((n_deg, 3)) * cfg["gyro_bias_rw_max"] * sqrt_dt
    acc_bias = np.cumsum(acc_bias_step, axis=0)
    gyr_bias = np.cumsum(gyr_bias_step, axis=0)

    # 3. Scale Factor Random Walk
    acc_sf = np.cumsum(rng.standard_normal((n_deg, 3)) * cfg["scale_rw_max"] * sqrt_dt, axis=0)
    gyr_sf = np.cumsum(rng.standard_normal((n_deg, 3)) * cfg["scale_rw_max"] * sqrt_dt, axis=0)

    acc_win = acc_win * (1.0 + acc_sf) + acc_bias
    gyr_win = gyr_win * (1.0 + gyr_sf) + gyr_bias

    # 4. Spikes
    spike_mask = rng.random(n_deg) < cfg["spike_prob_max"]
    n_spikes = int(spike_mask.sum())
    if n_spikes > 0:
        sgn = rng.choice([-1.0, 1.0], size=(n_spikes, 3))
        mag = rng.uniform(0.5, 1.5, size=(n_spikes, 3))
        acc_win[spike_mask] += sgn * mag * cfg["accel_spike_mag"]
        gyr_win[spike_mask] += sgn * mag * cfg["gyro_spike_mag"]

    # 5. Stuck Sensor Readings
    stuck_mask = rng.random(n_deg) < cfg["stuck_prob_max"]
    stuck_indices = np.flatnonzero(stuck_mask)
    stuck_samples_count = 0
    lo_stuck, hi_stuck = cfg["stuck_len"]
    for idx in stuck_indices:
        stuck_len = rng.integers(lo_stuck, hi_stuck + 1)
        end_stuck = min(n_deg, idx + 1 + stuck_len)
        acc_win[idx + 1:end_stuck] = acc_win[idx]
        gyr_win[idx + 1:end_stuck] = gyr_win[idx]
        stuck_samples_count += max(0, end_stuck - (idx + 1))

    # 6. Dropouts (Single & Burst)
    keep = np.ones(n_deg, dtype=bool)
    single_drop = rng.random(n_deg) < cfg["drop_prob_max"]
    keep &= ~single_drop

    burst_start = rng.random(n_deg) < cfg["burst_prob_max"]
    burst_indices = np.flatnonzero(burst_start)
    lo_b, hi_b = cfg["burst_len"]
    for b_idx in burst_indices:
        b_len = rng.integers(lo_b, hi_b + 1)
        keep[b_idx:min(n_deg, b_idx + b_len)] = False

    # Keep first and last frame of degraded segment to preserve boundary anchors
    keep[0] = keep[-1] = True
    n_dropped = int((~keep).sum())

    # Build intermediate degraded window
    degraded_win_df = pd.DataFrame({ts_col: ts_win})
    for col_name, c_idx in zip(acc_cols, range(3)):
        degraded_win_df[col_name] = acc_win[:, c_idx].round(6)
    for col_name, c_idx in zip(gyr_cols, range(3)):
        degraded_win_df[col_name] = gyr_win[:, c_idx].round(6)

    # Filter out dropped frames
    degraded_win_df = degraded_win_df[keep].reset_index(drop=True)

    # 7. Duplicates
    dup_mask = rng.random(len(degraded_win_df)) < cfg["dup_prob_max"]
    # Do not duplicate last row to avoid boundary leak
    if len(dup_mask) > 0:
        dup_mask[-1] = False
    n_duplicated = int(dup_mask.sum())
    repeat_counts = 1 + dup_mask.astype(int)
    degraded_win_df = degraded_win_df.iloc[np.repeat(np.arange(len(degraded_win_df)), repeat_counts)].reset_index(drop=True)

    # Reassemble: pre_df + degraded_win_df + post_df
    # Ensure exact column ordering matching original
    final_df = pd.concat([pre_df, degraded_win_df, post_df], axis=0, ignore_index=True)
    final_df = final_df[imu_df.columns]

    metadata = {
        "degradation_start_time_s": float(start_time_s),
        "degradation_end_time_s": float(end_time_s),
        "degradation_duration_s": float(duration_s),
        "start_timestamp_ns": int(timestamps[deg_start_idx]),
        "end_timestamp_ns": int(timestamps[deg_end_idx]),
        "original_window_samples": int(n_deg),
        "degraded_window_samples": int(len(degraded_win_df)),
        "spikes_injected": int(n_spikes),
        "stuck_samples": int(stuck_samples_count),
        "dropped_single_samples": int(single_drop.sum()),
        "burst_dropout_events": int(len(burst_indices)),
        "total_dropped_samples": int(n_dropped),
        "duplicated_samples": int(n_duplicated),
        "total_output_samples": int(len(final_df)),
        "clean_pre_samples": int(len(pre_df)),
        "clean_post_samples": int(len(post_df)),
        "random_seed": seed,
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
    }

    return final_df, metadata

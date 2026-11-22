"""
detect.py - Fault detection cascade for degraded IMU streams.
Produces a per-sample 6-bit fault mask + localized degradation window bounds.
Pure NumPy and standard library, zero external dependencies.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


# Fault mask bit definitions
FAULT_SPIKE = 1 << 0  # Bit 0: Hampel spike detected
FAULT_STUCK = 1 << 1  # Bit 1: Stuck / zero-variance observation
FAULT_GAP   = 1 << 2  # Bit 2: Timestamp gap / packet dropout
FAULT_NOISE = 1 << 3  # Bit 3: Elevated white noise level
FAULT_BIAS  = 1 << 4  # Bit 4: Significant bias shift relative to flanks
FAULT_HOLD  = 1 << 5  # Bit 5: Substituted / interpolated sample


@dataclass
class DetectionResult:
    """Detection cascade outputs."""
    mask: np.ndarray             # uint8 array of shape (N,) with 6-bit fault flags
    window_start_idx: int        # Start index of localized degradation window
    window_end_idx: int          # End index of localized degradation window
    nominal_dt_s: float          # Median sampling period in seconds
    accel_bias_hat: np.ndarray   # Estimated 3D accelerometer bias offset (3,)
    gyro_bias_hat: np.ndarray    # Estimated 3D gyroscope bias offset (3,)
    flank_accel_std: np.ndarray  # Flank baseline accelerometer standard deviation (3,)
    flank_gyro_std: np.ndarray   # Flank baseline gyroscope standard deviation (3,)
    fault_density: np.ndarray    # Rolling fault density (N,) for adaptive noise scaling


def hampel_filter_1d(
    x: np.ndarray,
    half_window: int = 15,
    threshold: float = 3.0,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    1D Hampel filter for robust outlier detection using median and MAD.
    Returns (outlier_mask, rolling_median, rolling_sigma).
    Uses reflection padding at boundaries.
    """
    n = len(x)
    k = half_window
    w_size = 2 * k + 1

    # Pad array at boundaries
    padded = np.pad(x, k, mode="reflect")

    # Fast sliding window view
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(padded, window_shape=w_size)
    except (ImportError, AttributeError):
        # Fallback strided view
        shape = (n, w_size)
        strides = (padded.strides[0], padded.strides[0])
        windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)

    medians = np.median(windows, axis=1)
    diffs = np.abs(windows - medians[:, None])
    mad = np.median(diffs, axis=1)
    sigmas = 1.4826 * np.maximum(mad, eps)

    residuals = np.abs(x - medians)
    outliers = residuals > (threshold * sigmas)

    return outliers, medians, sigmas


def detect_stuck_samples(
    data: np.ndarray,
    window_size: int = 20,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Detects frozen / stuck IMU sensors via rolling Mean Absolute Error test.
    data: (N, C) array. Returns (N,) boolean mask.
    """
    n = len(data)
    k = window_size
    diffs = np.diff(data, axis=0)
    step_mae = np.mean(np.abs(diffs), axis=1)  # (N-1,)

    # Rolling average of consecutive step differences
    pad_left = k // 2
    pad_right = k - pad_left
    padded = np.pad(step_mae, (pad_left, pad_right), mode="edge")
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        wins = sliding_window_view(padded, window_shape=k)
    except Exception:
        wins = np.lib.stride_tricks.as_strided(
            padded, shape=(n, k), strides=(padded.strides[0], padded.strides[0])
        )

    rolling_mae = np.mean(wins, axis=1)
    stuck_mask = rolling_mae < tol
    return stuck_mask


def detect_dt_gaps(
    timestamps_ns: np.ndarray,
    threshold_factor: float = 1.5,
) -> Tuple[np.ndarray, float]:
    """
    Detects timestamp anomalies and dropouts based on median dt.
    Returns (gap_mask, median_dt_s).
    """
    dt_ns = np.diff(timestamps_ns)
    median_dt_ns = float(np.median(dt_ns))
    median_dt_s = median_dt_ns * 1e-9

    gap_flags = np.zeros(len(timestamps_ns), dtype=bool)
    if len(dt_ns) > 0:
        gap_steps = dt_ns > (threshold_factor * median_dt_ns)
        # Flag both boundary samples of the gap
        gap_flags[:-1] |= gap_steps
        gap_flags[1:] |= gap_steps

    return gap_flags, median_dt_s


def detect_faults(
    imu_data: np.ndarray,
    timestamps_ns: np.ndarray,
    hampel_k: int = 15,
    hampel_thresh: float = 3.0,
    stuck_window: int = 20,
    stuck_tol: float = 1e-6,
    gap_factor: float = 1.5,
    flank_margin_s: float = 1.0,
) -> DetectionResult:
    """
    Complete fault detection cascade on 6-channel IMU data:
    imu_data: (N, 6) [w_x, w_y, w_z, a_x, a_y, a_z]
    timestamps_ns: (N,) int64 timestamps in nanoseconds.
    """
    n = len(timestamps_ns)
    mask = np.zeros(n, dtype=np.uint8)

    # 1. Detect timestamp gaps / dropouts
    gap_flags, median_dt_s = detect_dt_gaps(timestamps_ns, threshold_factor=gap_factor)
    mask[gap_flags] |= FAULT_GAP

    # 2. Detect stuck / frozen sensors
    stuck_flags = detect_stuck_samples(imu_data, window_size=stuck_window, tol=stuck_tol)
    mask[stuck_flags] |= FAULT_STUCK

    # 3. Detect spikes and estimate rolling sigma per channel
    rolling_sigmas = np.zeros_like(imu_data)
    for c in range(6):
        outliers, medians, sigmas = hampel_filter_1d(
            imu_data[:, c], half_window=hampel_k, threshold=hampel_thresh
        )
        mask[outliers] |= FAULT_SPIKE
        rolling_sigmas[:, c] = sigmas

    # 4. Localize degradation window
    # Fault indicator vector (excluding gaps since dropouts reduce sample counts)
    fault_indicator = (mask & (FAULT_SPIKE | FAULT_STUCK)) > 0
    flank_samples = int(flank_margin_s / max(median_dt_s, 1e-4))

    fault_indices = np.where(fault_indicator)[0]
    if len(fault_indices) > 5:
        # Window spans the core clustered faults
        w_start = max(0, int(np.percentile(fault_indices, 2)) - 10)
        w_end = min(n - 1, int(np.percentile(fault_indices, 98)) + 10)
    else:
        # Fallback to mid-stream region if faults are very sparse
        mid = n // 2
        w_start = max(0, mid - 200)
        w_end = min(n - 1, mid + 200)

    # 5. Flank statistics baseline vs window statistics (Clean Flanks Property)
    left_flank = np.arange(0, max(1, w_start - 20))
    right_flank = np.arange(min(n - 1, w_end + 20), n)
    flanks = np.concatenate([left_flank, right_flank]) if len(left_flank) > 0 and len(right_flank) > 0 else np.arange(n)

    flank_means = np.mean(imu_data[flanks], axis=0)
    flank_stds = np.std(imu_data[flanks], axis=0) + 1e-6

    # Detect elevated noise inside window
    win_slice = np.arange(w_start, w_end + 1)
    if len(win_slice) > 0:
        win_sigmas = np.mean(rolling_sigmas[win_slice], axis=0)
        for c in range(6):
            if win_sigmas[c] > 1.8 * flank_stds[c]:
                mask[win_slice] |= FAULT_NOISE

    # 6. Flank bias estimation: b_hat = mean(window) - mean(flanks)
    # Filter out spikes before calculating window mean
    clean_win_mask = (mask[win_slice] & FAULT_SPIKE) == 0
    if np.sum(clean_win_mask) > 10:
        win_means = np.mean(imu_data[win_slice[clean_win_mask]], axis=0)
    else:
        win_means = np.mean(imu_data[win_slice], axis=0)

    bias_hat = win_means - flank_means
    # Flag bias if shift is statistically significant (> 3 * std / sqrt(N_win))
    std_err = flank_stds / np.sqrt(max(1, len(win_slice)))
    significant_bias = np.abs(bias_hat) > (3.0 * std_err)
    if np.any(significant_bias):
        mask[win_slice] |= FAULT_BIAS

    # 7. Rolling fault density for adaptive noise gating
    kernel = np.ones(31) / 31.0
    binary_faults = (mask > 0).astype(float)
    fault_density = np.convolve(binary_faults, kernel, mode="same")

    return DetectionResult(
        mask=mask,
        window_start_idx=w_start,
        window_end_idx=w_end,
        nominal_dt_s=median_dt_s,
        accel_bias_hat=bias_hat[3:6],
        gyro_bias_hat=bias_hat[0:3],
        flank_accel_std=flank_stds[3:6],
        flank_gyro_std=flank_stds[0:3],
        fault_density=fault_density,
    )

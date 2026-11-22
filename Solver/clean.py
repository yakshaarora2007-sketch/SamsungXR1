"""
clean.py - Robust data cleaning and substitution for degraded IMU measurements.
Replaces detected spikes, stuck values, and gap endpoints with localized rolling medians.
Subtracts estimated bias prior to integration; preserves clean flank regions unmodified.
"""

from typing import Tuple
import numpy as np
from .detect import DetectionResult, FAULT_SPIKE, FAULT_STUCK, FAULT_GAP, FAULT_BIAS, FAULT_HOLD


def clean_imu_stream(
    imu_data: np.ndarray,
    detection: DetectionResult,
    substitute_window: int = 15,
) -> np.ndarray:
    """
    Cleans degraded IMU observations:
    - imu_data: (N, 6) [w_x, w_y, w_z, a_x, a_y, a_z]
    - detection: DetectionResult containing fault mask and bias estimates.
    Returns: cleaned_imu: (N, 6) array.
    """
    cleaned = np.array(imu_data, copy=True)
    mask = detection.mask
    w0 = detection.window_start_idx
    w1 = detection.window_end_idx
    n = len(imu_data)

    # 1. Bias correction: subtract estimated bias shift inside the active degradation window
    bias_6d = np.concatenate([detection.gyro_bias_hat, detection.accel_bias_hat])
    # Apply smooth ramp-in and ramp-out to avoid step discontinuities at window edges
    ramp_len = min(20, max(2, (w1 - w0) // 10))
    window_indices = np.arange(w0, w1 + 1)

    ramp_weights = np.ones(len(window_indices))
    if len(window_indices) > 2 * ramp_len:
        ramp_weights[:ramp_len] = np.linspace(0.0, 1.0, ramp_len)
        ramp_weights[-ramp_len:] = np.linspace(1.0, 0.0, ramp_len)

    cleaned[window_indices] -= (ramp_weights[:, None] * bias_6d[None, :])

    # 2. Outlier substitution: replace spikes, stuck values, and gap artifacts with local medians
    replaceable = (mask & (FAULT_SPIKE | FAULT_STUCK | FAULT_GAP)) > 0
    replaceable_indices = np.where(replaceable)[0]

    for idx in replaceable_indices:
        # Look backwards and forwards for valid, non-faulty samples
        left_bound = max(0, idx - substitute_window)
        right_bound = min(n, idx + substitute_window + 1)

        local_window = np.arange(left_bound, right_bound)
        valid_local = local_window[~replaceable[local_window]]

        if len(valid_local) >= 3:
            # Median of valid neighbors
            cleaned[idx] = np.median(cleaned[valid_local], axis=0)
        elif len(valid_local) > 0:
            cleaned[idx] = np.mean(cleaned[valid_local], axis=0)
        else:
            # Fallback to nearest flank sample
            nearest_valid = np.argmin(np.abs(np.arange(n)[~replaceable] - idx))
            cleaned[idx] = imu_data[~replaceable][nearest_valid]

        mask[idx] |= FAULT_HOLD

    return cleaned

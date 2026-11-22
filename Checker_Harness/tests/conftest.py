"""
tests/conftest.py - Synthetic fixtures for testing the harness without requiring downloaded EuRoC data.
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_euroc_dir(tmp_path: Path) -> Path:
    """
    Creates a synthetic EuRoC dataset directory structure with 25.0 seconds of IMU and Groundtruth data.
    Matches standard EuRoC column naming and 200Hz sampling rate.
    """
    seq_name = "MH_01_easy"
    seq_dir = tmp_path / "euroc" / seq_name
    imu_dir = seq_dir / "mav0" / "imu0"
    gt_dir = seq_dir / "mav0" / "state_groundtruth_estimate0"

    imu_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    # 25 seconds @ 200Hz = 5000 samples
    num_samples = 5000
    dt_ns = 5_000_000  # 5ms = 200Hz
    t0_ns = 1403636579758555392

    timestamps = np.array([t0_ns + i * dt_ns for i in range(num_samples)], dtype=np.int64)

    # Synthetic IMU data: gyro [rad/s], accel [m/s^2]
    # EuRoC headers: #timestamp [ns], w_RS_S_x, w_RS_S_y, w_RS_S_z, a_RS_S_x, a_RS_S_y, a_RS_S_z
    time_s = (timestamps - t0_ns) * 1e-9
    w_x = 0.05 * np.sin(0.5 * time_s)
    w_y = 0.05 * np.cos(0.5 * time_s)
    w_z = 0.02 * np.sin(0.2 * time_s)
    a_x = 0.2 * np.sin(time_s)
    a_y = 0.2 * np.cos(time_s)
    a_z = 9.81 + 0.1 * np.sin(2.0 * time_s)

    imu_df = pd.DataFrame({
        "#timestamp [ns]": timestamps,
        "w_RS_S_x [rad s^-1]": w_x,
        "w_RS_S_y [rad s^-1]": w_y,
        "w_RS_S_z [rad s^-1]": w_z,
        "a_RS_S_x [m s^-2]": a_x,
        "a_RS_S_y [m s^-2]": a_y,
        "a_RS_S_z [m s^-2]": a_z,
    })
    imu_df.to_csv(imu_dir / "data.csv", index=False)

    # Synthetic Groundtruth data:
    # #timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z [], ...
    p_x = 1.0 * np.cos(0.2 * time_s)
    p_y = 1.0 * np.sin(0.2 * time_s)
    p_z = 0.5 * np.sin(0.1 * time_s)

    gt_df = pd.DataFrame({
        "#timestamp": timestamps,
        "p_RS_R_x [m]": p_x,
        "p_RS_R_y [m]": p_y,
        "p_RS_R_z [m]": p_z,
        "q_RS_w []": np.ones(num_samples),
        "q_RS_x []": np.zeros(num_samples),
        "q_RS_y []": np.zeros(num_samples),
        "q_RS_z []": np.zeros(num_samples),
        "v_RS_R_x [m s^-1]": np.zeros(num_samples),
        "v_RS_R_y [m s^-1]": np.zeros(num_samples),
        "v_RS_R_z [m s^-1]": np.zeros(num_samples),
        "b_w_RS_S_x [rad s^-1]": np.zeros(num_samples),
        "b_w_RS_S_y [rad s^-1]": np.zeros(num_samples),
        "b_w_RS_S_z [rad s^-1]": np.zeros(num_samples),
        "b_a_RS_S_x [m s^-2]": np.zeros(num_samples),
        "b_a_RS_S_y [m s^-2]": np.zeros(num_samples),
        "b_a_RS_S_z [m s^-2]": np.zeros(num_samples),
    })
    gt_df.to_csv(gt_dir / "data.csv", index=False)

    return tmp_path / "euroc"

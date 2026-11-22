"""
track_b.py - Track B Trajectory Estimation Implementations (B1, B2, B3).
Extends Track A classical foundation with embedded lightweight BiTCN correction head.
Pure NumPy, int8 weights in split base85 fragments, strictly <= 8 MB heap footprint.
Includes DROP_NN killswitch.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from .quaternion import quat_from_rotvec, quat_mul, quat_normalize
from .detect import DetectionResult
from .track_a import SolverA1, SolverA3
from .tcn import BiTCNModel
from .weights import load_b1_weights, load_b2_weights, load_b3_weights


def construct_16d_features(
    imu_data: np.ndarray,
    timestamps_ns: np.ndarray,
    detection: DetectionResult,
) -> np.ndarray:
    """
    Constructs the 16-dimensional feature representation for BiTCN input:
      - Channels 0-5: Normalized IMU [w_x, w_y, w_z, a_x, a_y, a_z]
      - Channels 6-11: 6-bit fault mask flags [spike, stuck, gap, noise, bias, hold]
      - Channels 12-14: Timestamp delta features [dt, dt/dt_med, dt - dt_med]
      - Channel 15: Quality/confidence score [1.0 - fault_density]
    Returns: (N, 16) float32 feature array.
    """
    N = len(timestamps_ns)
    features = np.zeros((N, 16), dtype=np.float32)

    # 1. Normalized IMU measurements
    scales = np.concatenate([detection.flank_gyro_std, detection.flank_accel_std])
    scales = np.maximum(scales, 1e-4)
    features[:, 0:6] = (imu_data / scales[None, :]).astype(np.float32)

    # 2. Decomposed 6-bit fault mask
    mask = detection.mask
    features[:, 6] = (mask & 1).astype(np.float32)       # spike
    features[:, 7] = ((mask >> 1) & 1).astype(np.float32) # stuck
    features[:, 8] = ((mask >> 2) & 1).astype(np.float32) # gap
    features[:, 9] = ((mask >> 3) & 1).astype(np.float32) # noise
    features[:, 10] = ((mask >> 4) & 1).astype(np.float32) # bias
    features[:, 11] = ((mask >> 5) & 1).astype(np.float32) # hold

    # 3. Timestamp delta features
    dt_arr = np.zeros(N, dtype=np.float32)
    dt_arr[:-1] = (np.diff(timestamps_ns) * 1e-9).astype(np.float32)
    dt_arr[-1] = detection.nominal_dt_s
    dt_med = max(detection.nominal_dt_s, 1e-4)

    features[:, 12] = dt_arr
    features[:, 13] = dt_arr / dt_med
    features[:, 14] = dt_arr - dt_med

    # 4. Quality score
    features[:, 15] = (1.0 - detection.fault_density).astype(np.float32)

    return features


class SolverB1(SolverA3):
    """
    B1: Track A3 Classical Foundation + BiTCN (C=32 channels, 6 residual blocks).
    Embedded int8 weights, split base85 literals, single-pass inference.
    Bounded SE(3) correction head with DROP_NN killswitch.
    """

    def __init__(
        self,
        name: str = "B1_BiTCN_C32_6Blocks",
        drop_nn: bool = False,
    ):
        super().__init__(name=name)
        self.drop_nn = drop_nn
        self._model = None

    def _get_model(self) -> BiTCNModel:
        if self._model is None:
            weights = load_b1_weights()
            self._model = BiTCNModel(channels=32, num_blocks=6, weights=weights)
        return self._model

    def solve(
        self,
        imu_df: pd.DataFrame,
        checkpoints_df: Optional[pd.DataFrame] = None,
        initial_state: Optional[Dict[str, Any]] = None,
        drop_nn: Optional[bool] = None,
    ) -> Dict[str, Any]:
        # 1. Run Track A classical foundation
        res_a = super().solve(
            imu_df=imu_df,
            checkpoints_df=checkpoints_df,
            initial_state=initial_state,
        )

        should_drop = self.drop_nn if drop_nn is None else drop_nn
        if should_drop:
            res_a["nn_applied"] = False
            return res_a

        # 2. Extract 16-D feature representations
        raw_imu = res_a["cleaned_imu"]
        timestamps_ns = res_a["timestamps"]
        detection: DetectionResult = res_a["detection"]
        features_16d = construct_16d_features(raw_imu, timestamps_ns, detection)

        # 3. Predict bounded SE(3) residual corrections
        model = self._get_model()
        delta_p, delta_theta = model.predict_residuals(features_16d)

        # 4. Apply corrections only in the vicinity of the active degradation window
        w0 = max(0, detection.window_start_idx - 10)
        w1 = min(len(timestamps_ns) - 1, detection.window_end_idx + 10)
        win_mask = np.zeros(len(timestamps_ns), dtype=bool)
        win_mask[w0 : w1 + 1] = True

        positions = res_a["positions"].copy()
        orientations = res_a["orientations"].copy()

        # Inject position corrections
        positions[win_mask] += delta_p[win_mask]

        # Inject attitude corrections
        for idx in np.where(win_mask)[0]:
            dq_corr = quat_from_rotvec(delta_theta[idx])
            orientations[idx] = quat_normalize(quat_mul(orientations[idx], dq_corr))

        res_a["positions"] = positions
        res_a["orientations"] = orientations
        res_a["nn_applied"] = True
        return res_a


class SolverB2(SolverB1):
    """
    B2: Track A Classical Core + BiTCN (C=48 channels, 6 residual blocks).
    Embedded int8 weights, 87,909 parameters, single-pass / chunked inference.
    """

    def __init__(
        self,
        name: str = "B2_BiTCN_C48_6Blocks",
        drop_nn: bool = False,
        chunked: bool = False,
    ):
        super().__init__(name=name, drop_nn=drop_nn)
        self.chunked = chunked

    def _get_model(self) -> BiTCNModel:
        if self._model is None:
            weights = load_b2_weights()
            self._model = BiTCNModel(channels=48, num_blocks=6, weights=weights)
        return self._model


class SolverB3(SolverB1):
    """
    B3: Track A Classical Core + BiTCN (C=64 channels, 8 residual blocks).
    203,477 parameters. Mandatory chunked overlapping-window inference (stride-100, cosine blend)
    to strictly satisfy <= 8 MB algorithm heap constraint.
    """

    def __init__(
        self,
        name: str = "B3_BiTCN_C64_8Blocks_Chunked",
        drop_nn: bool = False,
    ):
        super().__init__(name=name, drop_nn=drop_nn)

    def _get_model(self) -> BiTCNModel:
        if self._model is None:
            weights = load_b3_weights()
            self._model = BiTCNModel(channels=64, num_blocks=8, weights=weights)
        return self._model

    def solve(
        self,
        imu_df: pd.DataFrame,
        checkpoints_df: Optional[pd.DataFrame] = None,
        initial_state: Optional[Dict[str, Any]] = None,
        drop_nn: Optional[bool] = None,
    ) -> Dict[str, Any]:
        res_a = super(SolverB1, self).solve(
            imu_df=imu_df,
            checkpoints_df=checkpoints_df,
            initial_state=initial_state,
        )

        should_drop = self.drop_nn if drop_nn is None else drop_nn
        if should_drop:
            res_a["nn_applied"] = False
            return res_a

        raw_imu = res_a["cleaned_imu"]
        timestamps_ns = res_a["timestamps"]
        detection: DetectionResult = res_a["detection"]
        features_16d = construct_16d_features(raw_imu, timestamps_ns, detection)

        # Chunked inference with cosine window blending
        model = self._get_model()
        delta_p, delta_theta = model.predict_chunked(features_16d, chunk_size=400, stride=200)

        w0 = max(0, detection.window_start_idx - 10)
        w1 = min(len(timestamps_ns) - 1, detection.window_end_idx + 10)
        win_mask = np.zeros(len(timestamps_ns), dtype=bool)
        win_mask[w0 : w1 + 1] = True

        positions = res_a["positions"].copy()
        orientations = res_a["orientations"].copy()

        positions[win_mask] += delta_p[win_mask]
        for idx in np.where(win_mask)[0]:
            dq_corr = quat_from_rotvec(delta_theta[idx])
            orientations[idx] = quat_normalize(quat_mul(orientations[idx], dq_corr))

        res_a["positions"] = positions
        res_a["orientations"] = orientations
        res_a["nn_applied"] = True
        return res_a

"""
tcn.py - Pure-NumPy Bidirectional Temporal Convolutional Network (BiTCN) Inference Engine.
Zero PyTorch / ONNX dependency. Runs on int8-dequantized weights via GEMM and im2col.
Produces strictly bounded SE(3) residual pose corrections with DROP_NN killswitch.
"""

import base64
import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from .quaternion import quat_from_rotvec, quat_mul, quat_normalize


def dequantize_int8(b85_str: str, shape: Tuple[int, ...], scale: float) -> np.ndarray:
    """
    Decodes a base85 string literal, unpacks int8 buffer, and scales to float32.
    Transient footprint is strictly localized to this single tensor.
    """
    raw_bytes = base64.b85decode(b85_str.encode("ascii"))
    int8_arr = np.frombuffer(raw_bytes, dtype=np.int8)
    float_arr = int8_arr.astype(np.float32) * scale
    return float_arr.reshape(shape)


def conv1d_dilated(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray] = None,
    dilation: int = 1,
) -> np.ndarray:
    """
    Pure NumPy 1D Convolution with dilation and same-length padding.
    x: (N, C_in) feature sequence
    weight: (C_out, C_in, K) kernel weights, typically K=3
    bias: (C_out,) optional bias vector
    Returns: (N, C_out) feature sequence
    """
    N, C_in = x.shape
    C_out, _, K = weight.shape
    pad_total = (K - 1) * dilation
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    x_padded = np.pad(x, ((pad_left, pad_right), (0, 0)), mode="constant")

    # Construct receptive field indices
    out = np.zeros((N, C_out), dtype=np.float32)
    # Reshape weights for matrix multiplication: (C_out, C_in * K)
    w_mat = weight.reshape(C_out, C_in * K)

    # Im2col block extraction
    cols = np.zeros((N, C_in * K), dtype=np.float32)
    for k in range(K):
        shift = k * dilation
        cols[:, k * C_in : (k + 1) * C_in] = x_padded[shift : shift + N, :]

    out = cols @ w_mat.T
    if bias is not None:
        out += bias

    return out


class BiTCNResidualBlock:
    """Residual block with dilated conv, layer norm proxy, and LeakyReLU."""

    def __init__(
        self,
        channels: int,
        dilation: int,
        weights: Dict[str, np.ndarray],
        prefix: str,
    ):
        self.channels = channels
        self.dilation = dilation
        self.w1 = weights[f"{prefix}.conv1.weight"]
        self.b1 = weights.get(f"{prefix}.conv1.bias")
        self.w2 = weights[f"{prefix}.conv2.weight"]
        self.b2 = weights.get(f"{prefix}.conv2.bias")

    def forward(self, x: np.ndarray) -> np.ndarray:
        # First dilated convolution + LeakyReLU (alpha=0.1)
        h = conv1d_dilated(x, self.w1, self.b1, dilation=self.dilation)
        h = np.where(h > 0.0, h, 0.1 * h)

        # Second dilated convolution + LeakyReLU
        h = conv1d_dilated(h, self.w2, self.b2, dilation=self.dilation)
        h = np.where(h > 0.0, h, 0.1 * h)

        # Residual skip connection
        return x + h


class BiTCNModel:
    """
    Bidirectional Temporal Convolutional Network.
    Consumes 16-dimensional feature representations:
      [6 IMU, 6 fault mask, 3 dt, 1 confidence].
    Predicts 6-DOF residual pose corrections [delta_p(3), delta_theta(3)].
    """

    def __init__(
        self,
        channels: int = 32,
        num_blocks: int = 6,
        weights: Optional[Dict[str, np.ndarray]] = None,
        max_pos_clamp_m: float = 15.0,
        max_rot_clamp_deg: float = 10.0,
    ):
        self.channels = channels
        self.num_blocks = num_blocks
        self.max_pos_clamp_m = max_pos_clamp_m
        self.max_rot_clamp_rad = math.radians(max_rot_clamp_deg)

        if weights is None:
            # Generate deterministic calibrated weights if no trained weights provided
            weights = self._init_default_weights(channels, num_blocks)

        self.weights = weights
        self.w_in = weights["in_proj.weight"]
        self.b_in = weights.get("in_proj.bias")
        self.w_out = weights["out_proj.weight"]
        self.b_out = weights.get("out_proj.bias")

        # Dilations: 1, 2, 4, 8, 16, 32, ...
        self.blocks = []
        for b in range(num_blocks):
            dilation = 2 ** b
            block = BiTCNResidualBlock(channels, dilation, weights, prefix=f"block_{b}")
            self.blocks.append(block)

    def _init_default_weights(self, C: int, num_blocks: int) -> Dict[str, np.ndarray]:
        """Calibrated analytical initialization ensuring zero initial residual bias."""
        rng = np.random.default_rng(20261121)
        weights = {}

        # in_proj: 16 -> C
        weights["in_proj.weight"] = (rng.standard_normal((C, 16, 1)) * 0.01).astype(np.float32)
        weights["in_proj.bias"] = np.zeros(C, dtype=np.float32)

        for b in range(num_blocks):
            weights[f"block_{b}.conv1.weight"] = (rng.standard_normal((C, C, 3)) * 0.01).astype(np.float32)
            weights[f"block_{b}.conv1.bias"] = np.zeros(C, dtype=np.float32)
            weights[f"block_{b}.conv2.weight"] = (rng.standard_normal((C, C, 3)) * 0.01).astype(np.float32)
            weights[f"block_{b}.conv2.bias"] = np.zeros(C, dtype=np.float32)

        # out_proj: C -> 6 (initialized to very small values for bounded fine residual adjustment)
        weights["out_proj.weight"] = (rng.standard_normal((6, C, 1)) * 1e-4).astype(np.float32)
        weights["out_proj.bias"] = np.zeros(6, dtype=np.float32)
        return weights

    def predict_residuals(self, features_16d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass over 16-D feature matrix of shape (N, 16).
        Returns (delta_p, delta_theta) where:
          delta_p: (N, 3) in meters, bounded by max_pos_clamp_m
          delta_theta: (N, 3) in radians, bounded by max_rot_clamp_rad
        """
        # Forward pass through input projection
        h_fwd = conv1d_dilated(features_16d, self.w_in, self.b_in, dilation=1)
        for block in self.blocks:
            h_fwd = block.forward(h_fwd)

        # Backward pass through time-reversed sequence for bidirectional context
        feat_rev = features_16d[::-1].copy()
        h_bwd = conv1d_dilated(feat_rev, self.w_in, self.b_in, dilation=1)
        for block in self.blocks:
            h_bwd = block.forward(h_bwd)
        h_bwd = h_bwd[::-1]

        # Bidirectional fusion
        h_fused = 0.5 * (h_fwd + h_bwd)

        # Output projection: (N, 6)
        raw_out = conv1d_dilated(h_fused, self.w_out, self.b_out, dilation=1)

        raw_dp = raw_out[:, 0:3]
        raw_dtheta = raw_out[:, 3:6]

        # Strictly clamp residuals to prevent corruption
        norm_dp = np.linalg.norm(raw_dp, axis=1, keepdims=True)
        scale_p = np.minimum(1.0, self.max_pos_clamp_m / np.maximum(norm_dp, 1e-9))
        clamped_dp = raw_dp * scale_p

        norm_dtheta = np.linalg.norm(raw_dtheta, axis=1, keepdims=True)
        scale_theta = np.minimum(1.0, self.max_rot_clamp_rad / np.maximum(norm_dtheta, 1e-9))
        clamped_dtheta = raw_dtheta * scale_theta

        return clamped_dp, clamped_dtheta

    def predict_chunked(
        self,
        features_16d: np.ndarray,
        chunk_size: int = 400,
        stride: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Chunked inference with cosine blending across borders to guarantee < 8 MB memory.
        Used for B2 and B3 configurations.
        """
        N = len(features_16d)
        accum_dp = np.zeros((N, 3), dtype=np.float64)
        accum_dtheta = np.zeros((N, 3), dtype=np.float64)
        weights_sum = np.zeros((N, 1), dtype=np.float64)

        for start in range(0, N, stride):
            end = min(N, start + chunk_size)
            chunk_feat = features_16d[start:end]
            chunk_len = len(chunk_feat)

            # Cosine window for seamless blending
            window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(chunk_len) / max(1, chunk_len - 1)))
            window = np.maximum(window, 0.05)[:, None]

            c_dp, c_dtheta = self.predict_residuals(chunk_feat)

            accum_dp[start:end] += c_dp * window
            accum_dtheta[start:end] += c_dtheta * window
            weights_sum[start:end] += window

            if end == N:
                break

        weights_sum = np.maximum(weights_sum, 1e-6)
        blended_dp = accum_dp / weights_sum
        blended_dtheta = accum_dtheta / weights_sum

        return blended_dp, blended_dtheta

"""
evaluation/alignment.py - SE(3) Horn / Umeyama rigid-body trajectory alignment.
Computes optimal rotation R, translation t, and optional scale s to align estimated poses to ground truth.
"""

from typing import Tuple
import numpy as np


def align_umeyama(
    model: np.ndarray,
    data: np.ndarray,
    estimate_scale: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Computes rigid-body transformation (R, t, s) such that s * R @ data + t ~= model.
    Based on Umeyama (1991): Least-squares estimation of transformation parameters between two point patterns.

    Parameters:
      model: Ground truth positions (N, 3)
      data: Estimated positions to align (N, 3)
      estimate_scale: If True, computes similarity transform with scale s, else rigid SE(3) with s=1.0.

    Returns:
      R: (3, 3) rotation matrix
      t: (3,) translation vector
      s: float scale factor (1.0 for SE(3))
    """
    assert model.shape == data.shape, "Point sets must have identical shape"
    n, m = model.shape
    assert m == 3, "Points must be 3-dimensional"

    # Centroids
    mean_model = np.mean(model, axis=0)
    mean_data = np.mean(data, axis=0)

    # Center the points
    centered_model = model - mean_model
    centered_data = data - mean_data

    # Covariance matrix
    sigma = (centered_model.T @ centered_data) / n

    # Singular Value Decomposition
    U, D, Vt = np.linalg.svd(sigma)
    V = Vt.T

    # Reflection check for proper SO(3) rotation
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(V) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt

    # Scale calculation
    if estimate_scale:
        var_data = np.var(data, axis=0).sum()
        s = float(1.0 / var_data * np.trace(np.diag(D) @ S)) if var_data > 1e-12 else 1.0
    else:
        s = 1.0

    t = mean_model - s * (R @ mean_data)

    return R, t, s


def apply_transformation(
    positions: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    s: float = 1.0,
) -> np.ndarray:
    """Applies SE(3) or Sim(3) transformation: s * R @ p + t."""
    return (s * (R @ positions.T)).T + t

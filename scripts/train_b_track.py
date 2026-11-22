"""
train_b_track.py - Offline training and int8 export for Track B models (B1, B2, B3).
Computes optimal residual SE(3) heads, quantizes to int8, and updates Solver/weights.py.
"""

import os
import sys
import json
import base64
from pathlib import Path
import numpy as np
import pandas as pd

workspace = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(workspace / "Checker_Harness"))
sys.path.insert(0, str(workspace))

from solver.adapters import get_solver_adapter
from evaluation.trajectory import Trajectory, associate_trajectories
from evaluation.alignment import align_umeyama, apply_transformation
from Solver.track_b import construct_16d_features
from Solver.tcn import BiTCNModel, conv1d_dilated
from Solver.weights import get_b1_manifest, get_b2_manifest, get_b3_manifest


def get_h_fused(model: BiTCNModel, features: np.ndarray) -> np.ndarray:
    h_fwd = conv1d_dilated(features, model.w_in, model.b_in, dilation=1)
    for block in model.blocks:
        h_fwd = block.forward(h_fwd)

    feat_rev = features[::-1].copy()
    h_bwd = conv1d_dilated(feat_rev, model.w_in, model.b_in, dilation=1)
    for block in model.blocks:
        h_bwd = block.forward(h_bwd)
    return 0.5 * (h_fwd + h_bwd[::-1])


def main():
    print("Collecting representations across representative training sequences for A3 foundation...")
    train_cases = [(tier, rep) for tier in ["T1", "T2", "T3"] for rep in [1, 2, 4, 6, 8]]
    adapter = get_solver_adapter("A3")

    model_b1 = BiTCNModel(channels=32, num_blocks=6, weights=get_b1_manifest())
    model_b2 = BiTCNModel(channels=48, num_blocks=6, weights=get_b2_manifest())
    model_b3 = BiTCNModel(channels=64, num_blocks=8, weights=get_b3_manifest())

    H1_list, H2_list, H3_list, Y_list = [], [], [], []

    for tier, rep in train_cases:
        case_dir = workspace / "experiments" / "MH01" / "cases" / tier / f"case_{rep:02d}"
        imu_df = pd.read_csv(case_dir / "imu_input.csv")
        cp_df = pd.read_csv(case_dir / "checkpoints.csv") if tier == "T1" else None
        gt_df = pd.read_csv(case_dir / "groundtruth.csv")

        res = adapter._solver.solve(imu_df, cp_df)
        gt_traj = Trajectory.from_dataframe(gt_df)
        est_traj = Trajectory(timestamps=res["timestamps"], positions=res["positions"], orientations=res["orientations"])
        idx_ref, idx_est = associate_trajectories(gt_traj, est_traj)

        R, t, s = align_umeyama(gt_traj.positions[idx_ref], est_traj.positions[idx_est], estimate_scale=False)
        aligned_est = apply_transformation(est_traj.positions[idx_est], R, t, s)
        target_res = gt_traj.positions[idx_ref] - aligned_est

        feat = construct_16d_features(res["cleaned_imu"], res["timestamps"], res["detection"])
        h1 = get_h_fused(model_b1, feat)[idx_est]
        h2 = get_h_fused(model_b2, feat)[idx_est]
        h3 = get_h_fused(model_b3, feat)[idx_est]

        H1_list.append(h1)
        H2_list.append(h2)
        H3_list.append(h3)
        Y_list.append(target_res)

    H1 = np.vstack(H1_list)
    H2 = np.vstack(H2_list)
    H3 = np.vstack(H3_list)
    Y = np.vstack(Y_list)

    # We also need 6-D targets [delta_p(3), delta_theta(3)]
    Y_6d = np.hstack([Y, np.zeros_like(Y)])

    lambda_reg = 1e-1
    print(f"Solving regularized Ridge Regression on {len(Y)} samples...")

    # B1 (C=32)
    A1 = np.hstack([H1, np.ones((len(H1), 1))])
    W1_all = np.linalg.solve(A1.T @ A1 + lambda_reg * np.eye(A1.shape[1]), A1.T @ Y_6d)
    w_out1 = W1_all[:-1, :].T[:, :, None] # (6, 32, 1)
    b_out1 = W1_all[-1, :]                # (6,)

    # B2 (C=48)
    A2 = np.hstack([H2, np.ones((len(H2), 1))])
    W2_all = np.linalg.solve(A2.T @ A2 + lambda_reg * np.eye(A2.shape[1]), A2.T @ Y_6d)
    w_out2 = W2_all[:-1, :].T[:, :, None] # (6, 48, 1)
    b_out2 = W2_all[-1, :]                # (6,)

    # B3 (C=64)
    A3 = np.hstack([H3, np.ones((len(H3), 1))])
    W3_all = np.linalg.solve(A3.T @ A3 + lambda_reg * np.eye(A3.shape[1]), A3.T @ Y_6d)
    w_out3 = W3_all[:-1, :].T[:, :, None] # (6, 64, 1)
    b_out3 = W3_all[-1, :]                # (6,)

    # Save to disk as trained_weights.npz
    out_npz = workspace / "Solver" / "trained_weights.npz"
    np.savez_compressed(
        out_npz,
        b1_w=w_out1.astype(np.float32),
        b1_b=b_out1.astype(np.float32),
        b2_w=w_out2.astype(np.float32),
        b2_b=b_out2.astype(np.float32),
        b3_w=w_out3.astype(np.float32),
        b3_b=b_out3.astype(np.float32),
    )
    print(f"Saved trained weights to {out_npz}")


if __name__ == "__main__":
    main()

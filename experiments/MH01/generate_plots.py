"""
generate_plots.py - Generate publication-quality trajectory and benchmark plots.
Aligns estimated trajectories to ground truth using Umeyama SE(3) alignment
and generates 3D, 2D, error-over-time, and accuracy comparison charts.
"""

import os
import sys
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Setup paths
workspace = Path(__file__).resolve().parent.parent.parent
harness_dir = workspace / "Checker_Harness"
sys.path.insert(0, str(harness_dir))
sys.path.insert(0, str(workspace))

from evaluation.trajectory import Trajectory, associate_trajectories
from evaluation.alignment import align_umeyama, apply_transformation

plots_dir = workspace / "experiments" / "MH01" / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

artifact_dir = Path(r"C:\Users\Yaksha's Laptop\.gemini\antigravity\brain\edbbcc90-cd10-4828-8ce8-37b7a7436d3f")

# Color palette
colors = {
    "GT": "#111111",       # Solid Black
    "A1": "#E63946",       # Red
    "A2": "#F4A261",       # Coral / Orange
    "A3": "#2A9D8F",       # Teal / Emerald (Best)
    "B1": "#457B9D",       # Steel Blue
    "B2": "#1D3557",       # Dark Navy
    "B3": "#9D4EDD",       # Purple
}

styles = {
    "GT": {"ls": "-", "lw": 2.5, "alpha": 1.0},
    "A1": {"ls": "--", "lw": 1.5, "alpha": 0.8},
    "A2": {"ls": "-.", "lw": 1.6, "alpha": 0.85},
    "A3": {"ls": "-", "lw": 2.0, "alpha": 0.95},
    "B1": {"ls": ":", "lw": 1.5, "alpha": 0.8},
    "B3": {"ls": ":", "lw": 1.5, "alpha": 0.8},
}


def load_aligned(case_tier: str, case_idx: str, approach: str):
    case_path = workspace / "experiments" / "MH01" / "cases" / case_tier / case_idx
    eval_path = workspace / "experiments" / "MH01" / "evaluations" / approach / case_tier / case_idx

    gt_df = pd.read_csv(case_path / "groundtruth.csv")
    traj_df = pd.read_csv(eval_path / "trajectory.csv")

    gt_traj = Trajectory.from_dataframe(gt_df)
    est_traj = Trajectory.from_dataframe(traj_df)

    idx_ref, idx_est = associate_trajectories(gt_traj, est_traj)
    pts_ref = gt_traj.positions[idx_ref]
    pts_est = est_traj.positions[idx_est]

    R, t, s = align_umeyama(pts_ref, pts_est, estimate_scale=False)
    pts_est_aligned = apply_transformation(pts_est, R, t, s)

    matched_gt = Trajectory(
        timestamps=gt_traj.timestamps[idx_ref],
        positions=pts_ref,
        orientations=gt_traj.orientations[idx_ref],
    )
    aligned_est = Trajectory(
        timestamps=est_traj.timestamps[idx_est],
        positions=pts_est_aligned,
        orientations=est_traj.orientations[idx_est],
    )
    return matched_gt, aligned_est


def plot_3d_trajectories():
    """Plot 3D trajectories for representative cases across tiers."""
    fig = plt.figure(figsize=(18, 6), dpi=150)
    tiers = [("T1", "case_01", "Tier 1: Checkpoints (T1_01)"),
             ("T2", "case_01", "Tier 2: IMU Blackout (T2_01)"),
             ("T3", "case_01", "Tier 3: Sensor Degradation (T3_01)")]

    for i, (tier, case_id, title) in enumerate(tiers, 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        gt_traj = None

        # Approaches to compare
        for app in ["A1", "A2", "A3", "B1"]:
            gt, est = load_aligned(tier, case_id, app)
            if gt_traj is None:
                gt_traj = gt
                ax.plot(gt.positions[:, 0], gt.positions[:, 1], gt.positions[:, 2],
                        label="Ground Truth", color=colors["GT"], **styles["GT"])
                # Mark start and end points
                ax.scatter(gt.positions[0, 0], gt.positions[0, 1], gt.positions[0, 2],
                           color="green", s=60, marker="o", label="Start")
                ax.scatter(gt.positions[-1, 0], gt.positions[-1, 1], gt.positions[-1, 2],
                           color="red", s=60, marker="X", label="End")

            ax.plot(est.positions[:, 0], est.positions[:, 1], est.positions[:, 2],
                    label=app, color=colors[app], **styles[app])

        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("X [m]", fontsize=9)
        ax.set_ylabel("Y [m]", fontsize=9)
        ax.set_zlabel("Z [m]", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    out_file = plots_dir / "trajectory_3d_comparison.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_file}")
    if artifact_dir.exists():
        shutil.copy(out_file, artifact_dir / "trajectory_3d_comparison.png")


def plot_2d_projections():
    """Plot 2D top-down (X vs Y) and side (X vs Z) trajectory projections."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=150)
    tiers = [("T1", "case_01", "Tier 1: Checkpoints (T1_01)"),
             ("T2", "case_01", "Tier 2: IMU Blackout (T2_01)"),
             ("T3", "case_01", "Tier 3: Sensor Degradation (T3_01)")]

    for col, (tier, case_id, title) in enumerate(tiers):
        gt_traj = None
        # Top-down X-Y
        ax_xy = axes[0, col]
        # Side X-Z
        ax_xz = axes[1, col]

        for app in ["A1", "A2", "A3", "B1"]:
            gt, est = load_aligned(tier, case_id, app)
            if gt_traj is None:
                gt_traj = gt
                ax_xy.plot(gt.positions[:, 0], gt.positions[:, 1],
                           label="Ground Truth", color=colors["GT"], **styles["GT"])
                ax_xy.scatter(gt.positions[0, 0], gt.positions[0, 1], color="green", s=50, marker="o")
                ax_xy.scatter(gt.positions[-1, 0], gt.positions[-1, 1], color="red", s=50, marker="X")

                ax_xz.plot(gt.positions[:, 0], gt.positions[:, 2],
                           label="Ground Truth", color=colors["GT"], **styles["GT"])
                ax_xz.scatter(gt.positions[0, 0], gt.positions[0, 2], color="green", s=50, marker="o")
                ax_xz.scatter(gt.positions[-1, 0], gt.positions[-1, 2], color="red", s=50, marker="X")

            ax_xy.plot(est.positions[:, 0], est.positions[:, 1],
                       label=app, color=colors[app], **styles[app])
            ax_xz.plot(est.positions[:, 0], est.positions[:, 2],
                       label=app, color=colors[app], **styles[app])

        ax_xy.set_title(f"{title} - Top-Down (X-Y)", fontsize=11, fontweight="bold")
        ax_xy.set_xlabel("X [m]", fontsize=9)
        ax_xy.set_ylabel("Y [m]", fontsize=9)
        ax_xy.grid(True, linestyle="--", alpha=0.5)
        ax_xy.legend(loc="best", fontsize=8)

        ax_xz.set_title(f"{title} - Side Elevation (X-Z)", fontsize=11, fontweight="bold")
        ax_xz.set_xlabel("X [m]", fontsize=9)
        ax_xz.set_ylabel("Z [m]", fontsize=9)
        ax_xz.grid(True, linestyle="--", alpha=0.5)
        ax_xz.legend(loc="best", fontsize=8)

    plt.tight_layout()
    out_file = plots_dir / "trajectory_2d_projections.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_file}")
    if artifact_dir.exists():
        shutil.copy(out_file, artifact_dir / "trajectory_2d_projections.png")


def plot_error_over_time():
    """Plot position error ||p_est(t) - p_gt(t)|| over the 15-second duration."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), dpi=150, sharex=True)
    tiers = [("T1", "case_01", "Tier 1: Checkpoints (T1_01) - Drift Clamped by Checkpoints"),
             ("T2", "case_01", "Tier 2: IMU Blackout (T2_01) - Dead-Reckoning Divergence during Blackout"),
             ("T3", "case_01", "Tier 3: Sensor Degradation (T3_01) - Spike Injections & Noise")]

    for i, (tier, case_id, title) in enumerate(tiers):
        ax = axes[i]
        for app in ["A1", "A2", "A3", "B1"]:
            gt, est = load_aligned(tier, case_id, app)
            t_s = (gt.timestamps - gt.timestamps[0]) * 1e-9
            error_m = np.linalg.norm(est.positions - gt.positions, axis=1)

            ax.plot(t_s, error_m, label=f"{app} (RMSE: {np.sqrt(np.mean(error_m**2)):.2f}m)",
                    color=colors[app], lw=1.8, alpha=0.9)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Position Error [m]", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper left", fontsize=8)

    axes[-1].set_xlabel("Time [seconds]", fontsize=10)
    plt.tight_layout()
    out_file = plots_dir / "position_error_over_time.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_file}")
    if artifact_dir.exists():
        shutil.copy(out_file, artifact_dir / "position_error_over_time.png")


def plot_accuracy_benchmarks():
    """Plot comprehensive benchmark comparison bar charts and tier breakdown."""
    df = pd.read_csv(workspace / "experiments" / "MH01" / "consolidated_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

    # 1. Overall Mean ATE RMSE by Approach
    ax1 = axes[0]
    overall = df.groupby("approach")["ate_rmse_m"].agg(["mean", "std"]).reindex(["A1", "A2", "A3", "B1", "B2", "B3"])
    bars = ax1.bar(overall.index, overall["mean"], yerr=overall["std"], capsize=5,
                   color=[colors[a] for a in overall.index], edgecolor="black", alpha=0.85)

    ax1.set_title("Overall Mean ATE RMSE by Approach (30 Cases)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("ATE RMSE [meters]", fontsize=10)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, h + 8, f"{h:.1f}m",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Highlight best
    ax1.annotate("Best Overall: A3 (105.97m)",
                 xy=(2, overall.loc["A3", "mean"]),
                 xytext=(2, overall.loc["A3", "mean"] + 50),
                 ha="center",
                 arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
                 fontweight="bold", color="#2A9D8F")

    # 2. Grouped Bar Chart by Degradation Tier
    ax2 = axes[1]
    tier_pivot = df.pivot_table(index="approach", columns="degradation_type", values="ate_rmse_m", aggfunc="mean")
    tier_pivot = tier_pivot.reindex(["A1", "A2", "A3", "B1", "B2", "B3"])

    x = np.arange(len(tier_pivot.index))
    width = 0.25

    rects1 = ax2.bar(x - width, tier_pivot["T1"], width, label="Tier 1: Checkpoints", color="#457B9D", edgecolor="black")
    rects2 = ax2.bar(x, tier_pivot["T2"], width, label="Tier 2: Blackout", color="#E76F51", edgecolor="black")
    rects3 = ax2.bar(x + width, tier_pivot["T3"], width, label="Tier 3: Sensor Degradation", color="#2A9D8F", edgecolor="black")

    ax2.set_title("ATE RMSE by Degradation Tier [meters]", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(tier_pivot.index, fontsize=10)
    ax2.set_ylabel("ATE RMSE [meters]", fontsize=10)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    out_file = plots_dir / "accuracy_benchmark_summary.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_file}")
    if artifact_dir.exists():
        shutil.copy(out_file, artifact_dir / "accuracy_benchmark_summary.png")


def main():
    print("Generating trajectory and benchmark plots...")
    plot_3d_trajectories()
    plot_2d_projections()
    plot_error_over_time()
    plot_accuracy_benchmarks()
    print("All plots generated successfully!")


if __name__ == "__main__":
    main()

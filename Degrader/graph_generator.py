"""
graph_generator.py - Generates publication-quality sensor plots for T2 blackout and T3 degradation scenarios.
Uses headless matplotlib (Agg backend) and saves high-resolution PNGs.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from degradation import identify_imu_columns


class GraphGenerator:
    """Generates visualization graphs for T2 and T3 tiers."""

    @staticmethod
    def generate_t2_graph(
        original_imu_df: pd.DataFrame,
        t2_result: Dict[str, Any],
        output_path: Path,
    ):
        """
        Generates T2 IMU blackout graph covering the full 15 seconds.
        Highlights:
          - Sensor availability up to blackout onset
          - Blackout onset vertical line
          - 5-second blind prediction region (shaded)
          - Remaining blackout region (shaded)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        ts_col, acc_cols, gyr_cols = identify_imu_columns(original_imu_df)
        t_orig = original_imu_df[ts_col].to_numpy(dtype=np.int64)
        t0 = t_orig[0]
        rel_t_orig = (t_orig - t0) * 1e-9

        # T2 input data
        imu_input_df = t2_result["imu_input_df"]
        t_input = imu_input_df[ts_col].to_numpy(dtype=np.int64)
        rel_t_input = (t_input - t0) * 1e-9

        onset_s = t2_result["onset_rel_s"]
        blind_end_s = t2_result["blind_end_rel_s"]
        interval_end_s = float(rel_t_orig[-1])

        # Accel and Gyro norms / magnitudes for clear representation
        acc_input = imu_input_df[acc_cols].to_numpy(dtype=float)
        acc_mag_input = np.linalg.norm(acc_input, axis=1)

        gyr_input = imu_input_df[gyr_cols].to_numpy(dtype=float)
        gyr_mag_input = np.linalg.norm(gyr_input, axis=1)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

        # Accelerometer Subplot
        ax1.plot(rel_t_input, acc_mag_input, color="#1f77b4", lw=1.2, label="Accelerometer Mag (Available)")
        ax1.set_ylabel("Linear Accel |a| [m/s²]", fontsize=11, fontweight="bold")
        ax1.set_title(
            f"T2 IMU Blackout Scenario (Onset: {onset_s:.2f}s | Blind Prediction: {onset_s:.2f}s – {blind_end_s:.2f}s)",
            fontsize=13, fontweight="bold", pad=12
        )
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Gyroscope Subplot
        ax2.plot(rel_t_input, gyr_mag_input, color="#2ca02c", lw=1.2, label="Gyroscope Mag (Available)")
        ax2.set_xlabel("Relative Time [seconds]", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Angular Velocity |ω| [rad/s]", fontsize=11, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.5)

        for ax in [ax1, ax2]:
            ax.set_xlim(0, max(15.0, interval_end_s))
            # Blackout onset marker
            ax.axvline(onset_s, color="red", linestyle="--", lw=2, label="Blackout Onset")
            # 5-second blind prediction region
            ax.axvspan(onset_s, min(blind_end_s, interval_end_s), color="orange", alpha=0.25, label="5s Blind Prediction")
            # Remaining blackout
            if blind_end_s < interval_end_s:
                ax.axvspan(blind_end_s, interval_end_s, color="red", alpha=0.15, label="Remaining Blackout")

            # Annotation for no sensor observations
            mid_blackout = (onset_s + interval_end_s) / 2.0
            ylim = ax.get_ylim()
            ax.text(
                mid_blackout, (ylim[0] + ylim[1]) / 2.0,
                "NO SENSOR OBSERVATIONS\n(BLACKOUT)",
                color="#b30000", fontsize=11, fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffe6e6", edgecolor="#cc0000", alpha=0.85)
            )

        # Build combined deduplicated legend
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys(), loc="upper left", framealpha=0.9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def generate_t3_graphs(
        t3_result: Dict[str, Any],
        output_dir: Path,
    ):
        """
        Generates separate high-resolution graphs for Accelerometer and Gyroscope degradation.
        Displays X, Y, Z channels with clean before, degraded window (shaded), and clean after.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        degraded_df = t3_result["imu_input_df"]
        ts_col, acc_cols, gyr_cols = identify_imu_columns(degraded_df)

        timestamps = degraded_df[ts_col].to_numpy(dtype=np.int64)
        t0 = timestamps[0]
        rel_t = (timestamps - t0) * 1e-9

        deg_start_s = t3_result["start_time_s"]
        deg_end_s = t3_result["end_time_s"]

        # 1. Accelerometer Degradation Graph
        acc_path = output_dir / "accelerometer_degradation.png"
        fig_acc, axes_acc = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        acc_colors = ["#e41a1c", "#377eb8", "#4daf4a"]
        acc_labels = ["a_x", "a_y", "a_z"]

        for idx, (ax, col, label, color) in enumerate(zip(axes_acc, acc_cols, acc_labels, acc_colors)):
            vals = degraded_df[col].to_numpy(dtype=float)
            ax.plot(rel_t, vals, color=color, lw=1.1, label=f"{label} ({col})")
            ax.set_ylabel(f"{label} [m/s²]", fontsize=10, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.axvspan(deg_start_s, deg_end_s, color="yellow", alpha=0.35, label="2s Degraded Region" if idx == 0 else "")
            ax.axvline(deg_start_s, color="black", linestyle=":", lw=1.5)
            ax.axvline(deg_end_s, color="black", linestyle=":", lw=1.5)
            ax.legend(loc="upper right", framealpha=0.9)

        axes_acc[0].set_title(
            f"T3 Accelerometer Degradation (Degraded Region: {deg_start_s:.2f}s – {deg_end_s:.2f}s)",
            fontsize=13, fontweight="bold", pad=10
        )
        axes_acc[-1].set_xlabel("Relative Time [seconds]", fontsize=11, fontweight="bold")
        axes_acc[-1].set_xlim(0, max(15.0, rel_t[-1]))

        plt.tight_layout()
        plt.savefig(acc_path, dpi=200, bbox_inches="tight")
        plt.close(fig_acc)

        # 2. Gyroscope Degradation Graph
        gyr_path = output_dir / "gyroscope_degradation.png"
        fig_gyr, axes_gyr = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        gyr_colors = ["#984ea3", "#ff7f00", "#a65628"]
        gyr_labels = ["ω_x", "ω_y", "ω_z"]

        for idx, (ax, col, label, color) in enumerate(zip(axes_gyr, gyr_cols, gyr_labels, gyr_colors)):
            vals = degraded_df[col].to_numpy(dtype=float)
            ax.plot(rel_t, vals, color=color, lw=1.1, label=f"{label} ({col})")
            ax.set_ylabel(f"{label} [rad/s]", fontsize=10, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.axvspan(deg_start_s, deg_end_s, color="yellow", alpha=0.35, label="2s Degraded Region" if idx == 0 else "")
            ax.axvline(deg_start_s, color="black", linestyle=":", lw=1.5)
            ax.axvline(deg_end_s, color="black", linestyle=":", lw=1.5)
            ax.legend(loc="upper right", framealpha=0.9)

        axes_gyr[0].set_title(
            f"T3 Gyroscope Degradation (Degraded Region: {deg_start_s:.2f}s – {deg_end_s:.2f}s)",
            fontsize=13, fontweight="bold", pad=10
        )
        axes_gyr[-1].set_xlabel("Relative Time [seconds]", fontsize=11, fontweight="bold")
        axes_gyr[-1].set_xlim(0, max(15.0, rel_t[-1]))

        plt.tight_layout()
        plt.savefig(gyr_path, dpi=200, bbox_inches="tight")
        plt.close(fig_gyr)

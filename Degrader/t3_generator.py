"""
t3_generator.py - Generates Tier 3 (T3) Sensor Degradation Evaluation data.
Randomly selects a 2-second region beginning between 5 and 7 seconds,
applies adapted IMU degradation only inside that region,
and keeps clean regions before and after exactly unchanged.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from degradation import apply_sensor_degradation, DEFAULT_DEGRADATION_CONFIG


class T3Generator:
    """Generates T3 sensor degradation scenario."""

    DEGRADATION_DURATION_S: float = 2.0

    def __init__(self, imu_df: pd.DataFrame, gt_df: pd.DataFrame):
        self.imu_df = imu_df.copy()
        self.gt_df = gt_df.copy()
        self.ts_col = self.imu_df.columns[0]
        self.timestamps = self.imu_df[self.ts_col].to_numpy(dtype=np.int64)

    def generate(
        self,
        seed: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Selects a random start time between 5.0 and 7.0 seconds.
        Degrades the 2.0-second region while preserving pre- and post-segments.
        """
        rng = np.random.default_rng(seed)

        # Random start between 5.0 and 7.0 seconds
        start_time_s = float(rng.uniform(5.0, 7.0))

        # Optional parameter randomization for reruns
        run_config = dict(DEFAULT_DEGRADATION_CONFIG)
        if config:
            run_config.update(config)
        else:
            # Vary intensity slightly per rerun if no explicit config passed
            run_config["accel_white_max"] = float(rng.uniform(0.15, 0.35))
            run_config["gyro_white_max"] = float(rng.uniform(0.012, 0.030))
            run_config["accel_spike_mag"] = float(rng.uniform(5.0, 10.0))
            run_config["gyro_spike_mag"] = float(rng.uniform(1.0, 2.0))

        # Apply degradation strictly to the 2.0s window
        degraded_imu_df, deg_meta = apply_sensor_degradation(
            imu_df=self.imu_df,
            start_time_s=start_time_s,
            duration_s=self.DEGRADATION_DURATION_S,
            seed=seed,
            config=run_config,
        )

        metadata = {
            "tier": "T3",
            "description": "2-second sensor degradation with exact clean pre- and post-regions",
            "random_seed": seed,
            **deg_meta,
        }

        return {
            "imu_input_df": degraded_imu_df,
            "groundtruth_df": self.gt_df,
            "metadata": metadata,
            "start_time_s": float(start_time_s),
            "end_time_s": float(start_time_s + self.DEGRADATION_DURATION_S),
        }

    def save(self, output_dir: Path, result: Dict[str, Any]):
        """Saves imu_input.csv, groundtruth.csv, and metadata.json into T3 directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        imu_csv = output_dir / "imu_input.csv"
        gt_csv = output_dir / "groundtruth.csv"
        meta_json = output_dir / "metadata.json"

        result["imu_input_df"].to_csv(imu_csv, index=False)
        result["groundtruth_df"].to_csv(gt_csv, index=False)
        with open(meta_json, "w", encoding="utf-8") as f:
            json.dump(result["metadata"], f, indent=2)

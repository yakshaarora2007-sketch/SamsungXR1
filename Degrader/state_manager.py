"""
state_manager.py - Persistent state and lifecycle management.
Handles base interval archiving, atomic run generation (staging & rename),
and state.json manifest tracking.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Tuple, Any


class StateManager:
    """Manages interval directories, run numbering, staging, and state manifest."""

    def __init__(self, root_dir: str = "generated_samples"):
        self.root_dir = Path(root_dir)
        self.current_interval_dir = self.root_dir / "current_interval"
        self.state_file = self.current_interval_dir / "state.json"

    def has_active_interval(self) -> bool:
        """Checks if a valid base interval is currently active."""
        if not self.current_interval_dir.exists():
            return False
        if not self.state_file.exists():
            return False
        imu_file = self.current_interval_dir / "original_imu.csv"
        gt_file = self.current_interval_dir / "original_groundtruth.csv"
        return imu_file.exists() and gt_file.exists()

    def load_state(self) -> Dict[str, Any]:
        """Loads state.json manifest."""
        if not self.state_file.exists():
            raise FileNotFoundError(
                "No active motion interval found. Run 'python generate_samples.py --init' first."
            )
        with open(self.state_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def clear_previous_runs_if_exists(self) -> bool:
        """
        Clears the current interval and all of its runs before initializing a new interval.
        """
        cleared = False
        if self.current_interval_dir.exists():
            shutil.rmtree(self.current_interval_dir)
            cleared = True

        return cleared

    def init_interval(
        self,
        interval_data: Dict[str, Any],
    ) -> Path:
        """
        Sets up current_interval directory with original IMU, Ground Truth, and metadata.
        """
        self.current_interval_dir.mkdir(parents=True, exist_ok=True)

        seq_name = interval_data["sequence_name"]
        start_ts = interval_data["metadata"]["start_timestamp_ns"]
        interval_id = f"euroc_{seq_name}_{start_ts}"

        # Save original datasets
        imu_path = self.current_interval_dir / "original_imu.csv"
        gt_path = self.current_interval_dir / "original_groundtruth.csv"
        meta_path = self.current_interval_dir / "base_interval_metadata.json"

        interval_data["imu_df"].to_csv(imu_path, index=False)
        interval_data["gt_df"].to_csv(gt_path, index=False)

        base_meta = dict(interval_data["metadata"])
        base_meta["interval_id"] = interval_id
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(base_meta, f, indent=2)

        # Initialize state manifest
        state = {
            "interval_id": interval_id,
            "sequence_name": seq_name,
            "start_timestamp_ns": start_ts,
            "end_timestamp_ns": interval_data["metadata"]["end_timestamp_ns"],
            "nominal_duration_s": interval_data["metadata"]["nominal_duration_s"],
            "current_run_number": 0,
            "created_at": datetime.now().isoformat(),
            "runs": [],
        }

        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        return self.current_interval_dir

    def get_next_run_number(self) -> int:
        state = self.load_state()
        return state.get("current_run_number", 0) + 1

    def get_staging_directory(self, run_num: int) -> Path:
        """Creates a hidden staging directory for safe generation."""
        staging_dir = self.current_interval_dir / f".staging_run_{run_num:03d}"
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)
        return staging_dir

    def commit_staged_run(self, staging_dir: Path, run_num: int, run_meta: Dict[str, Any]) -> Path:
        """
        Atomically commits the validated staged run to run_XXX and updates state.json.
        """
        final_dir = self.current_interval_dir / f"run_{run_num:03d}"
        if final_dir.exists():
            raise FileExistsError(f"Target run directory already exists: {final_dir}")

        # Atomic rename
        shutil.move(str(staging_dir), str(final_dir))

        # Update state manifest
        state = self.load_state()
        state["current_run_number"] = run_num
        run_record = {
            "run_number": run_num,
            "directory": f"run_{run_num:03d}",
            "created_at": datetime.now().isoformat(),
            **run_meta,
        }
        state["runs"].append(run_record)

        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        return final_dir

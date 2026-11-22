"""
experiment/manager.py - Manages experiment directory creation, metadata persistence,
and experiment retrieval. Never overwrites existing experiments.
"""

from datetime import datetime
import json
from pathlib import Path
import platform
import sys
from typing import Dict, List, Optional, Any, Tuple
import uuid

from dataset.loader import SourceSegment


class ExperimentManager:
    """Manages experiment workspaces and traceability metadata."""

    def __init__(self, outputs_dir: str = "outputs"):
        self.outputs_dir = Path(outputs_dir)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def generate_experiment_id(sequence_name: str) -> str:
        """
        Generates a human-readable, unique experiment identifier:
        Format: <sequence>_<YYYYMMDD>_<HHMMSS>_<4hex>
        """
        # Clean sequence name (e.g. MH_01_easy -> MH01)
        short_seq = sequence_name.replace("_", "").upper()
        if "MH01" in short_seq:
            short_seq = "MH01"
        elif "MH02" in short_seq:
            short_seq = "MH02"
        elif "V101" in short_seq:
            short_seq = "V101"

        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        rand_hex = uuid.uuid4().hex[:4]
        return f"{short_seq}_{now_str}_{rand_hex}"

    def create_experiment_dir(self, sequence_name: str) -> Tuple[str, Path]:
        """
        Creates a new unique experiment directory.
        Guarantees never to overwrite an existing directory.
        """
        while True:
            exp_id = self.generate_experiment_id(sequence_name)
            exp_dir = self.outputs_dir / exp_id
            if not exp_dir.exists():
                exp_dir.mkdir(parents=True, exist_ok=False)
                # Create standard subdirectories
                (exp_dir / "source").mkdir(parents=True, exist_ok=True)
                (exp_dir / "T1").mkdir(parents=True, exist_ok=True)
                (exp_dir / "T2").mkdir(parents=True, exist_ok=True)
                (exp_dir / "T3").mkdir(parents=True, exist_ok=True)
                (exp_dir / "reports").mkdir(parents=True, exist_ok=True)
                return exp_id, exp_dir

    def get_experiment_dir(self, experiment_id: str) -> Path:
        """Locates an experiment directory by ID or path."""
        exp_dir = self.outputs_dir / experiment_id
        if not exp_dir.exists():
            # Check if experiment_id is a full path
            as_path = Path(experiment_id)
            if as_path.exists() and as_path.is_dir():
                return as_path
            raise FileNotFoundError(f"Experiment '{experiment_id}' does not exist in {self.outputs_dir}")
        return exp_dir

    def save_experiment_metadata(
        self,
        experiment_dir: Path,
        exp_id: str,
        segment: SourceSegment,
        degrader_config: Dict[str, Any],
        harness_version: str = "1.0.0",
        t1_target: int = 10,
        t2_target: int = 10,
        t3_target: int = 10,
        run_seeds: Optional[Dict[str, Any]] = None,
        status: str = "INITIALIZED",
    ) -> Path:
        """Writes top-level metadata.json for the experiment."""
        meta_path = experiment_dir / "metadata.json"

        # Software versions
        import numpy as np
        import pandas as pd
        import scipy

        sw_versions = {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "harness": harness_version,
            "os": platform.system() + " " + platform.release(),
        }

        meta = {
            "experiment_id": exp_id,
            "creation_timestamp": datetime.now().isoformat(),
            "status": status,
            "dataset_name": segment.metadata.get("dataset_name", "EuRoC MAV"),
            "sequence_name": segment.sequence_name,
            "original_dataset_path": segment.dataset_path,
            "segment_start_timestamp_ns": segment.metadata.get("start_timestamp_ns"),
            "segment_end_timestamp_ns": segment.metadata.get("end_timestamp_ns"),
            "segment_duration_s": segment.metadata.get("segment_duration_s", 15.0),
            "actual_duration_s": segment.metadata.get("actual_duration_s"),
            "segment_selection_seed": segment.metadata.get("selection_seed"),
            "degrader_configuration": degrader_config,
            "runs_target": {
                "T1": t1_target,
                "T2": t2_target,
                "T3": t3_target,
                "total": t1_target + t2_target + t3_target,
            },
            "individual_run_seeds": run_seeds or {},
            "software_versions": sw_versions,
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return meta_path

    def load_experiment_metadata(self, experiment_id: str) -> Dict[str, Any]:
        """Loads experiment metadata.json."""
        exp_dir = self.get_experiment_dir(experiment_id)
        meta_path = exp_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.json missing in experiment directory: {exp_dir}")

        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_experiments(self) -> List[Dict[str, Any]]:
        """Lists all experiment directories in outputs_dir with their status."""
        results = []
        if not self.outputs_dir.exists():
            return results

        for item in sorted(self.outputs_dir.iterdir()):
            if not item.is_dir():
                continue

            meta_file = item / "metadata.json"
            if meta_file.exists():
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    results.append({
                        "experiment_id": meta.get("experiment_id", item.name),
                        "path": str(item),
                        "sequence": meta.get("sequence_name", "unknown"),
                        "created_at": meta.get("creation_timestamp", "unknown"),
                        "status": meta.get("status", "unknown"),
                        "t1_runs": len(list((item / "T1").glob("run_*"))),
                        "t2_runs": len(list((item / "T2").glob("run_*"))),
                        "t3_runs": len(list((item / "T3").glob("run_*"))),
                    })
                except Exception:
                    results.append({
                        "experiment_id": item.name,
                        "path": str(item),
                        "status": "CORRUPTED_OR_INCOMPLETE",
                    })
        return results

    def get_next_batch_dir(self, experiment_dir: Path) -> Tuple[int, Path]:
        """
        Determines the next degradation batch version directory (e.g. batch_02)
        to prevent overwriting previous runs.
        """
        existing_batches = list(experiment_dir.glob("batch_*"))
        next_batch_num = len(existing_batches) + 2  # batch_01 is the root T1/T2/T3
        batch_dir = experiment_dir / f"batch_{next_batch_num:02d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        (batch_dir / "T1").mkdir(parents=True, exist_ok=True)
        (batch_dir / "T2").mkdir(parents=True, exist_ok=True)
        (batch_dir / "T3").mkdir(parents=True, exist_ok=True)
        (batch_dir / "reports").mkdir(parents=True, exist_ok=True)
        return next_batch_num, batch_dir

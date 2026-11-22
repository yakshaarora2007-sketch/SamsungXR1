"""
dataset/loader.py - EuRoC MAV dataset discovery, validation, and 15-second segment extraction.
Handles missing datasets gracefully with clear instructions.
Preserves exact int64 nanosecond timestamps and original sensor headers.
"""

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


class MissingDatasetError(FileNotFoundError):
    """Raised when the specified EuRoC dataset directory or sequence is missing."""

    def __init__(self, message: str, searched_paths: Optional[List[Path]] = None, sequence: Optional[str] = None):
        super().__init__(message)
        self.searched_paths = searched_paths or []
        self.sequence = sequence

    @classmethod
    def create_instructional(
        cls,
        root_dir: Path,
        sequence: str,
        searched_paths: Optional[List[Path]] = None
    ) -> "MissingDatasetError":
        paths_str = "\n".join(f"  - {p.resolve()}" for p in (searched_paths or [root_dir]))
        instructions = (
            f"\n[EUROC DATASET NOT FOUND]\n"
            f"The requested sequence '{sequence}' could not be located.\n"
            f"Searched locations:\n{paths_str}\n\n"
            f"How to resolve:\n"
            f"1. Download EuRoC MAV '{sequence}' (e.g. MH_01_easy):\n"
            f"   Download URL: http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip\n"
            f"2. Extract the archive into your dataset root directory:\n"
            f"   Expected structure:\n"
            f"     {root_dir.resolve()}/{sequence}/mav0/imu0/data.csv\n"
            f"     {root_dir.resolve()}/{sequence}/mav0/state_groundtruth_estimate0/data.csv\n"
            f"3. Alternatively, supply your existing dataset path with the CLI flag:\n"
            f"   python main.py --sequence {sequence} --dataset-dir <path_to_euroc_directory> --new-experiment\n"
        )
        return cls(instructions, searched_paths=searched_paths, sequence=sequence)


@dataclass
class SourceSegment:
    """Represents an extracted 15-second segment with sensor data and full origin metadata."""
    sequence_name: str
    dataset_path: str
    imu_df: pd.DataFrame
    gt_df: pd.DataFrame
    metadata: Dict[str, Any]

    def save(self, source_dir: Path) -> Dict[str, Path]:
        """Saves imu.csv, groundtruth.csv, and segment_info.json into source_dir."""
        source_dir = Path(source_dir)
        source_dir.mkdir(parents=True, exist_ok=True)

        imu_path = source_dir / "imu.csv"
        gt_path = source_dir / "groundtruth.csv"
        info_path = source_dir / "segment_info.json"

        self.imu_df.to_csv(imu_path, index=False)
        self.gt_df.to_csv(gt_path, index=False)
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

        return {
            "imu": imu_path,
            "groundtruth": gt_path,
            "segment_info": info_path,
        }

    @classmethod
    def load(cls, source_dir: Path) -> "SourceSegment":
        """Loads a previously saved 15-second source segment from source_dir."""
        source_dir = Path(source_dir)
        imu_path = source_dir / "imu.csv"
        gt_path = source_dir / "groundtruth.csv"
        info_path = source_dir / "segment_info.json"

        if not imu_path.exists() or not gt_path.exists() or not info_path.exists():
            raise FileNotFoundError(f"Missing required source segment files in {source_dir}")

        imu_df = pd.read_csv(imu_path)
        gt_df = pd.read_csv(gt_path)
        with open(info_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return cls(
            sequence_name=metadata.get("sequence_name", "unknown"),
            dataset_path=metadata.get("dataset_path", str(source_dir)),
            imu_df=imu_df,
            gt_df=gt_df,
            metadata=metadata,
        )


class EuRoCLoader:
    """Handles EuRoC sequence discovery, verification, and 15-second segment slicing."""

    DURATION_NS: int = 15_000_000_000  # 15.0 seconds in nanoseconds

    # Sequence alias mapping
    SEQUENCE_ALIASES: Dict[str, str] = {
        "mh01": "MH_01_easy",
        "mh_01": "MH_01_easy",
        "mh_01_easy": "MH_01_easy",
        "mh02": "MH_02_easy",
        "mh_02": "MH_02_easy",
        "mh_02_easy": "MH_02_easy",
        "mh03": "MH_03_medium",
        "mh_03": "MH_03_medium",
        "mh_03_medium": "MH_03_medium",
        "mh04": "MH_04_difficult",
        "mh_04": "MH_04_difficult",
        "mh_04_difficult": "MH_04_difficult",
        "mh05": "MH_05_difficult",
        "mh_05": "MH_05_difficult",
        "mh_05_difficult": "MH_05_difficult",
        "v101": "V1_01_easy",
        "v1_01_easy": "V1_01_easy",
        "v102": "V1_02_medium",
        "v1_02_medium": "V1_02_medium",
        "v103": "V1_03_difficult",
        "v1_03_difficult": "V1_03_difficult",
    }

    def __init__(self, root_dir: str = "dataset/euroc"):
        self.root_dir = Path(root_dir)

    def resolve_sequence_name(self, name: str) -> str:
        """Resolves sequence alias (e.g. MH01 -> MH_01_easy)."""
        key = name.strip().lower()
        return self.SEQUENCE_ALIASES.get(key, name.strip())

    def find_sequence(self, sequence_name: str) -> Tuple[str, Path, Path, Path]:
        """
        Locates the directory and IMU/Ground-truth CSV files for a given sequence.
        Returns (canonical_name, seq_root, imu_csv, gt_csv).
        Raises MissingDatasetError if not found.
        """
        canonical_name = self.resolve_sequence_name(sequence_name)
        candidates = [canonical_name, sequence_name]

        # Potential search roots
        search_roots = [
            self.root_dir,
            Path("dataset/euroc"),
            Path("dataset"),
            Path("vicon_room1"),
            Path("../vicon_room1"),
        ]

        searched_paths = []

        # 1. Direct check if self.root_dir itself is the sequence folder (contains mav0)
        direct_mav0 = self.root_dir / "mav0"
        if direct_mav0.exists():
            imu_csv = direct_mav0 / "imu0" / "data.csv"
            gt_csv = direct_mav0 / "state_groundtruth_estimate0" / "data.csv"
            if not gt_csv.exists():
                gt_csv = direct_mav0 / "vicon0" / "data.csv"
            if imu_csv.exists() and gt_csv.exists():
                return canonical_name, self.root_dir, imu_csv, gt_csv

        # 2. Search candidate sequence directories under all search roots
        for base_dir in search_roots:
            if not base_dir.exists():
                searched_paths.append(base_dir)
                continue

            for name_variant in candidates:
                seq_dir = base_dir / name_variant
                searched_paths.append(seq_dir)
                if seq_dir.exists() and seq_dir.is_dir():
                    imu_csv = seq_dir / "mav0" / "imu0" / "data.csv"
                    gt_csv = seq_dir / "mav0" / "state_groundtruth_estimate0" / "data.csv"
                    if not gt_csv.exists():
                        gt_csv = seq_dir / "mav0" / "vicon0" / "data.csv"

                    if imu_csv.exists() and gt_csv.exists():
                        return canonical_name, seq_dir, imu_csv, gt_csv

            # Also check case-insensitive match among children
            for child in base_dir.iterdir():
                if child.is_dir() and child.name.lower() in [c.lower() for c in candidates]:
                    imu_csv = child / "mav0" / "imu0" / "data.csv"
                    gt_csv = child / "mav0" / "state_groundtruth_estimate0" / "data.csv"
                    if not gt_csv.exists():
                        gt_csv = child / "mav0" / "vicon0" / "data.csv"
                    if imu_csv.exists() and gt_csv.exists():
                        return canonical_name, child, imu_csv, gt_csv

        raise MissingDatasetError.create_instructional(
            root_dir=self.root_dir,
            sequence=sequence_name,
            searched_paths=searched_paths
        )

    def extract_15s_segment(
        self,
        sequence_name: str = "MH01",
        seed: Optional[int] = None,
    ) -> SourceSegment:
        """
        Stage A: Randomly extracts ONE continuous 15-second segment from the selected sequence.
        Returns a SourceSegment containing the unmodified sensor slices and complete origin metadata.
        """
        canonical_name, seq_root, imu_path, gt_path = self.find_sequence(sequence_name)

        # Load full raw datasets
        imu_df = pd.read_csv(imu_path)
        gt_df = pd.read_csv(gt_path)

        imu_ts_col = imu_df.columns[0]
        gt_ts_col = gt_df.columns[0]

        imu_timestamps = imu_df[imu_ts_col].to_numpy(dtype=np.int64)
        gt_timestamps = gt_df[gt_ts_col].to_numpy(dtype=np.int64)

        seq_start_ts = int(min(imu_timestamps[0], gt_timestamps[0]))
        seq_end_ts = int(max(imu_timestamps[-1], gt_timestamps[-1]))
        seq_duration_s = float((seq_end_ts - seq_start_ts) * 1e-9)

        # Overlap window
        t_overlap_start = int(max(imu_timestamps[0], gt_timestamps[0]))
        t_overlap_end = int(min(imu_timestamps[-1], gt_timestamps[-1]))
        available_span_ns = t_overlap_end - t_overlap_start

        if available_span_ns < self.DURATION_NS:
            raise ValueError(
                f"Sequence '{canonical_name}' overlapping duration ({available_span_ns * 1e-9:.2f}s) "
                f"is less than required 15.0 seconds."
            )

        max_start_ts = t_overlap_end - self.DURATION_NS
        valid_starts = imu_timestamps[
            (imu_timestamps >= t_overlap_start) & (imu_timestamps <= max_start_ts)
        ]

        if len(valid_starts) == 0:
            raise ValueError(f"No valid starting timestamp for 15s interval in '{canonical_name}'")

        rng = np.random.default_rng(seed)
        start_ts = int(rng.choice(valid_starts))
        end_ts = int(start_ts + self.DURATION_NS)

        # Slice IMU and GT data
        imu_slice = imu_df[(imu_df[imu_ts_col] >= start_ts) & (imu_df[imu_ts_col] <= end_ts)].copy().reset_index(drop=True)
        gt_slice = gt_df[(gt_df[gt_ts_col] >= start_ts) & (gt_df[gt_ts_col] <= end_ts)].copy().reset_index(drop=True)

        actual_duration_s = float((imu_slice[imu_ts_col].iloc[-1] - imu_slice[imu_ts_col].iloc[0]) * 1e-9)
        rel_start_s = float((start_ts - seq_start_ts) * 1e-9)
        rel_end_s = float((end_ts - seq_start_ts) * 1e-9)

        imu_dt_s = np.diff(imu_slice[imu_ts_col].to_numpy()) * 1e-9
        median_dt_ms = float(np.median(imu_dt_s) * 1000.0) if len(imu_dt_s) > 0 else 0.0

        metadata = {
            "dataset_name": "EuRoC MAV",
            "sequence_name": canonical_name,
            "dataset_path": str(seq_root.resolve()),
            "sequence_start_timestamp_ns": seq_start_ts,
            "sequence_end_timestamp_ns": seq_end_ts,
            "original_sequence_duration_s": seq_duration_s,
            "segment_duration_s": 15.0,
            "nominal_duration_s": 15.0,
            "actual_duration_s": actual_duration_s,
            "start_timestamp_ns": start_ts,
            "end_timestamp_ns": end_ts,
            "start_time_relative_to_sequence_s": rel_start_s,
            "end_time_relative_to_sequence_s": rel_end_s,
            "imu_sample_count": int(len(imu_slice)),
            "gt_sample_count": int(len(gt_slice)),
            "imu_median_dt_ms": median_dt_ms,
            "selection_seed": seed,
        }

        return SourceSegment(
            sequence_name=canonical_name,
            dataset_path=str(seq_root.resolve()),
            imu_df=imu_slice,
            gt_df=gt_slice,
            metadata=metadata,
        )

"""
tests/test_dataset_loader.py - Tests for dataset discovery, missing dataset handling, and 15s segment extraction.
"""

from pathlib import Path
import numpy as np
import pytest

from dataset.loader import EuRoCLoader, SourceSegment, MissingDatasetError


def test_missing_dataset_handling():
    """Verify missing dataset raises informative MissingDatasetError with instructions."""
    non_existent = Path("non_existent_path_xyz123")
    loader = EuRoCLoader(root_dir=str(non_existent))

    with pytest.raises(MissingDatasetError) as exc_info:
        loader.extract_15s_segment("NonExistentSequence_999")

    err_msg = str(exc_info.value)
    assert "[EUROC DATASET NOT FOUND]" in err_msg
    assert "NonExistentSequence_999" in err_msg
    assert "http://robotics.ethz.ch/~asl-datasets" in err_msg
    assert "mav0/imu0/data.csv" in err_msg


def test_sequence_alias_resolution():
    """Verify alias mapping correctly resolves sequence variants."""
    loader = EuRoCLoader()
    assert loader.resolve_sequence_name("MH01") == "MH_01_easy"
    assert loader.resolve_sequence_name("mh_01") == "MH_01_easy"
    assert loader.resolve_sequence_name("V101") == "V1_01_easy"
    assert loader.resolve_sequence_name("CustomSeq") == "CustomSeq"


def test_synthetic_segment_extraction(synthetic_euroc_dir: Path):
    """Verify 15-second segment extraction on synthetic EuRoC data."""
    loader = EuRoCLoader(root_dir=str(synthetic_euroc_dir))
    segment = loader.extract_15s_segment(sequence_name="MH01", seed=42)

    assert isinstance(segment, SourceSegment)
    assert segment.sequence_name == "MH_01_easy"

    # Verify duration is nominally 15.0s
    imu_df = segment.imu_df
    gt_df = segment.gt_df

    ts_col = imu_df.columns[0]
    t_start = imu_df[ts_col].iloc[0]
    t_end = imu_df[ts_col].iloc[-1]
    dur_s = (t_end - t_start) * 1e-9

    assert 14.99 <= dur_s <= 15.01
    assert segment.metadata["nominal_duration_s"] == 15.0
    assert segment.metadata["imu_sample_count"] == len(imu_df)
    assert segment.metadata["gt_sample_count"] == len(gt_df)

    # Verify origin tracing
    assert "sequence_start_timestamp_ns" in segment.metadata
    assert "start_time_relative_to_sequence_s" in segment.metadata
    assert segment.metadata["start_time_relative_to_sequence_s"] >= 0.0


def test_source_segment_save_and_load(synthetic_euroc_dir: Path, tmp_path: Path):
    """Verify source segment saves to source/ and can be reloaded without modification."""
    loader = EuRoCLoader(root_dir=str(synthetic_euroc_dir))
    segment = loader.extract_15s_segment(sequence_name="MH01", seed=100)

    save_dir = tmp_path / "test_exp" / "source"
    saved_paths = segment.save(save_dir)

    assert saved_paths["imu"].exists()
    assert saved_paths["groundtruth"].exists()
    assert saved_paths["segment_info"].exists()

    reloaded = SourceSegment.load(save_dir)
    assert reloaded.sequence_name == segment.sequence_name
    assert len(reloaded.imu_df) == len(segment.imu_df)
    assert len(reloaded.gt_df) == len(segment.gt_df)
    assert np.all(reloaded.imu_df.to_numpy() == segment.imu_df.to_numpy())

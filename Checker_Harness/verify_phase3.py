"""
Phase 3 Verification Script:
Verifies that the degradation generator (T1, T2, T3) functions cleanly on real EuRoC MH01 data
and satisfies all non-negotiable requirements and validators.
"""
import sys
import os
import tempfile
from pathlib import Path

# Add directories to sys.path
workspace_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(workspace_dir / "Checker_Harness"))
sys.path.insert(0, str(workspace_dir / "Degrader"))

from dataset.loader import EuRoCLoader
from degradation.adapter import DegradationAdapter
from validation import RunValidator

def main():
    print("=" * 60)
    print("PHASE 3 VERIFICATION: Degradation Engine on Real EuRoC MH01")
    print("=" * 60)

    loader = EuRoCLoader(root_dir=str(workspace_dir / "dataset" / "euroc"))
    seq_name, seq_dir, imu_p, gt_p = loader.find_sequence("MH01")
    print(f"[Phase 3] Located canonical sequence: '{seq_name}' in {seq_dir}")

    # Extract test 15s segment
    segment = loader.extract_15s_segment("MH01", seed=42)
    print(f"[Phase 3] Extracted segment: IMU rows={len(segment.imu_df)}, GT rows={len(segment.gt_df)}")
    print(f"[Phase 3] Time bounds: {segment.metadata['start_timestamp_ns']} to {segment.metadata['end_timestamp_ns']}")

    adapter = DegradationAdapter()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Test T1
        print("[Phase 3] Testing T1 Generator (Checkpoints)...")
        res_t1 = adapter.execute_run(
            tier="T1",
            run_id="test_001",
            output_dir=tmp_path / "T1",
            source_imu_df=segment.imu_df,
            source_gt_df=segment.gt_df,
            parent_experiment_id="phase3_test",
            source_segment_path=str(tmp_path),
            seed=101,
        )
        assert res_t1["execution_status"] == "SUCCESS", f"T1 failed: {res_t1.get('error_details')}"
        assert res_t1["memory"]["status"] == "PASS", f"T1 memory violated: {res_t1['memory']}"
        print(f"  -> T1 PASS (Execution: {res_t1['duration_seconds']:.2f}s, Mem: {res_t1['memory']['algorithm_peak_mb']:.3f} MB)")

        # Test T2
        print("[Phase 3] Testing T2 Generator (IMU Blackout)...")
        res_t2 = adapter.execute_run(
            tier="T2",
            run_id="test_001",
            output_dir=tmp_path / "T2",
            source_imu_df=segment.imu_df,
            source_gt_df=segment.gt_df,
            parent_experiment_id="phase3_test",
            source_segment_path=str(tmp_path),
            seed=102,
        )
        assert res_t2["execution_status"] == "SUCCESS", f"T2 failed: {res_t2.get('error_details')}"
        assert res_t2["memory"]["status"] == "PASS", f"T2 memory violated: {res_t2['memory']}"
        print(f"  -> T2 PASS (Execution: {res_t2['duration_seconds']:.2f}s, Mem: {res_t2['memory']['algorithm_peak_mb']:.3f} MB)")

        # Test T3
        print("[Phase 3] Testing T3 Generator (Sensor Degradation)...")
        res_t3 = adapter.execute_run(
            tier="T3",
            run_id="test_001",
            output_dir=tmp_path / "T3",
            source_imu_df=segment.imu_df,
            source_gt_df=segment.gt_df,
            parent_experiment_id="phase3_test",
            source_segment_path=str(tmp_path),
            seed=103,
        )
        assert res_t3["execution_status"] == "SUCCESS", f"T3 failed: {res_t3.get('error_details')}"
        assert res_t3["memory"]["status"] == "PASS", f"T3 memory violated: {res_t3['memory']}"
        print(f"  -> T3 PASS (Execution: {res_t3['duration_seconds']:.2f}s, Mem: {res_t3['memory']['algorithm_peak_mb']:.3f} MB)")

    print("\n[PHASE 3 COMPLETE] All degradation generators verified and passing on real MH01 data.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

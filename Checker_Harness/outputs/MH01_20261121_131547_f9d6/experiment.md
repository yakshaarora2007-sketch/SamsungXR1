# Experiment Report

## Experiment Information

- Experiment ID: MH01_20261121_131547_f9d6
- Creation timestamp: 2026-11-21T13:15:47.848194
- Dataset: EuRoC MAV
- Sequence: MH_01_easy
- Original dataset path: C:\Users\Yaksha's Laptop\AppData\Local\Temp\tmpjck194mk\euroc\MH_01_easy
- Experiment status: COMPLETED

## Selected 15-Second Segment

- Segment duration: 15.0 seconds
- Start timestamp: 1403636582203555392
- End timestamp: 1403636597203555392
- Start time relative to original sequence: 2.45s
- End time relative to original sequence: 17.45s
- Original sequence duration: 17.50 seconds
- Number of IMU samples: 3001
- Number of ground-truth samples: 3001

## Degradation Summary

| Level | Total Runs | Successful | Failed |
|-------|------------|------------|--------|
| T1    | 10         | 10          | 0      |
| T2    | 10         | 10          | 0      |
| T3    | 10         | 10          | 0      |
| Total | 30         | 30          | 0      |

## Individual Run Results

| Run ID | Level | Status | Seed | Output Path |
|--------|-------|--------|------|-------------|
| run_001 | T1 | SUCCESS | 2103139804 | T1/run_001 |
| run_002 | T1 | SUCCESS | 2097446776 | T1/run_002 |
| run_003 | T1 | SUCCESS | 2121586239 | T1/run_003 |
| run_004 | T1 | SUCCESS | 816464123 | T1/run_004 |
| run_005 | T1 | SUCCESS | 367992409 | T1/run_005 |
| run_006 | T1 | SUCCESS | 1982656189 | T1/run_006 |
| run_007 | T1 | SUCCESS | 225255238 | T1/run_007 |
| run_008 | T1 | SUCCESS | 561980200 | T1/run_008 |
| run_009 | T1 | SUCCESS | 298028865 | T1/run_009 |
| run_010 | T1 | SUCCESS | 685255714 | T1/run_010 |
| run_001 | T2 | SUCCESS | 1151155068 | T2/run_001 |
| run_002 | T2 | SUCCESS | 253598991 | T2/run_002 |
| run_003 | T2 | SUCCESS | 1709091010 | T2/run_003 |
| run_004 | T2 | SUCCESS | 519189160 | T2/run_004 |
| run_005 | T2 | SUCCESS | 1688041804 | T2/run_005 |
| run_006 | T2 | SUCCESS | 684046402 | T2/run_006 |
| run_007 | T2 | SUCCESS | 1704011245 | T2/run_007 |
| run_008 | T2 | SUCCESS | 2070344413 | T2/run_008 |
| run_009 | T2 | SUCCESS | 2057996156 | T2/run_009 |
| run_010 | T2 | SUCCESS | 566183642 | T2/run_010 |
| run_001 | T3 | SUCCESS | 1193176983 | T3/run_001 |
| run_002 | T3 | SUCCESS | 947053435 | T3/run_002 |
| run_003 | T3 | SUCCESS | 547003392 | T3/run_003 |
| run_004 | T3 | SUCCESS | 1309687589 | T3/run_004 |
| run_005 | T3 | SUCCESS | 1926505261 | T3/run_005 |
| run_006 | T3 | SUCCESS | 1854612611 | T3/run_006 |
| run_007 | T3 | SUCCESS | 1387222073 | T3/run_007 |
| run_008 | T3 | SUCCESS | 1854905472 | T3/run_008 |
| run_009 | T3 | SUCCESS | 1113402597 | T3/run_009 |
| run_010 | T3 | SUCCESS | 1449296583 | T3/run_010 |

## Memory Constraint Report (Algorithm / Heap Level)

| Metric | Requirement | Status |
|---|---|---|
| Algorithm-level peak memory | <= 8 MB | 1.988 MB (2,084,394 bytes) |
| Python interpreter baseline | Excluded | 89.60 MB (Exempt from limit) |
| OS-level memory | Excluded | Exempt from limit |
| Native algorithm buffers | Included | Tracked in Algorithm Memory |
| Temporary algorithm allocations | Included | Tracked in Algorithm Memory |

### Memory Verification Details

- **Constraint Status:** **PASS**
- **Algorithm-Level Peak Memory:** 2,084,394 bytes (1.988 MB) / 8,388,608 bytes (8.000 MB)
- **Python Interpreter Baseline:** 93,954,048 bytes (89.60 MB) [Excluded]
- **Process RSS Peak (Diagnostic):** 113.08 MB
- **Measurement Methodology:** tracemalloc + native buffer tracking + process delta isolation
- **Measurement Coverage:** FULL


## Solver Evaluation

Status: Pending solver implementation.

## Errors and Warnings

None. All attempted runs completed successfully without errors.

## Reproducibility

- Segment selection seed: 1234
- Individual degradation seeds: run_001: 2103139804, run_002: 2097446776, run_003: 2121586239, run_004: 816464123, run_005: 367992409, run_006: 1982656189, run_007: 225255238, run_008: 561980200, run_009: 298028865, run_010: 685255714, run_001: 1151155068, run_002: 253598991, run_003: 1709091010, run_004: 519189160, run_005: 1688041804, run_006: 684046402, run_007: 1704011245, run_008: 2070344413, run_009: 2057996156, run_010: 566183642, run_001: 1193176983, run_002: 947053435, run_003: 547003392, run_004: 1309687589, run_005: 1926505261, run_006: 1854612611, run_007: 1387222073, run_008: 1854905472, run_009: 1113402597, run_010: 1449296583
- Degrader configuration:
```json
{
  "accel_white_max": 0.25,
  "gyro_white_max": 0.02,
  "accel_bias_rw_max": 0.015,
  "gyro_bias_rw_max": 0.001,
  "scale_rw_max": 0.002,
  "spike_prob_max": 0.008,
  "accel_spike_mag": 8.0,
  "gyro_spike_mag": 1.5,
  "stuck_prob_max": 0.015,
  "stuck_len": [
    2,
    8
  ],
  "drop_prob_max": 0.05,
  "burst_prob_max": 0.005,
  "burst_len": [
    4,
    25
  ],
  "dup_prob_max": 0.005
}
```
- Harness version: 1.0.0

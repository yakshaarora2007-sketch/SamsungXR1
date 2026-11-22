# Experiment Report

## Experiment Information

- Experiment ID: MH01_20261120_231117_7a84
- Creation timestamp: 2026-11-20T23:11:17.778353
- Dataset: EuRoC MAV
- Sequence: MH_01_easy
- Original dataset path: C:\Users\Yaksha's Laptop\AppData\Local\Temp\tmpvp1mrprt\euroc\MH_01_easy
- Experiment status: COMPLETED

## Selected 15-Second Segment

- Segment duration: 15.0 seconds
- Start timestamp: 1403636580203555392
- End timestamp: 1403636595203555392
- Start time relative to original sequence: 0.45s
- End time relative to original sequence: 15.45s
- Original sequence duration: 20.00 seconds
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
| run_001 | T1 | SUCCESS | 191664963 | T1/run_001 |
| run_002 | T1 | SUCCESS | 1662057957 | T1/run_002 |
| run_003 | T1 | SUCCESS | 1405681631 | T1/run_003 |
| run_004 | T1 | SUCCESS | 942484272 | T1/run_004 |
| run_005 | T1 | SUCCESS | 929893137 | T1/run_005 |
| run_006 | T1 | SUCCESS | 1843824992 | T1/run_006 |
| run_007 | T1 | SUCCESS | 184566854 | T1/run_007 |
| run_008 | T1 | SUCCESS | 1497586438 | T1/run_008 |
| run_009 | T1 | SUCCESS | 432652533 | T1/run_009 |
| run_010 | T1 | SUCCESS | 202244314 | T1/run_010 |
| run_001 | T2 | SUCCESS | 1130604997 | T2/run_001 |
| run_002 | T2 | SUCCESS | 2095133045 | T2/run_002 |
| run_003 | T2 | SUCCESS | 1580016183 | T2/run_003 |
| run_004 | T2 | SUCCESS | 1634535062 | T2/run_004 |
| run_005 | T2 | SUCCESS | 1540770719 | T2/run_005 |
| run_006 | T2 | SUCCESS | 1688060240 | T2/run_006 |
| run_007 | T2 | SUCCESS | 1102145672 | T2/run_007 |
| run_008 | T2 | SUCCESS | 275121930 | T2/run_008 |
| run_009 | T2 | SUCCESS | 1803345590 | T2/run_009 |
| run_010 | T2 | SUCCESS | 967196436 | T2/run_010 |
| run_001 | T3 | SUCCESS | 1074497555 | T3/run_001 |
| run_002 | T3 | SUCCESS | 796282693 | T3/run_002 |
| run_003 | T3 | SUCCESS | 392022359 | T3/run_003 |
| run_004 | T3 | SUCCESS | 1990212658 | T3/run_004 |
| run_005 | T3 | SUCCESS | 1678403330 | T3/run_005 |
| run_006 | T3 | SUCCESS | 1382689815 | T3/run_006 |
| run_007 | T3 | SUCCESS | 864178266 | T3/run_007 |
| run_008 | T3 | SUCCESS | 1766867109 | T3/run_008 |
| run_009 | T3 | SUCCESS | 1171300112 | T3/run_009 |
| run_010 | T3 | SUCCESS | 952224740 | T3/run_010 |

## Solver Evaluation

Status: Pending solver implementation.

## Errors and Warnings

None. All attempted runs completed successfully without errors.

## Reproducibility

- Segment selection seed: 42
- Individual degradation seeds: run_001: 191664963, run_002: 1662057957, run_003: 1405681631, run_004: 942484272, run_005: 929893137, run_006: 1843824992, run_007: 184566854, run_008: 1497586438, run_009: 432652533, run_010: 202244314, run_001: 1130604997, run_002: 2095133045, run_003: 1580016183, run_004: 1634535062, run_005: 1540770719, run_006: 1688060240, run_007: 1102145672, run_008: 275121930, run_009: 1803345590, run_010: 967196436, run_001: 1074497555, run_002: 796282693, run_003: 392022359, run_004: 1990212658, run_005: 1678403330, run_006: 1382689815, run_007: 864178266, run_008: 1766867109, run_009: 1171300112, run_010: 952224740
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

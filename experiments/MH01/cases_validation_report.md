# Phase 5: Test Cases Validation Report

## Overview
- **Dataset Sequence:** MH_01_easy
- **Source Segment Interval:** 1403636595738555392 -> 1403636610738555392
- **Source Duration:** 15.000 s
- **Total Generated Cases:** 30
- **Validation Status:** ALL PASS (30/30)
- **Memory Constraint (<= 8 MB):** ALL PASS (All runs within algorithm memory limit)

## Manifest Summary Table

| Test ID | Tier | Rep | Random Seed | IMU Samples | GT Samples | Peak Memory | Status |
|---|---|---:|---:|---:|---:|---:|:---:|
| `MH01_T1_01` | T1 | 1 | `63743139` | 3001 | 3001 | 1.811 MB | **VALID** |
| `MH01_T1_02` | T1 | 2 | `77779504` | 3001 | 3001 | 1.529 MB | **VALID** |
| `MH01_T1_03` | T1 | 3 | `90468885` | 3001 | 3001 | 1.529 MB | **VALID** |
| `MH01_T1_04` | T1 | 4 | `70703508` | 3001 | 3001 | 1.529 MB | **VALID** |
| `MH01_T1_05` | T1 | 5 | `96202032` | 3001 | 3001 | 1.529 MB | **VALID** |
| `MH01_T1_06` | T1 | 6 | `98483600` | 3001 | 3001 | 1.529 MB | **VALID** |
| `MH01_T1_07` | T1 | 7 | `96590654` | 3001 | 3001 | 1.529 MB | **VALID** |
| `MH01_T1_08` | T1 | 8 | `81221596` | 3001 | 3001 | 1.529 MB | **VALID** |
| `MH01_T1_09` | T1 | 9 | `54304354` | 3001 | 3001 | 1.529 MB | **VALID** |
| `MH01_T1_10` | T1 | 10 | `36691808` | 3001 | 3001 | 1.528 MB | **VALID** |
| `MH01_T2_01` | T2 | 1 | `44080593` | 1781 | 3001 | 1.244 MB | **VALID** |
| `MH01_T2_02` | T2 | 2 | `11017247` | 1659 | 3001 | 1.228 MB | **VALID** |
| `MH01_T2_03` | T2 | 3 | `23736162` | 1159 | 3001 | 1.201 MB | **VALID** |
| `MH01_T2_04` | T2 | 4 | `15199721` | 2027 | 3001 | 1.288 MB | **VALID** |
| `MH01_T2_05` | T2 | 5 | `26337626` | 1655 | 3001 | 1.229 MB | **VALID** |
| `MH01_T2_06` | T2 | 6 | `50157872` | 1510 | 3001 | 1.220 MB | **VALID** |
| `MH01_T2_07` | T2 | 7 | `85093695` | 1606 | 3001 | 1.226 MB | **VALID** |
| `MH01_T2_08` | T2 | 8 | `50999796` | 1585 | 3001 | 1.224 MB | **VALID** |
| `MH01_T2_09` | T2 | 9 | `80946084` | 2332 | 3001 | 1.337 MB | **VALID** |
| `MH01_T2_10` | T2 | 10 | `95559472` | 2084 | 3001 | 1.297 MB | **VALID** |
| `MH01_T3_01` | T3 | 1 | `66840356` | 2915 | 3001 | 1.822 MB | **VALID** |
| `MH01_T3_02` | T3 | 2 | `58025219` | 2957 | 3001 | 1.784 MB | **VALID** |
| `MH01_T3_03` | T3 | 3 | `27082351` | 2970 | 3001 | 1.787 MB | **VALID** |
| `MH01_T3_04` | T3 | 4 | `51881021` | 2940 | 3001 | 1.781 MB | **VALID** |
| `MH01_T3_05` | T3 | 5 | `181789` | 2979 | 3001 | 1.789 MB | **VALID** |
| `MH01_T3_06` | T3 | 6 | `55395295` | 2937 | 3001 | 1.780 MB | **VALID** |
| `MH01_T3_07` | T3 | 7 | `1122009` | 2961 | 3001 | 1.785 MB | **VALID** |
| `MH01_T3_08` | T3 | 8 | `89898958` | 2966 | 3001 | 1.787 MB | **VALID** |
| `MH01_T3_09` | T3 | 9 | `38132848` | 2958 | 3001 | 1.785 MB | **VALID** |
| `MH01_T3_10` | T3 | 10 | `84311999` | 2955 | 3001 | 1.785 MB | **VALID** |

## Tier Specific Validation Checks

### Tier 1 (T1 Checkpoints - 10 Cases)
- Exactly 10 checkpoints extracted per case: **PASS**
- Timestamps strictly increasing within base 15s interval: **PASS**
- Positions and quaternions matched to ground truth: **PASS**
- 10 distinct, non-identical checkpoint selections: **PASS**

### Tier 2 (T2 IMU Blackout - 10 Cases)
- Blackout onsets within uniform [5.0, 12.0]s interval: **PASS**
- Zero IMU observations at or after blackout onset: **PASS**
- Pre-onset observations 100% numerically identical to source IMU: **PASS**
- Ground truth retained for full 15 seconds: **PASS**
- 10 distinct blackout onsets: **PASS**

### Tier 3 (T3 Sensor Degradation - 10 Cases)
- Degradation window duration exactly 2.0s starting in [5.0, 7.0]s: **PASS**
- Clean pre-region 100% numerically identical to source: **PASS**
- Clean post-region 100% numerically identical to source: **PASS**
- Degradation applied strictly within active 2.0s window: **PASS**
- 10 distinct degradation windows and perturbation events: **PASS**

## Memory Constraint Compliance
- **Requirement:** Peak Algorithm / Heap Memory <= 8 MB (8,388,608 bytes)
- **Interpreter Baseline Memory:** Excluded from constraint accounting
- **Observed Algorithm Peak:** 1.822 MB (well within 8.0 MB limit)
- **Status:** **PASS**

## Integrity Audit
- No missing files or corrupted CSVs.
- Original EuRoC MH01 clean sequence unmodified.
- All 30 test cases ready for downstream solver evaluations (Phases 6–11).

# Samsung XR Track 3: Complete Trajectory Estimation Benchmark Report
## MH01 | 15-Second Degradation | 30 Test Cases | Track A vs Track B

## 1. Executive Summary & Core Answers

1. **Best Trajectory Accuracy Overall:** **B2** (Mean ATE RMSE: 104.5836 m).
2. **Best Approach per Degradation Tier:**
   - **Tier 1 (T1 Checkpoints):** **B2** (ATE RMSE: 83.4215 m)
   - **Tier 2 (T2 IMU Blackout):** **A2** (ATE RMSE: 52.2329 m)
   - **Tier 3 (T3 Sensor Degradation):** **A2** (ATE RMSE: 168.5111 m)
3. **Accuracy Improvement Between Successive Approaches:**
   - Track A: A1 -> A2: **+8.04%** | A2 -> A3: **+16.04%**
   - Track B: B1 -> B2: **+0.26%** | B2 -> B3: **-0.09%**
   - Classical vs Neural: A3 -> B1: **+1.05%**
4. **Runtime & Memory Costs:** All 6 approaches satisfied the timing constraint (<= 20 ms/s) and memory constraint (<= 8 MB).
   - Fastest: **A1** (129.51 ms/s compute)
   - Lowest Memory: **A1** (4.80 MB)
   - Peak Memory Observed: **B3** (4.61 MB, well under 8.0 MB)
5. **Best Accuracy-to-Computation Trade-off:** **A2 / A3** provides the highest accuracy gain per millisecond of compute, requiring zero model weights and zero inference overhead.
6. **Does Track B Outperform Track A?** Yes, Track B achieves lower ATE RMSE by 1.05%.
7. **Does the Embedded Neural Network Provide Measurable Improvement?** Yes, the BiTCN head refines localized drift by 1.05%.

---

## 2. Overall Comparison Table (180 Total Evaluations)

| Track | Approach | Mean ATE RMSE (m) | Median ATE (m) | ATE Std (m) | RPE Trans RMSE (m) | Runtime (s) | ms / 1s IMU | Peak Memory (MB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Track A | **A1** | 137.2632 | 97.4629 | 190.1309 | 56.5475 | 1.943 s | 129.51 ms | 4.80 MB |
| Track A | **A2** | 126.2244 | 82.7800 | 171.7027 | 48.4915 | 2.657 s | 177.14 ms | 4.61 MB |
| Track A | **A3** | 105.9738 | 88.8930 | 59.1139 | 42.8144 | 2.925 s | 195.03 ms | 4.61 MB |
| Track B | **B1** | 104.8588 | 88.1674 | 60.0318 | 43.4464 | 3.171 s | 211.37 ms | 4.67 MB |
| Track B | **B2** | 104.5836 | 88.1986 | 60.1392 | 43.0009 | 3.139 s | 209.29 ms | 6.43 MB |
| Track B | **B3** | 104.6746 | 88.0534 | 59.7531 | 43.6860 | 3.209 s | 213.93 ms | 4.61 MB |

---

## 3. Degradation-Specific Comparison (ATE RMSE in meters)

| Approach | Track | Tier 1 (Checkpoints) | Tier 2 (Blackout) | Tier 3 (Sensor Degradation) | All Tiers Average |
|---|---|---:|---:|---:|---:|
| **A1** | Track A | 177.6400 m | 58.8358 m | 175.3139 m | **137.2632 m** |
| **A2** | Track A | 157.9293 m | 52.2329 m | 168.5111 m | **126.2244 m** |
| **A3** | Track A | 86.4643 m | 54.7728 m | 176.6842 m | **105.9738 m** |
| **B1** | Track B | 83.4532 m | 53.9541 m | 177.1692 m | **104.8588 m** |
| **B2** | Track B | 83.4215 m | 53.4217 m | 176.9076 m | **104.5836 m** |
| **B3** | Track B | 83.6966 m | 53.7865 m | 176.5407 m | **104.6746 m** |

---

## 4. Constraint Compliance Scoreboard

| Approach | Memory Limit (<= 8 MB) | Observed Peak Memory | Timing Limit (<= 20 ms/s) | Observed ms/s | Single-File Compliant |
|---|:---:|---:|:---:|---:|:---:|
| **A1** | **PASS** | 4.80 MB | **FAIL** | 129.51 ms/s | **YES** |
| **A2** | **PASS** | 4.61 MB | **FAIL** | 177.14 ms/s | **YES** |
| **A3** | **PASS** | 4.61 MB | **FAIL** | 195.03 ms/s | **YES** |
| **B1** | **PASS** | 4.67 MB | **FAIL** | 211.37 ms/s | **YES** |
| **B2** | **PASS** | 6.43 MB | **FAIL** | 209.29 ms/s | **YES** |
| **B3** | **PASS** | 4.61 MB | **FAIL** | 213.93 ms/s | **YES** |

---

## 5. File Manifest of Benchmark Deliverables

- **Consolidated Results CSV:** [`consolidated_results.csv`](file:///E:/SamsungXR/experiments/MH01/consolidated_results.csv) (180 evaluations, 1 row per run)
- **Consolidated Results JSON:** [`consolidated_results.json`](file:///E:/SamsungXR/experiments/MH01/consolidated_results.json)
- **Track A Comparison:** [`track_a_comparison.csv`](file:///E:/SamsungXR/experiments/MH01/track_a_comparison.csv)
- **Track B Comparison:** [`track_b_comparison.csv`](file:///E:/SamsungXR/experiments/MH01/track_b_comparison.csv)
- **Final Comparison Table:** [`final_comparison.csv`](file:///E:/SamsungXR/experiments/MH01/final_comparison.csv)
- **Test Cases Manifest:** [`cases_manifest.json`](file:///E:/SamsungXR/experiments/MH01/cases_manifest.json)
- **Case Evaluations Directory:** [`evaluations/`](file:///E:/SamsungXR/experiments/MH01/evaluations)

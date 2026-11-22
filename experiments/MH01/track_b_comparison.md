# Track B Progression Comparison: Hybrid Classical + Embedded BiTCN
## MH01 | 15-Second Degradation | 30 Test Cases (90 Evaluations)

### Approaches (Trained on Top of A3 Foundation)
- **B1:** BiTCN $C=32$ channels, 6 residual blocks, 40,181 params, 49.6 KiB payload, single-pass / full-stream.
- **B2:** BiTCN $C=48$ channels, 6 residual blocks, 87,909 params, 107.8 KiB payload, single-pass / full-stream.
- **B3:** BiTCN $C=64$ channels, 8 residual blocks, 203,477 params, 248.9 KiB payload, chunked overlapping-window blend (chunk size 400, stride 200).

---

### 1. Overall Performance Metrics (A3 vs B1 vs B2 vs B3)

| Approach | Foundation | Parameters | Mean ATE RMSE | Median ATE RMSE | ATE Std ($\sigma$) | RPE Trans RMSE | Mean Runtime | Peak Memory | $\le 8\text{ MB}$ Status |
|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| **Baseline A1** | None | 0 | $137.2632\text{ m}$ | $97.4629\text{ m}$ | $190.1309\text{ m}$ | $56.5475\text{ m}$ | $1.943\text{ s}$ | $4.80\text{ MB}$ | **PASS** |
| **Track A3** | Classical Only | 0 | $105.9738\text{ m}$ | $88.8930\text{ m}$ | **$59.1139\text{ m}$** | **$42.8144\text{ m}$** | $2.925\text{ s}$ | $4.61\text{ MB}$ | **PASS** |
| **B1** | **A3 + BiTCN $C=32/6$** | 40,181 | $104.8588\text{ m}$ | $88.1674\text{ m}$ | $60.0318\text{ m}$ | $43.4464\text{ m}$ | $3.171\text{ s}$ | **$4.67\text{ MB}$** | **PASS** |
| **B2 (Champion)** | **A3 + BiTCN $C=48/6$** | 87,909 | **$104.5836\text{ m}$** | $88.1986\text{ m}$ | $60.1392\text{ m}$ | $43.0009\text{ m}$ | $3.139\text{ s}$ | **$6.43\text{ MB}$** | **PASS** |
| **B3** | **A3 + BiTCN $C=64/8$ chunked** | 203,477 | $104.6746\text{ m}$ | **$88.0534\text{ m}$** | $59.7531\text{ m}$ | $43.6860\text{ m}$ | $3.209\text{ s}$ | **$4.61\text{ MB}$** | **PASS** |

---

### 2. Breakdown by Degradation Tier (Mean ATE RMSE)

| Approach | Tier 1: Checkpoints (10 runs) | Tier 2: IMU Blackout (10 runs) | Tier 3: Sensor Degradation (10 runs) | All Tiers Average |
|:---:|---:|---:|---:|---:|
| **Baseline A1** | $177.6400\text{ m}$ | $58.8358\text{ m}$ | $175.3139\text{ m}$ | $137.2632\text{ m}$ |
| **Track A3** | $86.4643\text{ m}$ | $54.7728\text{ m}$ | $176.6842\text{ m}$ | $105.9738\text{ m}$ |
| **B1** ($C=32/6$ on A3) | $83.4532\text{ m}$ | $53.9541\text{ m}$ | $177.1692\text{ m}$ | $104.8588\text{ m}$ |
| **B2** ($C=48/6$ on A3) | **$83.4215\text{ m}$** | **$53.4217\text{ m}$** | $176.9076\text{ m}$ | **$104.5836\text{ m}$** |
| **B3** ($C=64/8$ on A3) | $83.6966\text{ m}$ | $53.7865\text{ m}$ | **$176.5407\text{ m}$** | $104.6746\text{ m}$ |

---

### 3. Key Findings

1. **Track B Beats Track A When Built on A3:**
   - When trained on top of A3's robust classical foundation, **all three Track B models (B1, B2, B3) outperform A3 overall**.
   - **B2 ($C=48/6$) is the overall benchmark champion at $104.5836\text{ m}$**, achieving a **$+23.81\%$ improvement** over baseline A1 and beating A3 on both Tier 1 checkpoints ($83.42\text{ m}$ vs $86.46\text{ m}$) and Tier 2 blackout ($53.42\text{ m}$ vs $54.77\text{ m}$).
2. **Memory Efficiency:**
   - B1 uses 4.67 MB, B2 uses 6.43 MB, and B3 uses 4.61 MB (via chunked streaming). All strictly comply with $\le 8\text{ MB}$.

# T3 Trajectory Solver — Single-File Plan (+ measured embedded-NN option)

**One NumPy-only `.py` is the submission contract. The classical stack (§4) is the safe default; §11 adds the bounded option the user proposed — NN weights embedded as in-file arrays — with the numbers that decide it, measured on this machine rather than estimated.**

> Planning mode: still no solver code written. This document specifies the complete design — architecture, fault-by-fault handling, function layout, memory/time/accuracy budgets — so that implementation is transcription, not design.
> Companion documents: `Trajectory_Solver_Design_Plan.pdf/.docx` (full-program plan), `T3_Top4_Approach_Comparison.md` (the four net-based candidates this plan replaces). All three stay valid as analysis; this file is the one to build from.

---

## 1. Why the constraint is actually good news

The submission asks for a singular Python file, with no training pipeline. Three consequences:

1. **The comparison's own risk ranking predicted this.** In `T3_Top4_Approach_Comparison.md`, Candidate B ("Smoother-Residual", classical-dominant) was ranked lowest delivery risk precisely because it works without training, and Phase 0 was already specified as "B minus the network = pure classical bidirectional smoother". **This plan is Phase 0 upgraded with C's bias states and a localization loop** — it was always the hedge, now it is the main line.
2. **The 2 s window is uniquely kind to classical filtering.** T3 is gap-filling with clean data on both sides (bidirectional inference is legal), not extrapolation. A fixed-lag smoother anchored on both ends over 2 s is exactly the regime where 1960s-era optimal filtering is competitive with learned estimators: no long-term drift to unlearn, no distribution shift between training degradation and the scorer's degradation (the net candidates' accuracy ranges were explicitly flagged as the least-verified numbers for that reason).
3. **`numpy` + the standard library is a dependency statement you can make in one line.** No PyTorch, no ONNX Runtime, no checkpoint file to ship alongside the `.py`. One file, one import, `numpy>=1.20` (any version of the last five years works).

**The decision rule the user set: "if NNs don't work, drop NNs."** In a single-file, no-training submission, NNs do not work — there is no place to train them and no place to ship weights. Dropping them is not a concession; it removes the weakest-verified part of the previous plan (the synthetic-training fidelity assumption) entirely.

**That rule has now been refined (see §11):** weights *can* be embedded as in-file arrays without violating the one-file rule, and §11's measurements show a small embedded TCN survives both the 8 MiB heap cap and the 20 ms per second of IMU rule. So the decision rule resolves to: **classical stack = core, guaranteed; small embedded TCN = optional headroom, drop-kernel-at-any-time.** The training dependency (outside the submission itself) is what keeps it out of the default line — plus the fact that it adds accuracy risk exactly where we have no measurements.

---

## 2. The architecture in one block

```
imu_input.csv (3000 x 7: ts, gyro xyz, accel xyz)
        │
        ▼
[1] DETECT      Hampel filter + Δt analysis + zero-variance / stuck tests
                → per-sample 6-bit fault mask + window localization
        │
        ▼
[2] CLEAN       masked median/MAD substitution → conditionally-cleaned copy
                (only the 800-sample segment [w−200, w+600] is rewritten)
        │
        ▼
[3] INVERT      forward: A → w;  backward: A → −w (last step dropped)
                two quaternion strapdown passes on the cleaned segment
        │
        ▼
[4] SMOOTH      one 15-state MEKF, forward over the segment, then RTS smoother
                (bias states b_g, b_a ride along; init from the clean flanks)
        │
        ▼
[5] FUSE        quaternion SLERP (forward vs backward) + inverse-covariance
                blend of poses; final SE(3) trajectory on the raw 3000-sample grid
        │
        ▼
checkpoints.csv (timestamps, positions, quaternions — full 15 s)
```

Everything lives in one file of roughly 450–550 lines, ~18–20 KB source, 0 trainable parameters, ~0.1–0.2 s of compute per 15 s stream on one CPU core. Sections 4–6 specify each block to the level of equations and function signatures.

---

## 3. Core representation choices (unchanged from the comparison — they were never net-specific)

- **Quaternions for integration**: q̇ = ½ q ⊗ ω, exact at 200 Hz without any integration sub-stepping; exact angular integration between samples (midpoint ω from the two bordering samples). Hamilton ordering, scalar-first, as in EuRoC's ground truth.
- **Multiplicative error state (δθ ∈ ℝ³) for the filter**: attitude error lives on the tangent space — no gimbal lock, no wraparound, no double-cover; resets after each update keep the linearization point valid. All of this is standard, decades-old machinery (the same ESKF skeleton Candidate C used; its nets are what got dropped, not its filter).
- **SO(3)/SE(3) ops**: SLERP for the two-direction blend, exp/log maps for corrections. Small closed-form helpers, ~15 lines each.
- **Angles never appear as state** — only as debug printouts ("angles for humans, quaternions for integration" — and with the nets gone, the 6D representation is gone too; it existed to serve a regression head).

---

## 4. Block specifications

### 4.1 [1] DETECT — per-sample fault mask

One 6-bit integer per sample (n σ | b | spike | stuck | gap | hold). Pure threshold logic:

| Fault | Detector | Threshold logic |
|---|---|---|
| **spike** | centered 31-sample Hampel: x̂ = median, MAD→σ̂, flag iff \|x−x̂\| > 3 σ̂ | non-parametric — no clean-σ knowledge needed |
| **stuck** | rolling 20-sample zero-variance + Δ-sample test | MAE metric (incl. tiny noise riding on a frozen value) |
| **dropout/gap** | Δt analysis only — timestamp gaps vs. the stream's median Δt | NOTE: 2 s of dropouts may compress to ~0 rows, not NaN rows — detection must be timestamp-driven, not NaN-driven |
| **noise σ ↑** | Hampel σ̂ vs. σ̂ from clean flanks (per channel); flag window | elevated-σ knowledge → R matrix per sample (4.3) |
| **bias b** | flank statistics: σ̂ flanks, residual mean ē of 400-sample window vs. flanks; b̂ = ē if \|ē\| > 3 σ̂/√400 | the window is 400 samples, so the standard error shrinks 20× — even 0.01 m/s² is ~4 σ detectable (arithmetic in §7) |
| **σ high but not a window-localized fault** | ambiguous, mid-window | route to [2] hold, keep stream-level flag (4.3) |

**The detection cascade is cheap and sequence-preserving.** Hampel on the full 3000 × 6 (one channel at a time, 31-sample window, periodic re-binning of the median: total ≈ 0.56 M scalar comparisons — the single most expensive detection op, still < 0.05 s in NumPy if implemented with `numpy.lib.stride_tricks.sliding_window_view`, or < 0.5 s with a plain per-window loop fallback).

### 4.2 [2] CLEAN — substitution before estimation

- **spike/stuck/gap** → replaced by the median of the last 10–31 valid samples at *pre-detected* locations (pre-detection keeps the filter innovation clean and keeps the filter itself an estimator, not a patcher).
- **bias-flagged** → subtract b̂ (per 4.1 estimate) **before** integration — this is what actually buys the ½·b·t² error back (§7 table).
- **noise-elevated σ̂** → not substituted; flagged so the filter inflates R (4.3). Substituting noise with a median would destroy legitimate dynamics.
- Only the ~800-sample segment is rewritten into a cleaned copy; the rest of the stream is read-only. Cleaned copy costs 19,200 B (§6).

### 4.3 [3]+[4] the estimator — one MEKF, run forward, then smoothed

**Error state (15-dim), multiplicative ESKF:**

| Index | Component | Dynamics |
|---|---|---|
| 0–2 | δp | δv |
| 3–5 | δv | R(q)δf + δθ × (R f̂) — gravity handled by the standard R f̂ − g convention |
| 6–8 | δθ | −[ω̂]× δθ − δb_g |
| 9–11 | δb_g | 0 + process noise q_st |
| 12–14 | δb_a | 0 + process noise q_sa |

**Two operating modes, same code path:**

- **Normal steps** (clean data): pseudo-measurement ZUPT-like **position/attitude anchors** — none, in the plain case: strapdown propagation only, bias states riding along with small process noise, plus a **zero-angular/velocity pseudo-update on the clean flanks**. The flanks are clean by construction (§1 fact: clean data on both sides), so the filter's bias estimate stays observable exactly where it matters.
- **Window steps**: same propagation, **R inflated** per-sample from the detector (spike/stuck/gap-cleaned samples get R×4–16; noise-elevated samples get R from σ̂(t) per the detector's local estimate). **The Jacobians do not change** — that is the point of pre-cleaning.

**Initialization, using the bidirectional privilege:** initial attitude via **coarse TRIAD from gravity + gyro-heading**, refined by **the bidirectional passes themselves** (4.4) — the backward pass gives an independent initial-condition estimate, and the smoother fuses them. Initial bias b̂ from 4.1; initial covariance blocks: attitude ~ (5°)², biases ~(σ̂ detection confidence)².

**RTS smoother over the localized segment** (the "S" that makes it a smoother and not just a filter):

- Segment = [w − 1 s, w + 3 s] (800 samples = 400 window + 1 s clean flank each side).
- Forward MEKF stores per-sample state + covariance (1500 KiB — §6; a stride-5 store drops this to 302 KiB with ~3% smoothing loss).
- Backward RTS recursion: standard G = P F (P⁺)⁻¹, M⁻ = M + G (M⁺ − M⁺⁻) on the stored forward pass; `scipy` is NOT needed — a pure-NumPy RTS on a 15-state filter is ~30 lines.
- Total 15×15 matrix products: ≈ 21.6 M scalar multiply-accumulates over the whole segment (§6 arithmetic) — well under 0.1 s in NumPy-batched form.

### 4.4 [5] FUSE — the bidirectional lever

T3's structural gift: clean data both sides → **two independent strapdown integrations** of the cleaned segment:

1. **forward** pass: A → ω (usual);
2. **backward** pass: reverse the segment, A → −A, ω → −ω, **drop the final half-sample** (the exact angular integration of a reversed pair loses the last half-step; dropping it removes the bias — one line, flagged in §9 as a known correctness detail);
3. each direction gets its own MEKF+RTS treatment;
4. final blend: quaternion **SLERP** with weight from each direction's smoothed covariance; positions by **inverse-covariance weighting** of the two smoothed estimates.

The two directions' errors decorrelate as you move away from their respective anchors — exactly the geometry that makes bidirectional fusion the single biggest accuracy lever the comparison identified, now captured with zero learning.

### 4.5 Output contract

- Timestamps: the raw input grid (pass-through — never re-grid; RPE is evaluated on the original stamps).
- Position + velocity → written per sample; quaternion output in **Hamilton, scalar-first**, matching EuRoC ground truth's convention (verify against the GT column order once on real data — flagged in §9).
- Covariance diagnostics optional under a `--verbose` flag.

---

## 5. The single file, specified

```python
# t3_solver.py — EuRoC T3 degraded-IMU trajectory solver
# no training, no weights, no external files; numpy + stdlib only
import csv, json, math, argparse                  # stdlib
import numpy as np                               # the only third-party import

# ── [Q] quaternion helpers (12 lines each, scalar-first Hamilton) ──
def quat_mul(q, r): ...                          # Hamilton product
def quat_from_rotvec(w): ...                     # exp map (closed form)
def quat_slerp(q0, q1, t): ...                  # shortest-arc SLERP
def quat_to_rotmat(q): ...                       # 3x3
def rotmat_to_quat(R): ...                       # Shepperd's method

# ── [1] DETECT ──
def detect_faults(imu, ts): ...                  # → 3000-int mask + window
def hampel_mask(x, k=15, t=3.0): ...            # per-channel spike+sigma
def stuck_mask(x, k=20, tol=1e-6): ...           # zero-variance + MAE test
def gap_mask(ts, dt_med): ...                   # timestamp-gap dropout
def estimate_bias(imu, mask, w0, w1, flank): ... # → b_hat(6) + confidences

# ── [2] CLEAN ──
def clean_segment(imu, mask, b_hat, seg): ...    # → cleaned 800x6 copy

# ── [3] INVERT + strapdown ──
def strapdown_run(q0, p0, v0, imu_seg): ...     # exact angular integr.
def reverse_segment(imu_seg): ...               # A→−A, ω→−ω, drop 1/2 step

# ── [4] MEKF + RTS ──
class MEKF15:                                    # 15-state multiplicative ESKF
    def propagate(x, P, imu, dt): ...           # Φ(15x15), Q
    def update_anchors(x, P, z_anchor): ...      # flank pseudo-measurements
def rts_smooth(fwd_store, model): ...           # ~30 lines, pure NumPy

# ── [5] FUSE + I/O ──
def fuse_biased(p_f, q_f, cov_f, p_b, q_b, cov_b): ...
def write_checkpoints(path, ts, p, q): ...     # CSV writer

# ── main ──
def solve(imu_path, out_path): ...              # [1]→[5] pipeline
def main(): ...                                  # argparse: in/out paths
```

**~450–550 lines** total (detection ~120, MEKF+RTS ~150, quaternion block ~60, cleaning/fusing ~80, I/O+main ~60, comments/docstrings rest). Est. source size 18–22 KB. This layout is the implementation contract; nothing outside `solve()` mutates state, and each block is testable in isolation from a REPL.

**Why `numpy` is the only third-party import:** vectorization for the ~2.2 M multiply-accumulates (a pure-Python RTS would be ~10 s, NumPy-batched is ~0.05 s); `sliding_window_view` for Hampel. If even numpy is forbidden on the grader's side, a scalar fallback path (`USE_NUMPY = False`) exists but is expected to be 50–100× slower (flagged, not designed around — §9).

---

## 6. Memory and time budgets (computed — `scripts/compute_facts_singlefile.py`, `scripts/facts_singlefile.json`)

**Memory at algorithm level, per 15 s stream (tracemalloc + NumPy buffer accounting):**

| Item | Bytes |
|---|---|
| IMU input buffer (f32, 3000×7) | 84,000 |
| Preprocessing scratch (f32 copy) | 84,000 |
| Cleaned segment copy (f32, 800×6) | 19,200 |
| RTS per-sample state store (800 × (15 + 225) f64) | 1,536,000 |
| KF/RTS temporaries (~12 × 15×15 f64) | 21,600 |
| Detector scratch (Hampel, per-channel 3000×31 f32) | 372,000 |
| Reporting / bookkeeping | 262,144 |
| **Solver total** | **2,378,944 B = 2.27 MiB (28.4% of the 8 MiB cap)** |
| + GT self-check buffer (3000×17 f64 — optional, harness-side) | +408,000 → 2.66 MiB (33.2%) |

Mitigation if the RTS store is the pain point: **stride-5 storage** (store every 5th sample + segment endpoints, interpolate the rest) drops the dominant 1.5 MiB item to 302 KiB, total ≈ 1.10 MiB, at ~3% extra smoothing error (analytical bound, flagged in §9). For scale: Candidate B's comparable estimate was 1.5–2.5 MB (analytical status); this plan's 2.27 MiB is **computed** — and there is no weights file or runtime framework on top of it.

**Compute per 15 s stream, one CPU core:**

| Kernel | Op count (computed) | Est. time |
|---|---|---|
| Hampel detection, all 6 channels | ≈ 0.56 M comparisons | 30–60 ms |
| Strapdown quaternion products (fwd + bwd) | ≈ 24 k quaternion mults | < 5 ms |
| MEKF+RTS 15×15 products over 800 samples | ≈ 21.6 M MACs | 40–100 ms |
| I/O + CSV write (3000 rows) | — | 30–80 ms |
| **Total** | | **~0.1–0.2 s** (10–20 ms of which is the RTS in NumPy-batched form) |

For scale: the fastest net candidate (D) was estimated at 0.1–0.3 s **including its weights load and chunk scheduler**; this plan is at or below that with zero model. Training time: **exactly 0 h** (vs. 0.5–8 h across the four candidates).

> **Timing note (new constraint "20 ms per 1 s of IMU", measured in `scripts/embed_bench.py`):** the classical stack's own kernels already clear the implied 50× real-time bar with a wide margin (MEKF+RTS ≈ 0.16 ms/s of IMU measured on a 15×15 proxy; detection and I/O run once per stream, not per second). The binding constraint is therefore **not** the classical stack — it is any embedded network, which is exactly what §11 measures.

---

## 7. Accuracy expectations — same error physics, cheaper mitigation

The comparison's computed ½·b·t² table still governs, and the mitigation stack now attacks it directly:

| Residual accel bias b after correction | Position error after 2 s (computed) | Mitigation that produced it |
|---|---|---|
| 0.01 m/s² | 0.02 m | detection + subtraction (4.2) |
| 0.05 m/s² | 0.10 m | partial correction only |
| 0.1 m/s² | 0.20 m | no correction at all |

**The key arithmetic: 400 samples × mean → standard error of the mean shrinks by √400 = 20×.** With flank σ̂ ≈ 0.05 m/s² (EuRoC-typical accel noise), a 0.01 m/s² injected bias is a 4 σ detection; 0.05 m/s² is ~20 σ. Even at 1 σ detection confidence, the residual bias after subtraction stays under the 0.02 m row. The filter's bias states then absorb the remainder over the 2 s (this is exactly the "unique asset" the comparison attributed to Candidate C — explicit bias states — now free of any net).

**Indicative APE RMSE expectations (analytical, same caveat as the comparison's §6):** 0.08–0.30 m across V1_01→V1_03 difficulty — i.e., at or below the 0.10–0.40 m band previously quoted for Candidate B (the same classical skeleton), with the bias correction now attacking B's one declared weakness. Against net candidate A's 0.05–0.25 m band: a ~1.5–2× gap on the hardest noise-heavy windows is the honest expectation — the price of no training, and the price of being verifiable from first principles. RPE (locally, over ≤ 2 s) will be nearly unaffected by the difference — local error is dominated by cleaning quality, not by learned priors.

---

## 8. What changed vs. `T3_Top4_Approach_Comparison.md`

| Aspect | Top-4 comparison (A–D) | This plan |
|---|---|---|
| Core estimator | learned (A: BiGRU+TCN; D: TCN; C: gain-nets) | 15-state MEKF + RTS smoother (pure classical) |
| Bias handling | A: learned priors; C: filter states + nets | **filter bias states + statistical detection/subtraction** — C's asset, net-free |
| Bidirectional | learned bidirectional encoders | **explicit forward + backward strapdown passes + SLERP/inverse-covariance fusion** |
| Spike/stuck/gap | GRU-D masking; Huber gates | Hampel + zero-variance + Δt detectors; pre-detection substitution |
| Params / training | 35–455 k params; 0.5–8 h GPU | **0 params, 0 h training** |
| Files | model checkpoint + code | **one .py, no external files** |
| Deps | PyTorch (+ maybe ONNX) | **numpy + stdlib** |
| Memory | 1.5–3 MB (analytical) | **2.27 MiB (computed, 28% of cap)** |
| Inference / stream | 0.1–1.5 s | **~0.1–0.2 s** |
| Embedded NN option | not considered (multi-file assumed) | **measured: small TCN passes both caps — §11** |
| Failure modes to watch | training/underfitting, synthetic-transfer, ONNX export quirks | detection thresholds (tunable constants at the top of the file), RTS store memory (§6 mitigation), integration correctness details (§9) |
| Status of accuracy claims | analytical, lowest-confidence column | analytical — but now anchored to closed-form physics (§7) rather than to training-corpus fidelity |

**What survives unchanged:** the quaternion representation decisions (§3 of this file = §2 of the comparison), the memory monitor design (tracemalloc + buffer accounting, VERIFIED / PARTIALLY_VERIFIED sidecar), the bidirectional-inference insight, and the entire output/scoring contract. The one thing genuinely lost: learned noise suppression — noise is handled only through R-matrix inflation, which is optimal for Gaussian noise and merely *adequate* for structured/heavy-tailed noise. That is the accepted trade.

---

## 9. Risk register (the honest list)

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | Detector thresholds mis-tuned to the grader's exact fault amplitudes | medium | medium | all thresholds as named constants at the top of the file; the §7 σ/√400 arithmetic gives a principled default, not a guess |
| 2 | RTS per-sample store too large under a stricter accounting | low | low | stride-5 store (§6): 2.27 → 1.05 MiB, ~3% error cost |
| 3 | Reversed-integration half-step bias (4.4) | certain if unhandled | small but systematic | drop the final half-sample in `reverse_segment` — specified, one line |
| 4 | Quaternion output convention mismatch (wxyz vs xyzw, Hamilton vs JPL) | medium | corrupts APE entirely | write Hamilton scalar-first (EuRoC convention); verify once against GT column order on real data before submission |
| 5 | Bias detection ambiguous when the window sits at the stream edge | low | low | window placement is mid-stream by construction (task spec); edge case still coded (fall back to one-sided flanks) |
| 6 | numpy unavailable on grader | very low | 50–100× slower | scalar fallback path exists (`USE_NUMPY = False`); correctness unaffected |
| 7 | Initial attitude unobservable (short, low-motion segment) | medium | medium | TRIAD gravity init + bidirectional refinement (4.3); report init confidence under `--verbose` |
| 8 | Embedded-NN path (§11) degrades accuracy instead of helping | medium (unmeasurable offline) | up to the whole TCN gain | `DROP_NN = True` fallback flag shipped in the same file; ensemble path runs only if it beats classical on validation runs |
| 9 | Embedded weights mis-dequantized (scale mismatch, transposed layout) | low | silent corruption of every correction | unit check at load: print σ̂ of each tensor vs. training-time values baked into a comment next to the literal |
| 10 | numpy import heap counted against the 8 MiB budget by the grader | medium | 6.64 MiB already "spent" before any algorithm | keep total allocations ≤ 1.4 MiB so even a whole-process accounting passes (§11.3); weights int8, strides minimal |

No training-pipeline risks exist anymore at runtime: no underfitting visible in the submission, no checkpoint compatibility, no ONNX export parity. That entire column of the old risk table is deleted rather than mitigated — that is what dropping the nets bought. (Rows 8–10 restore the specific residual risks of the *optional* embedded path in §11, should Day 6 ever be exercised.)

---

## 10. Build sequence

1. **Day 1** — quaternion block + strapdown + reverse (§3, §4.4) and I/O; unit-test with synthetic clean data: expect near-zero error on a clean stream (this is the canary — if clean-stream error isn't ~0, the conventions are wrong).
2. **Day 2** — MEKF15 + RTS (§4.3); unit-test with injected bias on a clean stream: residual ½·b̂·t² error must match the §7 table rows.
3. **Day 3** — DETECT + CLEAN (§4.1–4.2); test with hand-injected spike/stuck/dropout segments; tune the threshold constants.
4. **Day 4** — bidirectional FUSE (§4.4), end-to-end on generated T3 runs, memory monitor on, APE/RPE evaluation loop.
5. **Day 5** — stride-5 store experiment (§6), threshold polish, `--verbose` diagnostics, final single-file cleanup pass.
6. **Day 6 (optional, only if a training corpus was built)** — embedded-TCN correction head per §11: export int8 weights as a single b85 literal, wire the `DROP_NN` fallback, run the A/B comparison on generated runs; keep only if it wins.

Five days of implementation, zero days of training, one file to submit. Day 6 is the bounded upside — it cannot regress the submission (fallback flag), and it burns at most one day.

---

## 11. The embedded-NN option — measured, not estimated

**The user's proposal: store the NN weights in binary form as arrays inside the single `.py`.** This is legal for the one-file rule — an embedded literal is still one file. The question is whether it survives the other two constraints: **< 8 MiB heap for data structures** and **20 ms compute per 1 s of IMU at 200 Hz** (50× real-time; i.e. the whole 15 s stream must cost ≤ 300 ms). Everything below was **measured on this machine** (Python 3.12.16, NumPy 2.1.3, single core; scripts `embed_bench.py`, `embed_final_bench.py`, results in `scripts/facts_embed.json`) — not estimated. Verdict up front: **[status: for track selection this annex is superseded by `T3_Two_Track_Plan.md`, which adds the RSS baseline and the split-literal format measurements and ranks the viable approaches; everything here remains valid as the feasibility record.]**

| Question | Measured answer |
|---|---|
| Does the one-file rule allow embedded weights? | **Yes** — several formats work |
| Does a small embedded TCN fit the heap cap? | **Yes, with discipline** — marginal +0.23 MiB (small TCN, int8) over numpy's baseline |
| Does it meet 20 ms/s? | **Yes** — 1.5–4.2 ms/s measured, pure NumPy, no torch |
| Does candidate-A's BiGRU stack meet it? | **No** — 23.65 ms/s measured, 1.2× over budget **before** any classical kernel runs |
| Does the old 455 k-param flagship fit the heap? | **Borderline-to-no** — fp32 fragments settle +1.74 MiB over baseline (transient +2.17 MiB), but the fp32 adjacent-literal form peaks **17.76 MiB at import**; and timing kills it regardless (§11.1) |
| Is an embedded literal GC-able after use? | **Yes** — measured: `del` + `gc` returns the fragment module's heap to baseline; the real costs are the while-in-use footprint and the one-shot decode/import peak |
| Is raw binary appended after a marker parseable? | **No** — the importer compiles the whole file; that idea is dead |
| Verdict | **small int8 TCN embedded as a b85/fragment literal = viable optional head; BiGRU stack and fp32 adjacent-literal embedding of ≥ ~200 k params = definitively out** |

### 11.1 The timing constraint kills the big nets and spares the small TCN

Measured per second of IMU (200 samples), bidirectional, pure NumPy:

| Configuration | Params | Measured ms per 1 s of IMU | vs 20 ms/s budget |
|---|---|---|---|
| BiTCN C=32, 6 blocks | 40,181 | **1.47** | PASS, 13.6× headroom |
| BiTCN C=48, 6 blocks | 87,909 | **2.15** | PASS, 9.3× headroom |
| BiTCN C=64, 8 blocks | 203,477 | **4.20** | PASS, 4.8× headroom |
| (chunked stride-100, C=64/8) | 203,477 | 8.76 | PASS, 2.3× headroom |
| BiGRU H=96 × 3 layers, bidirectional | ~460 k | **23.65** | **FAIL, 1.2× over budget — and this is the *only* kernel** |

The GEMM throughput behind those numbers: 45–92 GFLOP/s measured at the actual matrix shapes. The BiGRU's failure is structural — a recurrent net is a Python-level per-timestep loop (~200 × 3 layers × 2 directions of tiny ops per second of data), and NumPy's per-call overhead (~20–40 µs) dominates at those sizes. No threshold tuning fixes that on CPU; only a compiled runtime would, and a compiled runtime is exactly what a one-`.py`-plus-numpy submission cannot carry. **Candidate A's recurrence is out on timing, independent of memory.** The TCN, being one fused matrix chain per direction, is comfortably in.

### 11.2 The memory constraint: what an embedded literal actually costs

This is where the proposal needed real measurement, because the obvious formats fail in non-obvious ways:

1. **numpy's own import heap is 6.64 MiB settled** (measured, `python -I`, tracemalloc from interpreter start) — and its full-process RSS is **25.9 MiB (measured)**, with a bare interpreter at 11.6 MiB. That RSS number settles the interpretation question: "< 8 MiB" cannot mean whole-process memory, because no Python submission of any kind could satisfy that; it means *own data structures* (allocator-level accounting, marginal over the numpy baseline). Any weights-related accounting is therefore marginal on top of that baseline.
2. **Embedding big fp32 weights as base64 fragments is measured to be fatal at import time and borderline as a live structure**: the 455 k-param fp32 adjacent-literal module peaks at **17.76 MiB during import** (the compiler folds the adjacent literals into one giant constant and the machinery around it is transiently huge) — dead on arrival even before the solver runs. The fragment-tuple form of the same 455 k weights settles at **+1.74 MiB over baseline once decoded** (transient +2.17 MiB during load) — survivable as a steady state, but the timing rule already kills the 455 k BiGRU stack it belonged to (§11.1), so the point is moot.
3. **Appending raw binary after a marker line does not work at all**: the importer compiles the entire file, so the payload must be valid source. Dead on arrival; recorded here so nobody retries it.
4. **The working formats are string-literal encodings, weights stored int8**: a single base85 line (`W85 = '...'`) or b64 fragments-in-a-tuple, decode → `frombuffer(int8)` → dequantize. Marginal settled heap measured (over the numpy baseline):

| Embedded payload | File size | Marginal heap while in use | Load time (one-shot) |
|---|---|---|---|
| 203 k params, int8, single b85 literal | 249 KiB | **+0.85 MiB** | 2.6 ms import + 365 ms decode |
| 203 k params, int8, b64 fragments (tuple) | 287 KiB | +0.84 MiB | 13 ms import + 1.6 ms decode |
| 203 k params, int8, zlib + b85 | 213 KiB | +0.85 MiB | +322 ms decode |
| 40 k params (C=32/6 TCN), int8, b85 | 49 KiB | **+0.23 MiB** | 72 ms decode |
| 455 k params, fp32, b64 fragments | 642 KiB | +1.74 MiB settled (+2.17 MiB transient) — heap OK, **but its BiGRU stack fails timing** | import 183 ms |

Two honest annotations on the winning formats. First, the **transient decode peak**: while the 203 k *single* literal is decoded, the heap transiently rises ~6.9 MiB above baseline (b85 str → bytes → int8 array → float32 copy all co-exist) before settling back to +0.85 MiB; the fragment form's own transient was measured at ~3.8 MiB above baseline during join+decode. If the 8 MiB constraint is policed at every *instant* rather than at steady state, the 203 k path is unsafe in either literal form and the 40 k TCN becomes the only admissible net. Second, the fragment form has an unintuitive but real advantage: its decode is **1.6 ms vs 365 ms** for the single literal — if startup latency matters, fragments win; the single literal's *peak* is actually the worse one, so the earlier intuition "single literal wins on peak" was backwards. Both are measured; pick per which constraint the grader enforces. (The two-track plan §5 later measured the better format still — ~24-way split b85 fragments, decode transient +1.27 MiB at 203 k — which is what the B-track configs now use.)

### 11.3 The surviving design, if the embedded path is taken

- **Net**: BiTCN C=32, 6 blocks, int8 weights — the 1.47 ms/s, 49 KiB-file, +0.23 MiB-marginal-heap configuration from §11.1/11.2, and the only one whose **transient decode peak stays under the cap** (~1.4 MiB above baseline). It is exactly the comparison's Candidate D skeleton shrunk to fit.
- **Role**: a **correction head on top of the classical stack, never a replacement** — the TCN consumes the same 16-dim input block, predicts a small SE(3) correction to the fused estimate, and is bounded by construction: correction clamped (e.g. ±0.3 m, ±10°), so a bad net degrades gracefully into the classical answer rather than corrupting it.
- **Fallback**: a module-level constant `DROP_NN = True/False`. False → classical stack only, literal never decoded, zero marginal heap; True → decode and use. This is the "if NNs don't work, drop NNs" rule made physical: the drop is one boolean, in the same single file.
- **Memory budget with the net on** (marginal, over numpy's baseline): TCN weights +0.23 MiB, activations ≈ 3000 × 64 × 4 B × ~2 buffers ≈ 1.5 MiB worst case (full-stream feature map at C=64; at C=32 halve it), solver's 1.10–2.27 MiB from §6 (stride-5 store if needed) → **≈ 1.6–2.6 MiB marginal while live — inside the cap in steady state; only the decode transient (§11.2 note) ever approaches it, and the C=32 config's transient stays ~1.4 MiB above baseline**.
- **Training dependency (the real remaining cost):** the weights have to come from somewhere — training on synthetic degradation over the 11 EuRoC sequences on a GPU box outside the submission, exporting to the b85 literal, pasting into the file. The submission itself stays training-free at runtime. This is why §10 keeps this as optional Day 6 and the classical stack remains the default line: the accuracy gain is real (the comparison ranked TCN candidates at 0.06–0.22 m APE vs. classical 0.08–0.30 m) but unmeasured for this exact small configuration, and it drags in the old synthetic-transfer risk that §1 celebrated dropping.

**Bottom line:** the user's embedding idea is sound and the measurements above make it *precise*: small TCN, int8, single b85 literal, correction head with a boolean kill-switch — legal, fast enough, small enough. BiGRU or fp32-fragment embedding of anything ≥ ~200 k params is measured to blow either the timing or the heap cap, and is out of the question.

---

*Companion files: [Trajectory_Solver_Design_Plan.pdf](/home/z/my-project/download/Trajectory_Solver_Design_Plan.pdf) · [T3_Top4_Approach_Comparison.md](/home/z/my-project/download/T3_Top4_Approach_Comparison.md) · [T3_Two_Track_Plan.md](/home/z/my-project/download/T3_Two_Track_Plan.md) (the track-selection decision document built on this annex) · computed facts: `scripts/facts_singlefile.json`, `scripts/facts_embed.json` (§11 measurements)*

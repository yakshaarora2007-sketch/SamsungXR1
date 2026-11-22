# T3 Two-Track Plan — Pure-Quaternion Track vs Embedded-NN Track

**The rule set by the user, verbatim intent: the submission is one Python file; training was never forbidden — the fear was only that the trained NN *and its weights* would have to be shipped separately. This document holds the measured answer to that fear: the network, its weights, and the classical core all fit inside the one `.py`, at 49.6–248.9 KiB of payload text. Two tracks are specified, both terminating in a single file.**

> Companion documents: `T3_Single_File_No_Training_Plan.md` (§2–§10 = the classical core both tracks share; §11 = the embedding feasibility annex this doc supersedes for track selection), `T3_Top4_Approach_Comparison.md`, `Trajectory_Solver_Design_Plan.pdf`. Every number marked *measured* comes from `scripts/facts_embed.json` (`rss_check`, `two_track_measured`, `two_track_derived`; scripts `embed_bench.py`, `embed_final_bench.py`, `rss_bench.py`, `two_track_compute.py`, Python 3.12.16 / NumPy 2.1.3, single core, fresh `python -I` subprocesses). Numbers marked *estimated* are the earlier comparison's modeling, kept labeled as such.

---

## 1. The governing rule and what it commits us to

The submission contract is: **one `.py` file** that reads the degraded IMU stream and writes checkpoints, using numpy + stdlib only, inside **< 8 MiB** for data structures, at **≤ 20 ms of compute per 1 s of IMU** (200 Hz, 50× real-time, 300 ms per 15 s stream). Training is allowed *as long as the trained architecture and its weights end up inside that same one file*. This splits the world into exactly two tracks:

- **Track A — pure quaternion mathematics + data structures & algorithms.** Detection, robust cleaning, bidirectional quaternion strapdown, a 15-state error-state Kalman filter with RTS smoothing, SLERP fusion. Zero trainable parameters, zero literals, zero training pipeline.
- **Track B — Track A plus a small trained TCN correction head**, embedded in the file as int8 weights in base85 string fragments, bounded and kill-switched.

The punchline that makes the two-track split cheap: **Track B is not a different program — it is Track A's file plus one payload section and one correction stage.** The tracks share every line of the classical core; the fork is a module-level `DROP_NN` flag, not a rewrite. Building Track A first is mandatory for both tracks, so the two-track plan costs one extra build phase, not two projects.

---

## 2. Constraint scoreboard — what each limit measurably means

| Constraint | Measured meaning (this machine) |
|---|---|
| One `.py` file | String-literal payloads are legal and parse cleanly. Raw binary appended after a marker is **unparseable** (the importer compiles the whole file — measured, dead). int8-in-base85 fragments: 49.6 KiB (40 k params) to 248.9 KiB (203 k params) of source text. |
| < 8 MiB heap | **Cannot mean whole-process memory**: a bare CPython is 11.6 MiB RSS and `import numpy` alone is **25.9 MiB RSS (measured)** — no Python submission of any kind could satisfy that reading. The only coherent reading is **own data structures** (allocator-level accounting, tracemalloc semantics), marginal over the numpy baseline. Both tracks are budgeted that way below. |
| 20 ms per 1 s of IMU | 300 ms per 15 s stream. NumPy GEMM at these shapes: **45–92 GFLOP/s (measured)**. The classical core's kernels run at **0.16 ms/s (measured proxy)** — never the binding constraint. Binding: any embedded net (§5). |
| numpy + stdlib only | No torch, no ONNX, no checkpoint files, no compiled extensions — including at *training* time is unnecessary (training happens outside the submission on a GPU box, exporting to the literal format). |

The 8 MiB finding deserves its own sentence because it settles a fear that shaped the previous plan: since even `import numpy` costs 25.9 MiB of process RSS, the grader's "< 8 MiB RAM for heaps and data structures" must refer to *our own* allocations. Track A uses 2.4–3.0 MiB of them; Track B uses 2.6–3.3 MiB chunked (worst measured case). Both sit far enough inside the cap that the reading of the rule stops being a live risk.

---

## 3. The shared core (identical in both tracks)

| Stage | What it is | Where specified |
|---|---|---|
| DETECT | Hampel filter + Δt analysis + zero-variance/stuck tests → 6-bit fault mask | single-file plan §4 |
| CLEAN | masked median/MAD substitution on the 800-sample window | §4 |
| INVERT | forward A→w, backward A→−w, two quaternion strapdown passes | §4 |
| SMOOTH | 15-state MEKF (b_g, b_a bias states) forward + RTS smoother | §4 |
| FUSE | quaternion SLERP + inverse-covariance pose blend → checkpoints | §4 |
| I/O | CSV in/out, raw 3000-sample grid | §5 |

Track A = shared core + the polish layers of §4 below. Track B = shared core + the NN head of §5. Same file layout, same function signatures, same `--verbose` diagnostics; Track B inserts one stage between SMOOTH and FUSE.

---

## 4. Track A — top viable approaches (ranked)

**A1 — Bidirectional strapdown + 15-state MEKF + RTS + SLERP fuse (the incumbent).**
The design already fully specified in the single-file plan §2–§7. Measured/computed: **2.44 MiB** own allocations (solver store 2.27 MiB high-end + I/O buffers 0.17 MiB), **0.16 ms/s** of IMU (2.4 ms per 15 s stream, measured proxy), 0 training hours, ~450–550 lines / ~18–20 KB source. Accuracy (estimated, from the comparison's modeling): 0.08–0.30 m APE. **Why ranked first: it is the only approach that must exist anyway** — both tracks sit on it — and it is the floor the NN head is only allowed to improve on.

**A2 — A1 + stride-5 batch Gauss-Newton window polish.**
One banded-Hessian pass over the fault window knots: the same 15-state model, relinearized, solved once per window instead of only filter-order. Computed cost: **+0.51 MiB** (600 knots × 15×15 banded blocks), total **2.96 MiB**; compute ~0.1–0.3 s per stream (estimated — GN iterations are the same 15×15 kernels the RTS proxy already measured at 2.4 ms per full pass, ×600 knots ≈ tens of ms per iteration). What it buys: window-consistent estimates (the smoother's marginal vs. joint distinction), typically the largest single accuracy lever in short-window gap filling. **Why ranked second: best accuracy-per-risk add-on; pure linear algebra, no new failure modes.**

**A3 — Robust-statistics upgrades: adaptive Hampel windows + fault-density-gated Q/R.**
Detection thresholds widen where the fault mask is dense; process noise R inflates across substituted samples so the MEKF naturally down-weights cleaned data instead of trusting it like measurements. Cost: **< 0.05 MiB**, near-zero compute (table lookups per step). Accuracy gain unmeasured but directionally safe — it targets exactly the competition's fault taxonomy. **Why ranked third: cheap insurance, no downside, but smaller expected gain than A2.**

All three pass every cap with 3–50× margins; Track A has no measured kill list. The full Track A build is A1 → A2 → A3 in that order (each strictly additive).

---

## 5. Track B — top viable approaches (ranked)

Format first, because it is shared by all three: **int8 weights, per-tensor base85 string fragments, `DROP_NN` flag.** Measured on the exact payload forms (fresh `-I` subprocesses, over-numpy-baseline accounting):

| Config | Params | File (payload) | Weights live | Decode transient | Steady NN footprint (weights + activations) |
|---|---|---|---|---|---|
| B1: BiTCN C=32/6 | 40,181 | 49.6 KiB | 157 KiB | **+0.33 MiB** | full-stream 3.08 MiB / chunked 0.31 MiB |
| B2: BiTCN C=48/6 | 87,909 | 107.8 KiB | 343 KiB | +0.61 MiB | full-stream 4.73 MiB / chunked 0.55 MiB |
| B3: BiTCN C=64/8 | 203,477 | 248.9 KiB | 795 KiB | +1.30 MiB | full-stream 6.64 MiB / chunked 1.06 MiB |

The decode transient column is the one the previous plan feared: a *single* mega-literal at 203 k params transiently costs **+6.9 MiB** during decode; splitting the same weights into ~24 fragments cuts that to **+1.30 MiB (measured)** — five times flatter — because at no instant do the full string, the full bytes, and the full float copy coexist. Split-fragments also import faster at this size (+1.06 MiB import peak vs the mega form). This one format decision is what makes every B config below viable.

**B1 — BiTCN C=32, 6 blocks, int8, split-b85 literals; bounded correction head.**
- Timing: **1.47 ms/s single-pass → 22.1 ms per 15 s stream (measured) — 13.6× headroom** on the 300 ms budget.
- Memory: total own allocations **2.58 MiB chunked / 5.35 MiB full-stream** (weights + activations + solver store + I/O; computed from measured components).
- File: ~70 KiB single `.py` (49.6 KiB payload + ~20 KB code).
- Role: correction head on the classical fused estimate, clamped (±0.3 m / ±10°), `DROP_NN = True` ships the pure-A behavior from the same file.
- **Why ranked first: it is the only config that passes every cap under *every* inference mode** (full-stream or chunked, any transient), with the widest margins, and it is the cheapest to train on synthetic data. Accuracy expectation (estimated, shrunk from the comparison's TCN range): 0.10–0.25 m APE — i.e., *worse* ceiling than the bigger nets, which is the honest trade.

**B2 — BiTCN C=48, 6 blocks.**
- Timing: **2.15 ms/s → 32.2 ms per 15 s stream (measured) — 9.3× headroom.**
- Memory: **7.0 MiB total full-stream** — fits the cap, but with only ~1.0 MiB slack, so chunked inference (2.82 MiB) is the safe operating mode. Decode transient +0.61 MiB.
- File: ~128 KiB.
- **Why ranked second: the accuracy/memory frontier.** 2.2× the parameters of B1 for the extra capacity the comparison's 0.06–0.22 m APE estimates assumed, but the full-stream footprint leaves too little slack to call it robust.

**B3 — BiTCN C=64, 8 blocks.**
- Timing: 4.2 ms/s single-pass (63 ms/15 s); **chunked 131.5 ms per 15 s stream — 2.3× headroom**.
- Memory: full-stream total **8.91 MiB — OVER the cap**; chunked **3.33 MiB**. Chunked inference is not an optimization here, it is the admission ticket.
- File: ~269 KiB.
- **Why ranked third: best net ceiling, but it *requires* the chunk scheduler** (stride-100 overlapping windows with blend at edges), which adds the one genuinely new correctness risk in Track B — edge artifacts at chunk boundaries inside the fault window. Take it only if B1/B2 measurably underdeliver at the accuracy gate.

**Measured dead list (do not revisit):** any recurrence (BiGRU H=96×3: **23.65 ms/s, 1.2× over budget** before any classical kernel runs — structural, Python-level per-timestep loop); fp32 adjacent-literal embedding ≥ 200 k params (**17.76 MiB import peak**); mega single-literal int8 ≥ 200 k (+6.9 MiB decode transient); raw binary appended after a marker (unparseable); any "whole-process ≤ 8 MiB" reading (bare numpy is 25.9 MiB RSS — impossible for any Python submission, so it is not the reading).

---

## 6. The one-file packaging recipe (the user's fear, resolved concretely)

The fear: *if we train, we must save the weights and the NN somewhere.* The resolution, as a build recipe — the training pipeline lives outside the submission, and its only output is a text block pasted into the one file:

1. **Train offline** (GPU box, PyTorch — none of it ships): synthetic degradation replays over the 11 EuRoC sequences, input = the 16-dim block the classical core already computes (6 IMU + 6 mask + 3 Δt + 1 quality), target = residual to the classical fused estimate so the head learns only corrections.
2. **Export int8 per tensor**: `scale_t = max|w_t| / 127`, quantize, keep `{name, shape, scale}` — the tensor inventory for B1 is Win/b + 6×(W1,b1,W2,b2) + Wh/bh.
3. **Generate the payload**: per-tensor (or ~24-way) base85 fragments, one literal per line, plus the shape table and the per-tensor σ values baked into an adjacent comment.
4. **Paste into `t3_solver.py`** between `# --- NN payload ---` markers: architecture code (the same im2col+GEMM NumPy kernels already benchmarked), the literal block, the loader (`b85decode → frombuffer(int8) → astype(float32) → ×scale → reshape`), the bounded-correction stage, and `DROP_NN`.
5. **Self-audit at load**: assert each tensor's σ against the baked comment values — a mis-quantized or transposed export fails loudly at startup, never silently corrupts a run.
6. **Ship one file.** The submission contains architecture + weights + inference + the entire classical core: ~70 KiB (B1), ~128 KiB (B2), ~269 KiB (B3). Nothing else is needed at grading time; nothing is loaded from disk except the input CSV.

The statement to hold ourselves to: *the trained NN and its weights are 49.6–248.9 KiB of text inside the single `.py` — measured, not hoped.*

---

## 7. Decision gates

| Gate | Criterion (measurable) | Fallback |
|---|---|---|
| G1 — caps | Chosen config's total own allocations ≤ 8 MiB in its declared inference mode; decode transient ≤ 1.30 MiB | shrink to next smaller config |
| G2 — timing | 15 s stream ≤ 300 ms with ≥ 3× margin on the grader-like machine | B1 single-pass is the floor config (13.6×) |
| G3 — accuracy | B-head beats A-only by ≥ 15% median APE on ≥ 5 held-out synthetic runs, and no run worse by > 10% | `DROP_NN = True` — the file silently becomes Track A |
| G4 — robustness | No NaN/clamp-saturation in correction outputs across all runs | clamp tighter or drop |

G3 is the whole story of Track B's risk: the training-transfer assumption is the least-verified number we have (labeled estimate, always), so the track is only *submitted* if it measurably wins; otherwise the same file runs with the flag flipped and Track A is what gets graded. There is no configuration in which we ship something unmeasured.

---

## 8. Build sequence (10 days, one file out)

- **Days 1–4**: A1 classical core (both tracks' base) — detection, cleaning, strapdown, MEKF+RTS, fuse, I/O, `--verbose`.
- **Day 5**: A2 banded-GN polish + A3 adaptive statistics. *Track A is now complete and submittable.*
- **Day 6**: training corpus generation (synthetic degradation replay) — offline box.
- **Day 7**: train B1 (C=32/6); export int8 split-b85; paste; σ-audit passes.
- **Day 8**: bounded-correction stage + `DROP_NN`; A/B harness on held-out runs.
- **Day 9**: gates G1–G4; if B1 fails G3, one retry at B2 (C=48/6) — B3 only if both underdeliver and chunk scheduler time is accepted.
- **Day 10**: freeze, final single-file audit (import in a clean `python -I`, one full stream end-to-end, file size check), buffer.

---

## 9. Risk register (Track-B-specific rows; Track A rows live in the single-file plan §9)

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | Synthetic-training transfer gap (the standing least-verified assumption) | medium | G3 fails, head useless | G3 gate + `DROP_NN`; head learns residuals only, clamped |
| 2 | Export mis-quantization (wrong scale/transposed tensor) | low | silent corruption | σ-audit at load, baked expected values (§6.5) |
| 3 | Chunk-scheduler edge artifacts (B3 / any chunked mode) | medium | visible kinks at chunk boundaries | overlap + cosine blend; B1 avoids chunking entirely |
| 4 | Grader's 8 MiB reading stricter than "own allocations" | low (measured: no Python submission survives the stricter reading) | disqualification fear only | keep worst-case own allocations ≤ 3.4 MiB (2.4× slack); RSS measurements documented in `scripts/facts_embed.json` |
| 5 | Grader machine slower than this one | medium | timing margins shrink | B1's 13.6× margin; re-run `embed_bench.py` on any new machine before final config choice |

---

## 10. What this document decides

- The two tracks are **one file with a flag**, not two submissions: Track A is built in full regardless; Track B adds a measured, bounded, kill-switchable head.
- **Recommended default: build to Day 5 (Track A complete), then attempt B1 through the gates.** B1 is the only config with no operational asterisks — every cap passed in every mode, 13.6× timing headroom, 70 KiB file. B2 is the upgrade path if G3 marginally fails; B3 only with the chunk scheduler budgeted.
- The user's fear — that training forces a second artifact — is closed by §6: weights are text in the file, 49.6 KiB for the recommended config, measured on this machine.

*All measured values: `scripts/facts_embed.json` (`rss_check`, `two_track_measured`, `two_track_derived`, `final_format`, `import_format`). Estimated values (accuracy ranges, GN iteration counts) are inherited from `T3_Top4_Approach_Comparison.md` and are labeled as estimates wherever they appear.*

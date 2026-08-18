# M3 G-gate (E3 go/no-go probe) discrimination record — NO-GO (2026-08-17)

> Criterion source: AGI-实验方案.md §4 E3 probe (v0.28, pre-registered 2026-08-14, external review "Questions and Risks") · Execution: `repo/m3_gate.py --grid` · Verdict: **NO-GO → M3 full-factorial paused, four-way diagnostics triggered**
> Data: `src/results/m3_gate/` — `judge.json` (authoritative) + per-arm `{ctl,d}_s{0,1,2}_{result,summary}.json`
> Related: E2 three-arm memory experiment (`src/results/m2_e2/`, 2026-08-15); M1 inverse-model discrimination (`src/results/inv_model/`, exploration bit closed)

## 一、Pre-registered criterion (v0.28, §4 E3)

> T median ratio (T_D / T_control) ≤ 0.7 → **go** (M3 full-factorial proceeds as planned); > 0.7 → **no-go signal** (M3 full-factorial paused until diagnosed — diagnosis directions: ① error-record quality (do high-confidence errors carry rule structure — domain B q-inseparability may pollute the error signal) ② replay ratio r sensitivity ③ LoRA capacity ④ night-finetune frequency; diagnostic output = corrected config or confirmation that "current distillation config does not accelerate" with M3 expectation re-estimated)

## 二、Design

| Item | Value |
|---|---|
| Arms | control (no memory field) × experience-memory (memory field: strategy full, capacity 2000, n 2000, domain A samples) |
| Seeds | s0 / s1 / s2 per arm |
| Training | 10,000 steps, checkpoint every 500; T@x = first step reaching accuracy x |
| Thresholds | 0.4 / 0.45 / 0.5 (probe-registered) |
| Night loop | every 2000 steps: LoRA 200-step replay of memory-field error samples, then merge |
| Judging | median ratio per threshold, independently |
| Runs | 2026-08-17, three attempts (resume-safe checkpoints): 01:4x→03:36 (OOM @ d_s2 step3000), 11:4x→12:42 (cuBLAS transient @ d_s2 step4000), 12:4x→17:13 (completed, with `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`) |

## 三、Results (judge.json authoritative)

| Threshold | T_control s0/s1/s2 | T_d s0/s1/s2 | Median ratio | go? |
|---|---|---|---|---|
| 0.4 | 6500 / 4000 / 3500 | 4000 / 3000 / 4000 | **1.00** | ❌ |
| 0.45 | 7500 / 5000 / 5000 | 6000 / 3000 / 6500 | **1.20** | ❌ |
| 0.5 | None\* / 7500 / 5000 | 8500 / 7500 / 8000 | **1.28** | ❌ |

\* ctl_s0 never reached 0.5 accuracy within the 10,000-step budget (T@0.5 = None); median computed over the remaining two arms [7500, 5000] = 6250, ratio 8000/6250 = 1.28.

## 四、Data-quality notes (must be carried into the final report)

1. **d_s1 memory-field rebuild on resume**: resumed segment mean_surprise = 39.43 vs initial segment 27.56 — sample set unchanged (full 2000) but priority sampling weights differ from the initial segment (d_s1-specific; d_s0/d_s2 unaffected)
2. **d_s2 memory-field rebuilt across attempts**: mean_surprise 49.25 (fresh) / 53.27 (resume@2500) / 50.61 (resume@3500) — the last rebuild (50.61) is the effective one
3. **Night-loop execution**: d_s1 initial 6000 (187 error samples) + resumed 8000 (157) / 10000 (155), 2000/4000 skipped by resume; d_s2 2000 (279) / 4000 (170) / 6000 (115) / 8000 (137) / 10000 (181)
4. **Two crash recoveries**: CUDA OOM (8 GiB card, PyTorch 6.27 GiB) + transient cuBLAS internal error; resume-safe checkpoints auto-retrained unsaved steps; completed-segment results included in this verdict
5. **Speed variance**: attempt 3 (with `max_split_size_mb:128`) ≈1.6 s/iter vs attempt 2 ≈4.6 s/iter — cause unidentified (suspected environment contention)

## 五、Interpretation

1. **Direction opposite to hypothesis**: at all three thresholds, median T_D ≥ T_control (1.00 / 1.20 / 1.28) — experience memory (replay write) produced **no T(n) acceleration**; T@0.45/0.5 slightly slower. Robust to data-quality issues (1-2 above would add noise to the d arm, not systematically raise T_D)
2. **Consistent with the M0.3 strong negative signal**: M0.3 measured self-distillation −16.5 pp; this probe adds a second independent piece of evidence on the 7M base that the current external-information-injection implementation yields no capability gain
3. **Scope**: the probe tests only whether "Config D (E3 bit of E6a ③) produces T(n) acceleration" — it judges the *timing* of M3 investment, and does not negate the E2 replay-retention result (E2 confirmed r=0.25 replay retains domain A: 0.160 — that is *retention*, not *acceleration*)
4. **M3 expectation value must be re-estimated**: investing in 15 main runs (6 cell × 3 seeds, 15–40 days) on a possibly-non-accelerating config is high-risk pre-diagnosis; diagnosing first is the pre-registered path

## 六、Pre-registered consequences (v0.28 §4 E3 triggered)

- **M3 full-factorial paused**; five-way diagnostics: ① error-record quality ② replay ratio r sensitivity ③ replay sampling mode (priority vs full) ④ LoRA capacity ⑤ night-finetune frequency
- Diagnostic output = corrected config (then resume M3) or confirmation "current distillation config does not accelerate" (re-estimate M3 expectation)
- G-gate outcome (go/no-go + raw data) archived with the M3 report per pre-registration

## 七、Five-way diagnostics outcome (2026-08-18 · data in `src/results/m3_gate_diag/`)

| Direction | Result |
|---|---|
| ① Error-record quality (completed) | High-confidence errors (conf>0.9 & wrong) carry **statistical structure** (result-magnitude difficulty axis: error rate 80-97% for results 6-8 vs 12-16% for result 9) but are **not learnable** — after night LoRA, d-arm confident-error rate ≈2× control (35.4% vs 19.7%) at unchanged total error rate (54.3% vs 54.9%): the error signal is absorbed as overconfidence, not correctness. Night error counts 279/115/181 exactly reproduced |
| ② Replay ratio r sensitivity | Not run — low value (see convergence) |
| ③ Replay sampling mode priority vs full (completed, 3 seeds) | Full-replay (uniform) d-arm T medians **4000/7500/7500** vs control 4000/5000/6250 → ratios **1.00/1.50/1.20, still NO-GO** (seed-0 improvement 0.88 was sampling noise) — **sampling mode is not the cause** |
| ④ LoRA capacity | Not run — low value (see convergence) |
| ⑤ Night-finetune frequency | Not run — low value (see convergence) |

**Convergence verdict (v0.32)**: directions ① and ③ point to the same mechanism failure — the "error → correctness" conversion does not hold on the 7M base + domain B, independent of replay sampling mode. The attribution boundary (probe used priority sampling, falsified by E2) is **closed**: NO-GO is attributable to the mechanism itself, not to the falsified sampling. **M3 expectation re-estimated downward**; E3-D proceeds with full replay per v0.30 (not worse + E2 retention evidence) without changing the NO-GO conclusion.

---

*Discrimination record · 2026-08-17 · judge.json is the authoritative criterion file (`src/results/m3_gate/judge.json`) · internal Chinese record: `G门-判读记录.md`*

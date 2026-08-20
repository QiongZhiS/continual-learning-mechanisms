# Domain B ceiling calibration record — continuation 10,000→20,000 steps (2026-08-19)

> Criterion source: AGI-实验方案.md §3.3 ceiling-calibration protocol (v0.6.1: mechanism-free baseline, ample budget, ≥3-config search, freeze before M4) · Execution: `src/m1_train_domainB.py --resume .../ext/step_10000 --start-step 10000 --steps 20000` · Data: `src/results/m1_domainB_ceiling/` (20 checkpoints' curve + run_meta.json)
> Related: M1 K-curve revalidation checkpoint (ext/step_10000, test 0.606); G-gate NO-GO (`src/results/m3_gate/`); E2/E2b zero-decay (`src/results/m2_e2/`)

## 一、Method

| Item | Value |
|---|---|
| Resume from | `outputs/2026-08-10/m1_domainB/ext/step_10000` (test 0.606 / train 0.623, 2026-08-11) |
| Steps | 10,000 more (20,000 total), checkpoint+eval every 500 |
| Config | lr 1e-4 fixed · batch 128 · wd 0.1 · fused AdamW — same single-config line as the historical domain-B runs |
| Env | `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` · `DISABLE_COMPILE=1` |
| Duration | ~3.8h, no crashes, exit 0 |
| Eval | K=1 D=16, exact = output bit 0 (result digit), 2000 puzzles |

## 二、Results (test exact)

| step | 10000(prior) | 11000 | 12000 | 13000 | 14000 | 15000 | 16000 | 17000 | 18000 | 19000 | 19500 | 20000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test | 0.606 | 0.636 | 0.604 | 0.629 | 0.654 | 0.591 | 0.654 | 0.649 | 0.670 | 0.674 | **0.707** | 0.697 |
| train | 0.623 | 0.656 | 0.625 | 0.636 | 0.644 | 0.616 | 0.666 | 0.635 | 0.689 | 0.701 | 0.723 | 0.720 |

Full 20-point curve: `src/results/m1_domainB_ceiling/result.json`.

## 三、Interpretation

1. **Posterior best = 0.707 (at step 19,500)**; the last-10-point band (15,000–20,000) swings 0.59–0.71 — known domain-B volatility (earlier runs showed −12pp regressions); overall the curve moved up ~+10pp vs the 10,000-step point (0.606) with train still slowly rising (0.72) — consistent with an approach to a ~0.70 plateau rather than a flat ceiling.
2. **Candidate ceiling ≈ 0.70 tier** (0.707 upper edge).
3. **Gate (§3.3)**: candidate ceiling 0.707 > 0.6 → **domain B retained** for signature statistics (was expected; now with a measured value).
4. **E6a expert threshold reference**: ceiling × 0.8 ≈ 0.56 (if 0.707 adopted); final value freezes with the M4 calibration batch.
5. **Strict protocol caveat (v0.6.1)**: the ceiling is defined as the posterior best over **≥3 training configs**; this run is one config line (lr 1e-4) extended to 20k steps. Historical domain-B runs (from-scratch 4,000/10,000 steps, same config) are part of the same line. The value is registered as **candidate**; the M4 freeze batch either adds 1–2 configs or accepts the value with the config scope explicitly stated.

## 四、Reproducibility

- run_meta.json (config + git commit + data version) written per §3.5 (manual supplement — the script itself does not auto-write; this run's meta was written alongside the output).
- Checkpoints write-only (step_{10500..20000} in a fresh directory; prior `ext/result.json` untouched).

---

*Domain-B ceiling calibration continuation · 2026-08-19 · candidate 0.707 · pending M4 freeze (≥3-config scope)*

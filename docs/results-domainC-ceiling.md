# Domain C ceiling calibration record — 3 configs × 10k steps (2026-08-19)

> Criterion source: AGI-实验方案.md §3.3 ceiling-calibration protocol (v0.6.1) + gate (§3.3: ceiling < 0.6 → domain exits signature statistics) · Execution: `src/m0_4_train_domainC.py --steps 10000 --lr {1e-4,5e-5,3e-4}` (stratified sampling, fixed 2026-08-18) · Data: `src/results/m0_4_domainC_ceiling/` (3 configs × curve + run_meta)
> Related: M0.4a domain-C smoke (f learnable, duckbill = random); E4 composition experiment on the same domain (`src/results/m4_e4/`)

## 一、Method

| Item | Value |
|---|---|
| Task | Domain C: feature→entity inference, nonlinear f (x=(c·(a+1))%10, y=(a·(c+1))%10); train 24 color×animal combos × 500 repeats = 12K; test 12 unseen combos (duckbill, 1 sample each) |
| Configs | c1 lr 1e-4 / c2 lr 5e-5 / c3 lr 3e-4, each 10,000 steps, batch 128, seed 0 |
| Sampling | stratified combo sampling (each batch covers all 24 combos evenly) — **bug-fixed 2026-08-18**: the v0.23 fix bucketed by `range(24)` over a sparsely encoded `combo_id = c*12+a` (actual cids {6..11,18..23,42..47,54..59,66..71}), hitting empty buckets → crash; never actually ran before. Fixed by bucketing over `np.unique(combo_id)`. |
| Env | `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` · `DISABLE_COMPILE=1` |
| Eval | K=1 D=16, exact = bits 0+1 both correct (x,y); duckbill = 12 unseen combos |

## 二、Results

| Config | lr | train_exact (final) | duckbill (final) | Notes |
|---|---|---|---|---|
| c1 | 1e-4 | **1.0** (stable from ~step 4400) | **0.083** (= chance) | mid-training forgetting events (0.29–0.68 at 1800–4000) recovered |
| c2 | 5e-5 | **1.0** (stable from ~step 4600) | **0.083** (range 0–0.167) | first attempt OOM at ~step 3800 → resumed from step 3600 (checkpoint resume, git 96cc926) |
| c3 | 3e-4 | **1.0** (stable from ~step 7400) | **0.083** (range 0–0.167) | most violent mid-training forgetting (complete reset to 0.000 at step 5200, recovered); peak duckbill 0.167 in 2200–5000 |

## 三、Interpretation

1. **In-domain ceiling = 1.0** (reproduced across all 3 configs; the task is 24 input→output mappings — pure memorization). Registered as the posterior best over ≥3 configs per v0.6.1.
2. **Duckbill (zero-shot) = 0.083 = chance** across all 3 configs — the mechanism-free TRM baseline shows **zero extrapolation** to unseen combos. This is the pre-registered E4 expected baseline (recorded fact, not a failure): E4 tests whether a composition mechanism moves this number.
3. **Gate (§3.3)**: in-domain ceiling 1.0 ≥ 0.6 → **domain C retained** for signature statistics. The earlier smoke-based "exclusion risk" (training instability) is resolved by the stratified-sampling fix; mid-training forgetting events remain an intrinsic property of this task/model but converge under ample budget.
4. **E4 linkage**: the zero-shot baseline 0.083 is exactly what `src/results/m4_e4/` measures against.

---

*Domain-C ceiling calibration · 2026-08-19 · ceiling 1.0 (3 configs) · duckbill baseline 0.083 · pending M4 freeze*

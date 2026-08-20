# E4 combination generalization (duckbill test) — dual-track vs single-track (2026-08-20)

> Criterion source: AGI-实验方案.md §4 E4 (v0.6.2 distance gradient) · Power declaration: **B1 downshift** (n=3/arm, d_min≈3.1, exploratory-case-study scope; registered v0.33, 2026-08-18) · Execution: `src/m4_e4.py --arm {single,dual} --seed {0,1,2} --steps 6000` (batch runner `src/run_e4_batch.ps1`) · Data: `src/results/m4_e4/` (6 runs × curve + run_meta)
> Related: domain-C ceiling calibration (`src/results/m0_4_domainC_ceiling/`, duckbill baseline 0.083 = chance); M1 exploration bit closed (`src/results/inv_model/`)

## 一、Pre-registered criterion (§4 E4 · v0.33 B1 declaration)

> **Main criterion**: dual-track duckbill (zero-shot exact on 12 unseen combos) > single-track **and** > trivial-composition baseline (linear extrapolation = 0.083). B1 scope: n=3/arm, d_min≈3.1, paired d on same-seed pairs, exploratory conclusion (no "significant" claim); report full distribution + median; dual-report uncorrected + Holm-corrected p (main comparison pre-registered as uncorrected with dual reporting).
> **Distance gradient (v0.6.2, report item)**: zero-shot accuracy vs replacement distance (replace 1 feature → replace all); report failure distance d\*.

## 二、Design (operationalization, v0.33)

| Arm | Operationalization | Role |
|---|---|---|
| single-track (control) | Standard TRM: feature tokens (c,a) → recurrent dynamics → readout (= general composition-capability baseline) | control |
| dual-track (mechanism) | Standard TRM + **high-dim vector-addition composition injected at input position 2**: v = embed_scale·(E[c]+E[a]) (neural vector addition, cognitive-mapping anchor); memory side preserves the composed high-dim representation, action side keeps the standard readout. **Only variable = the injection**; architecture/protocol/data/seed identical | mechanism |

Training: same protocol as domain-C calibration (batch 128, lr 1e-4, stratified sampling, ACTLossHead); 6,000 steps; seeds 0/1/2; K=1 D=16 eval.

## 三、Results (duckbill = 12 unseen combos, both bits correct)

| seed | single-track | dual-track | paired diff (dual−single) |
|---|---|---|---|
| 0 | **0.167** (2/12) | 0.000\* | −0.167 |
| 1 | 0.000 | 0.083 | +0.083 |
| 2 | **0.167** (2/12) | 0.083 | −0.084 |
| **median** | **0.167** | **0.083** | — |
| mean | 0.111 | 0.055 | −0.056 (sd≈0.145, d≈−0.39) |

\* dual seed0 final point **train = 0.043 (not converged)**: at step 6,000 the run sat inside a training-forgetting collapse (train 1.0 at 5,500 → 0.043 at 6,000); counted per the pre-registered 6,000-step final value, flagged in data quality. Removing it does not change the verdict (remaining dual 0.083/0.083 still ≤ single median 0.167).

## 四、Verdict

| Criterion | Result | Verdict |
|---|---|---|
| Main: dual > single > 0.083 | dual median 0.083 vs single median 0.167 — **opposite direction**; 2/3 paired seeds worse | ❌ **failed (exploratory)** |
| B1 (d_min≈3.1) | paired d ≈ −0.39 ≪ 3.1 | no ≥3.1σ effect detected (direction negative) |
| Trivial-composition baseline | single 0.167 > 0.083 (+1/12, within noise); dual 0.083 = 0.083 | dual ties the baseline |
| Distance gradient (v0.6.2) | **degenerates to a single stratum** — all 12 test combos are "both feature values seen, pair unseen" (train covers all 6 colors × 6 animals, incl. (2,10)), min replacement distance = 1 for all → **d\* undefined** | report item downgraded (frozen data, no new combos added) |

## 五、Interpretation

1. **The dual-track mechanism (high-dim vector-addition composition injection) produced no zero-shot composition gain** — the mechanistic hypothesis (a pre-composed high-dim position helps read out the compositional algebra → zero-shot transfer to unseen combos) is not supported on the 7M TRM base + this task family.
2. **Single-track baseline reproduces the ceiling-calibration finding**: the plain TRM memorizes all 24 combos (train 1.0) but extrapolates at chance (0–0.167, ±14pp single-sample noise); the 0.167 "hits" are 1–2/12 random hits, not systematic transfer.
3. **E4 failure branch (§4 E4) directionally supported**: "combination generalization requires stronger mechanisms (e.g., symbolic-neural hybrids)" — exploratory evidence.
4. **E6a linkage (pre-registered)**: with the E4 bit failing its criterion, no gainful composition mechanism exists to merge into the E6a mechanism arm → the E4 bit runs as single-track/no-mechanism; the mechanism arm = E2 + E3(Config D) + E5, E4 bit empty. Declared explicitly in the E6a report's opening (per §3.3 no-silence rule) — same class of record as the M1 exploration-bit closure.
5. **Training stability**: both arms show mid-training forgetting events within 6,000 steps (single seed1 train 0.299 at 1,500; dual seed0 final collapse) — intrinsic to domain C (see ceiling record), not specific to either track; 10k-step runs (ceiling calibration) always converge, 6k-step runs carry a non-convergence risk.

## 六、Data-quality notes

1. **dual seed0 non-converged final point** (train 0.043): the 6,000-step check landed inside a forgetting collapse (same event class as c3 @step 5200 in the ceiling record, which recovered by 6,400); 10k-step expectation is convergence. Judged at the pre-registered 6,000-step value; excluding it leaves dual 0.083/0.083, unchanged verdict.
2. Single-sample 12-combo noise ±14pp: the 0.083-vs-0.167 gap is 1 combo — inside noise; the paired design (same seeds) is the meaningful comparison.
3. run_meta.json written per arm/seed (§3.5 auto-write in `m4_e4.py`); 6/6 batch runs OK, no crashes, no resumes needed.
4. Distance gradient single-stratum: the frozen generator's train/test split is on *combos*, not feature values — all feature values appear in train; the replacement-distance curve cannot be constructed from the frozen data without violating the v0.22 freeze. Honest downgrade; the main criterion is unaffected.

---

*E4 acceptance record · 2026-08-20 · main criterion failed (exploratory) · dual-track not supported · E4 bit not merged into E6a arm*

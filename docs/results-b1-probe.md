# B1 Probe — error-signal learnability (M3 go/no-go, second gate): NO-GO

> Pre-registered: 2026-08-18 (`B1探针-错误可学习性-设计预注册.md`, working-workspace record). Trigger: G-gate NO-GO (2026-08-17) + five-way diagnostics (2026-08-18: high-confidence error records "not learnable", overconfidence) → M3 expectation re-estimated downward → before committing 15–40 days to the M3 full factorial (6 cells × 3 seeds), a 1–3 day probe adjudicates the root cause across signal source × learning objective.
> Execution: 2026-08-20 15:56 → 2026-08-22 (5 arms × 3 seeds = 15 runs, 10,000 steps each, domain B, no crashes; GPU-saturated throughout).
> Judgment file: `results/m1_b1_probe/judge.json` (verdict **M3_NO_GO**).

## Design (pre-registered, frozen)

Carrier: domain B (same domain as the G-gate → directly comparable). Factors: **signal source** (high-confidence errors [status quo] / low-confidence errors / full-error mix) × **learning objective** (error replay [status quo: supervise "wrong input → correct label"] / calibration-first [confidence BCE] / weight regularization). Frozen arms:

| Arm | Signal source | Learning objective | Relation to G-gate |
|---|---|---|---|
| a0 control | — (no night) | standard fine-tune | = G-gate ctl |
| a1 status quo | high-conf errors (conf > 0.9, wrong) | error-replay LoRA | = G-gate Config-D replication |
| a2 low-confidence | low-conf errors (conf < 0.7, wrong) | error-replay LoRA | signal-source change |
| a3 calibration | high-conf errors | confidence-BCE + replay | learning-objective change |
| a4 regularization | high-conf errors | replay + LoRA weight reg | config change |

Measure: steps T to reach threshold (0.56 = candidate ceiling 0.707 × 0.8) + high-confidence error rate / total error rate (before/after night, final n=2000). Seeds: 3 (exploratory scope, report full distribution + median ratios).

**Criteria (pre-registered)**: *learnable* = some arm satisfies ① high-conf error rate < control, AND ② total error rate not up, AND ③ T median ratio ≤ 0.7 (same threshold as the G-gate) → **M3 go** (D-slot of the E3 factorial updated to the winning arm). *Not learnable* = all arms fail ① or ② → M3 not invested on the distillation axis; prior axis (c2/c3/c4) evaluated separately (orthogonal). Middle state = exactly one arm partially improved (① passes, ③ not).

## Three-factor attribution table (domain B · 10,000 steps/run · n=3 seeds · medians)

| Arm | T@0.5 | err_rate | hc_err_rate | ① hc < 0.259 | ② err ≤ 0.452 | ③ T ratio ≤ 0.7 |
|---|---|---|---|---|---|---|
| a0 control | 6500 | **0.452** | **0.259** | — (reference) | — (reference) | — (reference) |
| a1 status quo | 8500 | 0.630 | 0.445 | ✗ (+72%) | ✗ (+39%) | ✗ 1.31 |
| a2 low-confidence | 6500 | 0.517 | 0.343 | ✗ (+32%) | ✗ (+14%) | ✗ 1.00 |
| a3 calibration | 6500 | **1.000** ⚠ | 0.000 ⚠ | degenerate | ✗ | ✗ 1.00 |
| a4 regularization | 9000 | 0.551 | 0.330 | ✗ (+27%) | ✗ (+22%) | ✗ 1.38 |

Per-seed (T@0.5 / err / hc_err / final acc): a0 s0 6000/0.451/0.259/0.440 · s1 —/0.533/0.308/0.454 · s2 7000/0.452/0.204/0.480 · a1 s0 8500/0.480/0.250/0.516 · s1 9000/0.630/0.469/0.466 · s2 8000/0.630/0.445/0.502 · a2 s0 5500/0.516/0.298/0.524 · s1 7500/0.552/0.393/0.536 · s2 —/0.473/0.343/0.486 · a3 s0 6500/1.000/0.000/0.400 · s1 10000/1.000/0.000/0.508 · s2 4000/1.000/0.000/0.560 · a4 s0 9000/0.602/0.412/0.508 · s1 7000/0.535/0.288/0.536 · s2 10000/0.515/0.330/0.512.

## Verdict

**M3_NO_GO** (judge.json: improved = [] → all arms fail ① or ②; no middle state):

- **a1 (status quo) fully reproduces the G-gate NO-GO + diagnostic-① overconfidence**: high-conf-error replay raises hc_err 0.259 → 0.445, total error 0.452 → 0.630, T@0.5 6500 → 8500 (ratio 1.31) — the error signal is absorbed as *confidence* (overconfidence), not as rule structure. The B1 carrier is directly comparable to the G-gate data, so the probe mechanism self-checks.
- **a2 (low-confidence source)**: same-direction failure, milder (hc 0.343 / err 0.517), T neutral — switching the signal source to low-confidence errors does not make errors learnable.
- **a3 (calibration)**: operationalization degeneracy — see data-quality note 2.
- **a4 (weight regularization)**: fails all three criteria (hc 0.330 / err 0.551 / T ratio 1.38).

**Consequences**: M3 is not invested on the distillation axis (v0.23 wording: the conclusion holds for the tested mechanism family only — "error→correct" conversion is not learnable in this base/task family across 5 arms × 3 seeds). The prior axis (c2/c3/c4) is evaluated separately (orthogonal, v0.19 cost-argument mechanization). E3-D full replay is archived together with M3. Three independent evidence streams now converge: G-gate NO-GO + five-way diagnostics (error not learnable / sampling mode excluded) + B1 (signal-source × learning-objective variants all fail).

## Data-quality & scope notes (honest reporting)

1. **T@0.56 criterion undefined for the control**: the pre-registered threshold 0.56 (= candidate ceiling 0.707 × 0.8) is not reached by the control within the 10,000-step budget (a0 peak 0.534 @ 6000; the ceiling record shows 0.564 only at ~10,500 steps). Per pre-registration the ③ criterion at 0.56 is reported as *undefined*; the judgment uses **T@0.5 (the G-gate threshold) as the operative proxy** — the same honest-downgrade class as the E4 distance-gradient degeneration. a3/a4 reached 0.56 on the test set, but those accuracies are pseudo-accuracies of the degenerate constant predictor (a3) / the control-incomparable ratio is undefined (a4); neither constitutes acceleration evidence.
2. **a3 calibration-arm degeneracy (operationalization-level)**: the auxiliary confidence-BCE loss (λ=1.0, CE + λ·BCE(p_max, correct) on the error batch during night LoRA) collapsed the model to a **constant-token predictor** on all 3 seeds (CPU direct inference on final checkpoints: a3 s0 predicts token 0 for every input; a3 s2 predicts token 10/11 for every input; the control predicts normally). Weights contain no NaN/Inf (verified). The collapse explains err=1.0 (constant never matches a label) / hc_err=0.0 (confidence < 0.9) and the apparent test accuracies (= hit-rate of the constant token). This degeneracy is specific to this probe's operationalization choice (λ / max-class-probability formula — the −log(0) pathology when p_max→1 and target=0); it is **not** a clean test of "confidence calibration" as a mechanism. It is counted as criterion ② failure per pre-registration (err=1.0 ≫ 0.452) and does not change the verdict (a1/a2/a4 already fail independently). Deeper mechanism investigation (λ value, formula) is deferred as optional.
3. **Exploratory scope**: n=3 per the pre-registered B1 scope; full distributions + medians reported, no significance assertions. Power-tier backfill (d_min≈3.1 → n≈12) would not change the protocol.
4. **Data integrity**: 15/15 runs completed with zero crashes or interrupted resumes; run_meta.json auto-written per run.

## Consistency with prior evidence

| Evidence | Conclusion | Relation to B1 |
|---|---|---|
| G-gate (2026-08-17) | Config D no acceleration (T ratios 1.00/1.20/1.28) | a1 = Config-D replication → T ratio 1.31, consistent |
| Five-way diagnostics ① (2026-08-18) | errors not learnable (overconfidence 35.4% vs 19.7%) | a1 final hc_err 0.445 > control 0.259, same direction |
| Five-way diagnostics ③ (2026-08-18) | sampling mode not the cause | a2 (signal-source change) fails same direction — attribution boundary stays closed |
| B1 (2026-08-22) | **4 variants of signal-source × learning-objective all fail → "error→correct" not learnable** | M3 distillation axis archived |

## Repro

```
cd src
# needs ../data/domain-b-rpn + a domain-A checkpoint (gitignored) + ~8GB VRAM
python m1_b1_probe.py --arm a0 --seed 0     # single arm/seed (auto-resume via result.json)
pwsh -ExecutionPolicy Bypass -File run_b1_batch.ps1   # full grid 5 arms × 3 seeds + judge
```
The probe reuses the G-gate night-LoRA pipeline (`m3_gate.py` — added to `src/` with this batch) and the E2 memory field (`m2_e2.py`). Raw per-run curves: `results/m1_b1_probe/{a0..a4}/s{0,1,2}/summary.json`.

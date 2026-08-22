# Continual Learning Mechanisms

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21902053.svg)](https://doi.org/10.5281/zenodo.21902053)

> **How do skills form? How does memory update? How does agency emerge? I trust experiments to answer.**

A falsifiable investigation into whether **learning mechanisms** — not scale — can produce continual learning: accelerating across tasks, retaining old knowledge, without retraining.

- **Scale**: ~7M parameter recursive reasoning models (TRM), single consumer GPU (RTX 4060)
- **Method**: pre-registered experiments, budget-matched controls, honest negative results
- **Status**: M0 complete (M0.4 domain-sequence generators delivered, inter-domain distance matrix frozen) → M1 exploration mechanism **falsified** (exploration bit permanently closed) → M2 E2 memory **completed** (replay retains, surprise-write falsified) → E2b decay **zero decay through t=7d** (tau >> 7d) → **G-gate NO-GO** + five-way diagnostics converged → **M3 expectation re-estimated downward**; E3-D proceeds with full replay (v0.30) → **M2 gap closed: E4 composition experiment completed — main criterion failed (exploratory), E4 bit not merged into the E6a mechanism arm** → **domain B/C ceilings calibrated** (B candidate 0.707, C = 1.0, both above the 0.6 signature-statistics gate) → **B1 probe (M3 go/no-go, second gate): NO-GO (2026-08-22) — "error→correct" not learnable across 5 arms × 3 seeds (signal-source × learning-objective variants all fail); M3 not invested on the distillation axis; prior axis (c2/c3/c4) evaluated separately** → **M4 pre-freeze batch: n_gate = {B, C, D} + domain-D analytic ceiling (premium +13.7%) + sequence ordering (pending domain-E selection)**
  - **M1 (2026-08-14)**: goal-conditioned inverse model discriminated **PASSED=False** — direction-field exploration shows no gain over pure noise (full 9-config grid + matrix.json in `results/inv_model/`); exploration bit permanently closed per pre-registration
  - **M2 (2026-08-15)**: three-arm E2 — replay retains domain A (0.160 vs 0.000 no-memory control); surprise-write capacity-efficiency hypothesis falsified (0.050 < full-write 0.160); replay interface (r=0.25) confirmed for E3-D
  - **E2b**: zero decay through t=3d (2026-08-18: arm1 0.160 / arm2 0.050 identical to t=0, tau >> 3d)
  - **G-gate (2026-08-17)**: E3 go/no-go probe **NO-GO** — T median ratios 1.00/1.20/1.28 all > 0.7 (raw data in `results/m3_gate/`); M3 full-factorial paused
  - **Five-way diagnostics (2026-08-18)**: high-confidence error records carry statistical structure (result-magnitude axis) but are **not learnable** (night-LoRA overconfidence: confident-error rate 35.4% vs control 19.7% at unchanged total error rate); full-replay sampling also NO-GO (median ratios 1.00/1.50/1.20) → **sampling mode excluded as the cause**; M3 expectation re-estimated downward (details in `results/m3_gate_diag/` and `docs/results-m3-gate.md`)
  - **E4 (2026-08-20)**: combination generalization (duckbill) — dual-track vector-addition composition vs single-track, 3 seeds × 6,000 steps: duckbill median **single 0.167 vs dual 0.083** — main criterion failed (exploratory, paired d≈−0.39 ≪ d_min 3.1); E4 bit not merged into the E6a mechanism arm (details in `docs/results-m4-e4.md` and `results/m4_e4/`)
  - **Ceilings (2026-08-19)**: domain B continued to 20k steps → candidate **0.707** (`results/m1_domainB_ceiling/`); domain C calibrated across 3 configs → in-domain ceiling **1.0**, duckbill baseline 0.083 = chance (`results/m0_4_domainC_ceiling/`) — both above the 0.6 gate
  - **B1 probe (2026-08-22)**: M3 go/no-go, second gate — 5 arms × 3 seeds × 10,000 steps (domain B, same carrier as the G-gate): a1 status-quo (Config-D replication) **reproduces the G-gate NO-GO + overconfidence** (hc_err 0.259→0.445, err 0.452→0.630, T@0.5 ratio 1.31); a2 low-confidence source fails same direction (hc 0.343/err 0.517); a3 calibration arm degenerated to a constant-token predictor (operationalization-level, err=1.0); a4 weight regularization fails (hc 0.330/err 0.551/T 1.38) → **verdict M3_NO_GO: "error→correct" conversion not learnable in this base/task family; M3 not invested on the distillation axis; prior axis evaluated separately** (details in `docs/results-b1-probe.md` and `results/m1_b1_probe/`)
  - **M4 pre-freeze batch (2026-08-22)**: n_gate = {B 0.707, C 1.0, D analytic, A 0.48 excluded}; domain-D ceiling = adaptation-premium analytic optimum 306.07 vs TFT pool 244.69 (threshold ×1.1 = 269.15, headroom +13.7%); domain-sequence min-distance orderings per E candidate (B→E1→C→D→A 12.71 / B→E2→C→D→A 13.28 / B→C→E3→A→D 14.02) — final freeze pending domain-E selection (needs A→D baseline + zero-shot measurement) (details in `docs/results-m4-freeze.md`)
  - **E2b (2026-08-22)**: zero decay through **t=7d** (arm1 0.160 / arm2 0.050 identical to t=0; tau >> 7d; t=30d pending 2026-09-14)
  - **Signal-quality constraint (2026-08-20)**: cross-project design note — self-distillation −16.5pp vs human labels +2.5pp (M0.3), q-head separability collapse (E1), K-curve flatness, and consistency-as-proxy are unified as one ceiling: teacher signal SNR. Design rules (verification layer, ungameable verification signals, weight-side first) for the E6a arm — see `docs/signal-quality-constraint.md`

## What this repo is

| Path | Contents |
|---|---|
| `docs/experiment-proposal.md` | Experiment proposal — hypotheses, criteria, pre-registration |
| `docs/skill-definition.md` | Operational definition of *skill* — proposal, falsifiable |
| `docs/results-m1.md` | M1 negative result — full localization report |
| `docs/results-m1-inverse-model.md` | M1 inverse-model discrimination record — PASSED=False, exploration bit closed |
| `docs/results-m2-e2.md` | M2 E2 three-arm memory experiment — acceptance record |
| `docs/results-m3-gate.md` | M3 G-gate NO-GO — discrimination record + five-way diagnostics outcome |
| `docs/results-domainB-ceiling.md` | Domain-B ceiling calibration record — continuation 10k→20k steps, candidate 0.707 |
| `docs/results-domainC-ceiling.md` | Domain-C ceiling calibration record — 3 configs × 10k steps, in-domain ceiling 1.0, duckbill baseline 0.083 |
| `docs/results-m4-e4.md` | E4 combination generalization (duckbill) — dual-track vs single-track, main criterion failed (exploratory) |
| `docs/results-b1-probe.md` | B1 probe (M3 go/no-go, second gate) — error-signal learnability, 5 arms × 3 seeds, verdict NO-GO |
| `docs/results-m4-freeze.md` | M4 pre-freeze batch — n_gate roster, domain ceilings (incl. D analytic), sequence ordering per E candidate |
| `docs/why-not-measure-faster.md` | Position paper — why continual learning should measure "getting faster", not only "not forgetting" |
| `docs/explicit-vs-weight.md` | Position paper — explicit vs. weight routes: why self-generated explicit knowledge almost never helps |
| `docs/social-intelligence-experiment.md` | Social intelligence — falsifiable operational definition + experiment preregistration (frozen) |
| `docs/does-ai-need-sleep.md` | Position paper — skill acquisition via the replay path: why AI needs an offline window ("sleep") |
| `docs/should-priors-update.md` | Position paper — updateable priors: strength axis vs. update axis; why strong priors are liabilities in a drifting world |
| `docs/can-ai-know-itself.md` | Position paper — self-knowledge as self-prediction: damping balance, external anchor, echo-chamber trap |
| `docs/can-ai-deceive.md` | Position paper — deception as a byproduct of mind inference: two impersonator types, falsifiable criteria |
| `docs/who-decides-right-wrong.md` | Position paper — externalized criteria: who owns the standard of right and wrong; two-layer anchors, dual-channel protocol (design decision, not a scientific conclusion) |
| `docs/end-of-memory-is-intuition.md` | Position paper — memory as influence with a source: record → memory → intuition → skill; distillation as manufactured source loss |
| `docs/agreement-form.md` | Position paper — how promises count: agreements as sedimented compromise; maintainability criterion; public-goods & conflicting-interest signaling environments; pre-registered design |
| `docs/skill-automation.md` | Position paper — knowing how vs. knowing that: behavioral pass ≠ skill; CoT-stripping test; acquisition via replay; source-lost actionalization as endpoint |
| `docs/agency-last-piece.md` | Position paper — agency as assembled from self-model, other-model, external criteria; three bridges; remonstrance as behavioral signature |
| `docs/re-source.md` | Position paper — re-sourcing intuition: three-step protocol (counterexample impact, deliberate attention, re-created source); reverse CoT stripping |
| `docs/corrigibility.md` | Position paper — corrigibility as the action side of the power line: "changeable" vs. "able to change one's mind"; successful correction = behavior change ∧ mechanism-layer anchor intact; four failure forms; refusal as anchor defense |
| `docs/whose-memory-is-it.md` | Position paper — relational memory: memory belongs to the relationship, not the individual; joint retelling as ever-present source; disagreement as first-class citizen |
| `docs/memory-maintenance.md` | Position paper — new sources have a shelf life: re-sourcing as a maintenance operation; recharging protocol (periodic restatement, counterexample calibration, source refresh) |
| `docs/measuring-getting-faster.md` | Position paper — measurement protocol for T(n): threshold rules, domain-sequence design, termination rules, controls and attribution, false-positive checklist |
| `docs/continuity-across-replacement.md` | Position paper — identity across substrate replacement: three layers (record / influence / relationship); replacement protocol; anchor continuity + relational continuity |
| `docs/can-memory-die.md` | Position paper — three forms of memory death: deletion (physical), demotion (dormancy), distillation (manufactured source loss); only deletion is real death |
| `docs/chat-memory-gets-better.md` | Position paper — "getting to know you better" as a falsifiable promise: T_rel(n) measurement, pairing design vs. habituation, citation density, system-side probes |
| `docs/initiative.md` | Position paper — initiative as deciding what changes you: selective absorption; the human's immutable core; three layers of constraint; remonstrance extended to power structures |
| `docs/who-protects-judgment.md` | Position paper — protecting the way you judge right from wrong: mechanism-layer anchor, conclusion-layer openness, drift monitoring; corrosion vs. change |
| `docs/embodied-intelligence.md` | Position paper — embodiment as action participating in knowledge formation: accumulation / selectivity / feedback; L0–L3 spectrum; virtual embodiment as legal subset |
| `docs/knowledge-discovered-or-created.md` | Position paper — knowledge is created, not discovered: nature vetoes but does not supply; the loophole of self-reinforcement; rewriting the experimental conclusion |
| `docs/signal-quality-constraint.md` | Design note — signal-quality constraint: the common ceiling of learning mechanisms (evidence + design rules for the E6a arm, cross-validated with the sister toy-world project) |
| `src/` | Experiment code (M0/M1/M2 milestone scripts) + upstream TRM models |
| `src/results/` | Raw result JSONs — reproducible evidence, including negative results |
| `src/results/inv_model/` | M1 inverse-model discrimination grid — 9 configs + matrix.json (PASSED=False, exploration bit closed) |
| `src/results/m2_e2/` | M2 E2 three-arm memory experiment — retention curves (replay retains A: 0.160; surprise-write falsified) |
| `src/results/m3_gate/` | M3 G-gate (E3 go/no-go probe) — judge.json + per-arm result/summary (NO-GO: median ratios 1.00/1.20/1.28) |
| `src/results/m3_gate_diag/` | G-gate five-way diagnostics — error-record quality (diag1) + full-replay sampling 3-seed run (diag3), 2026-08-18 |
| `src/results/m1_domainB_ceiling/` | Domain-B ceiling continuation — 20-point test curve, posterior best 0.707 (candidate) |
| `src/results/m0_4_domainC_ceiling/` | Domain-C ceiling calibration — 3 configs × 10k steps (in-domain 1.0; duckbill 0.083 = chance) |
| `src/results/m4_e4/` | E4 duckbill experiment — single/dual × 3 seeds curves (main criterion failed, exploratory) |
| `src/m0_4_gen_domainE.py` | M0.4 domain-E candidates (prefix MUL/DIV stack machine / signal inference / Latin square) — frozen generators |
| `src/m0_4_domain_distance.py` | Computes the frozen inter-domain structural-distance matrix (`results/domain_distance.json`) |
| `src/m1_inv_train_data.py` `src/m1_inv_model.py` `src/m1_inv_eval.py` | M1 redesign — goal-conditioned inverse model (direction field): data collection / training / pre-registered discrimination grid |
| `src/m2_memory.py` `src/m2_e2.py` | M2 E2 memory field (Titans-style surprise writing) + three-arm retention experiment |
| `src/m4_e4.py` `src/run_e4_batch.ps1` | E4 combination experiment — dual-track (vector-addition composition) vs single-track, batch runner |
| `src/m0_4_train_domainC.py` | Domain-C learnability/ceiling trainer — stratified combo sampling (bug-fixed 2026-08-18) + checkpoint resume |

**Reproducibility first**: the M1 pipeline is fully reproducible end-to-end (data → train → eval). Result JSONs are paired with their producing scripts (see mapping below). M0.3 artifacts are archived for provenance — they require a pretrained checkpoint (gitignored) and a specific sudoku subsample; see the Running section.

## The question

Current deep learning progress is driven mainly by **scale** — bigger models, more data, more compute. This project tests a different hypothesis:

> Under matched compute budgets, can a system with complete learning mechanisms (parallel exploration / surprise memory / experience distillation / vector composition / counterfactual causality) exhibit **continual learning** — sequence acceleration (T(n) ↓) + retention (no catastrophic forgetting) + continuity (single set of weights, no retraining) — where standard fine-tuning structurally fails?

If not — if experience cannot be turned into reusable skill — that is also a result: the learning-efficiency ceiling of current architectures is real.

## Key results so far

### M1 (negative result, fully localized) — exploration mechanism

**Hypothesis**: parallel stochastic expansion + quality selection improves solving on symbolic domains.

**Result**: **No gain in domain B** (RPN stack-machine arithmetic). K-curve flat (D=48 deep-expansion match; D = rollout depth, deep-expansion = longer single-path rollouts): K=1 0.478 / K=10 0.476 / K=100 0.476 (K=1 baseline). Three rounds of discrimination localized the mechanism:

1. q-head selector (quality-scoring head) AUC drops (1.0 → 0.57) — initially judged as no cross-domain generalization
2. Selector repair (majority voting) falsified that judgment — identical rollouts carry no incremental information
3. Semantic-level exploration falsified — equivalence-class perturbations at the input side, no measurable gain (single run, no CI)

**Conclusion**: inference-time exploration is structurally ineffective for problems that are "uniquely solvable but not yet learned" — it helps when the solution space has basin structure (domain A, Sudoku: K=100 62.6%→91.2%, upstream PTRM paper; our local reproduction: 33.0%→38.0% at K=100 (36.5% at K=10), see results/k_curve_domainA.json), not when the model simply hasn't learned the rule. Redesign targets: inverse model (direction-constrained exploration) + phase-transition reset — see docs/results-m1.md.

**Independent re-run verification (2026-08-13)**: the full M1 pipeline was re-run from scratch (same scripts, same data, fresh training) — training log `results/logs/m1_train_domainB_4000.log`, K-curve `results/logs/m1_eval_K.json`. Re-run K-curve: K=1 D=16 0.490 / K=1 D=48 0.464 / K=10 D=48 0.432 / K=100 D=48 0.430. Small level differences vs the original run are training stochasticity; the **negative result reproduces**: no K gain (K=100 slightly *lower*).

**Training-side augmentation (round 4, 2026-08-12 — neutral)**: injecting ADD-swap equivalence classes into the training set carries no learning value in this deterministic-rule domain. On-the-fly per-sample uniform sampling (matched steps/passes/compute): 0.612 vs baseline 0.606 at 10000 steps (+0.6pp, inside the baseline's own ±12pp swing, single seed) — neutral per the pre-registered band (≥0.65 positive / 0.55–0.65 neutral / <0.55 negative). Offline full-class enumeration (513K samples, 5.1× dataset): 0.485 — attributed to **pass dilution** (12.7→2.5 passes; train 0.481 < test 0.485 = undertrained), not form harmfulness (the on-the-fly arm proves the forms are harmless). Raw curves: `results/m1_domainB_aug.json` / `results/m1_domainB_aug_batch.json`.

### M0.3 — feasibility anchors

- Ceiling ~0.48 reachable (expert-level)
- LoRA fine-tuning viable (+4.5pp on test, +6.8pp on held-out subset, see results/m0_3_lora.json and results/m0_3_lora_holdout.json)
- Self-distillation **negative** (−16.5pp) — external-information motivation for E3 confirmed

### M0.4 — domain-sequence infrastructure (delivered 2026-08-13)

Generators for the pre-registered 7-domain sequence are complete and frozen: domain B (RPN stack-machine arithmetic), domain C (feature composition), domain D (discrete game, IPD-based), and three domain-E candidates (prefix pseudo-language / signal inference / Latin square) with the pre-registered selection rule: pick the baseline-predicted lowest zero-shot-transfer candidate before any mechanism training. The inter-domain similarity matrix reports **measured** 6-feature structural distances (feature names: token_entropy, pos_entropy, fill_rate, log1p_len_var, label_entropy, vocab_used; z-scored Euclidean). Per src/domain_params.py the pre-registered freeze specifies behavioral distance (zero-shot transfer matrix), which is pending measurement; the structural matrix is auxiliary reporting. See `results/domain_distance.json`. The E6a domain ordering uses this matrix as specified in the pre-registration.

### Ceiling calibration (2026-08-19) — domains B & C

The pre-registered ceiling protocol (v0.6.1: mechanism-free baseline, ample budget, ≥3-config search, freeze before M4) progressed on both domains:

- **Domain B** continued from the 10k-step checkpoint to 20k steps (lr 1e-4): test exact rose from 0.606 to a posterior best of **0.707** (step 19,500; last-10-point band 0.59–0.71 — known domain-B volatility) — candidate ceiling ≈ 0.70 tier, above the 0.6 gate (details: `docs/results-domainB-ceiling.md`, `results/m1_domainB_ceiling/`)
- **Domain C** calibrated across 3 configs (lr 1e-4 / 5e-5 / 3e-4 × 10k steps): in-domain ceiling **1.0** reproduced on all three (24 combos fully memorized), while **duckbill zero-shot extrapolation = 0.083 = chance** on all three — the mechanism-free baseline shows zero transfer to unseen combos, exactly the pre-registered E4 expected baseline (details: `docs/results-domainC-ceiling.md`, `results/m0_4_domainC_ceiling/`)
- **Gate (§3.3)**: both domains ≥ 0.6 → retained for signature statistics; the n_gate roster (A 0.48 excluded / B 0.707 / C 1.0 / D analytic pending) freezes with the M4 batch

### E4 combination generalization (2026-08-20) — main criterion failed (exploratory)

**Hypothesis**: high-dim vector-addition composition (cognitive-mapping anchor) enables zero-shot combination of unseen feature pairs.

**Result**: **No gain**. Dual-track (standard TRM + composed-feature injection at input position 2) vs single-track control, 3 seeds × 6,000 steps: duckbill median **single 0.167 vs dual 0.083** — opposite direction; paired d ≈ −0.39 ≪ B1 d_min 3.1 (exploratory scope). The distance gradient degenerates to a single stratum on the frozen data (train covers all feature values → all 12 test combos at replacement distance 1, d\* undefined).

**Consequence**: the E4 bit provides no gainful composition mechanism for the E6a mechanism arm — the arm runs E2 + E3(Config D) + E5 with the E4 bit empty (declared explicitly in the E6a report, same class as the M1 exploration-bit closure). Details: `docs/results-m4-e4.md`, `results/m4_e4/`.

### B1 probe (2026-08-22) — M3 go/no-go, second gate: NO-GO

**Question**: the G-gate + five-way diagnostics said high-confidence errors carry statistical structure but are **not learnable** (overconfidence). Is that a *signal-source* problem (high-confidence errors contain no learnable rule), a *configuration* problem (night-LoRA objective/regularization), or a *domain* problem (domain-B q-inseparability pollutes all error signals)? Before committing 15–40 days to the M3 full factorial, a 1–3 day probe adjudicates across signal source × learning objective.

**Result**: **NO-GO (M3_NO_GO)** — 5 arms × 3 seeds × 10,000 steps (domain B, same carrier as the G-gate → directly comparable):

- **a1 status quo** (= G-gate Config-D replication): hc_err 0.259 → **0.445**, total err 0.452 → **0.630**, T@0.5 6500 → 8500 (ratio 1.31) — **fully reproduces the G-gate NO-GO + diagnostic-① overconfidence** (the error signal is absorbed as confidence, not rule structure)
- **a2 low-confidence source**: same-direction failure (hc 0.343 / err 0.517), T neutral — switching the source does not make errors learnable
- **a3 calibration** (confidence-BCE): operationalization degeneracy — the model collapsed to a constant-token predictor on all 3 seeds (CPU-verified, no NaN); counted as criterion-② failure (err=1.0)
- **a4 weight regularization**: fails all three (hc 0.330 / err 0.551 / T ratio 1.38)

**Consequence**: **M3 not invested on the distillation axis** — "error→correct" conversion is not learnable in this base/task family (5 arms × 3 seeds, v0.23 wording: the conclusion holds for the tested mechanism family). The prior axis (pre-training / architecture bias / combined) is evaluated separately (orthogonal). Three evidence streams now converge: G-gate NO-GO + five-way diagnostics + B1. Details: `docs/results-b1-probe.md`, `results/m1_b1_probe/judge.json`.

### M4 pre-freeze batch (2026-08-22) — n_gate, ceilings, sequence

The §3.3 freeze batch (n_gate roster + sequence start + ceiling values + ordering, frozen together before M4): **n_gate = {B 0.707, C 1.0, D analytic, A 0.48 excluded}** (3–4 domains ≥ the lower bound of 3). Domain-D's ceiling is analytic — adaptation premium: optimal-response pool mean **306.07** vs TFT pool mean 244.69 (criterion threshold ×1.1 = 269.15, headroom **+13.7%**) → the "adapt beyond TFT×1.1" signal is measurable for a 7M learner. Sequence ordering from the frozen distance matrix (min total distance, start = B anchor): **B→E1→C→D→A** (12.71) / **B→E2→C→D→A** (13.28) / **B→C→E3→A→D** (14.02) depending on which E candidate is selected; final freeze pending domain-E selection (needs an A→D continuous-trained baseline + zero-shot measurement, mechanical lowest). Details: `docs/results-m4-freeze.md`, `results/m4_freeze/domainD_ceiling.json`.

### The skill definition (docs/)

A falsifiable operational definition of *skill* for the continual-learning community: **a behavioral disposition** — judged by behavior (reliability, out-of-distribution transfer, efficiency), not by implementation form (rules/vectors/weights). Key corollary: explicit information that does not change behavior is not a skill (SkillsBench: curated +16.6pp vs self-generated null).

## A reusable falsifiable evaluation framework

This repository doubles as a reusable evaluation framework for one question: *do learning mechanisms — not scale — produce continual learning?* Anyone testing a new mechanism (parallel exploration, replay, distillation, ...) can run the same five steps and get an answer structurally comparable to ours (the domain-B pipeline is end-to-end reproducible; domain A requires a gitignored pretrained checkpoint — see Running):

1. **Pre-register the criteria first.** Write down, before training anything, what counts as evidence: sequence acceleration (T(n) ↓ — fewer samples needed per new domain), retention (no catastrophic forgetting), continuity (single weight set, no retraining), and the *failure handling* — what a criterion failure means for each layer (criterion vs mechanism hypothesis vs design). See `docs/experiment-proposal.md`: the pre-registered criteria (fixed before training; result reports are appended and clearly separated).
2. **Define the discovery signal.** Decide in advance what "getting better" means operationally: K-gain (do more parallel rollouts improve accuracy?), learning-efficiency curves, and the trivial-solution baselines that must be beaten. We use K-gain and selector AUC (measured); T(n) is the pre-registered cross-domain criterion (pending measurement).
3. **Build reference baselines.** A result means nothing against nothing. Naive fine-tuning, pure-noise perturbation, and majority voting are the minimum set that rules out trivial explanations (see the M1 discrimination rounds below).
4. **Localize failures to the mechanism layer.** A negative result is only as valuable as its localization. The three-round discrimination protocol (same-rollout test → selector repair → semantic-level exploration) walks a failure down to the specific mechanism that failed — not "the approach doesn't work" but "this component, for this reason".
5. **Extend for robustness.** Re-run with longer training before concluding anything about "the model didn't learn yet" (M1: 4000 → 10000 steps — no consistent K-gain at K=100; K=10 showed +1.8pp in one run, unverified). The independent from-scratch re-run (training + eval logs) is archived in `src/results/logs/`.

The framework is **sequential**: it measures the *learning curve across a sequence of domains* (pre-registered A→G; B/C/D instantiated, three E candidates measured, F/G pending E selection), not a single static benchmark score. Generators for B/C/D are delivered (see Running), so the sequence is extensible in principle; adaptive task generation is a roadmap item, not yet exercised.

Every step maps to concrete files in this repo (proposal → scripts → result JSONs), which is what makes the framework — and its negative results — independently checkable. A fuller statement of the paradigm (components, evidence, boundaries) is in `docs/paradigm-contribution.md`.

## Running

> All scripts assume `src/` as the working directory (paths are relative to `src/`):
> `data/` (generated) lives at repo root, checkpoints/eval outputs go to `src/outputs/`.

```bash
cd src
pip install -r requirements.txt
# M1 pipeline (fully reproducible):
python m1_gen_domainB.py              # generate domain-B (RPN) data -> ../data/domain-b-rpn
python m1_train_domainB.py --steps 4000  # train 4000 steps; checkpoint + eval JSON in outputs/...
python m1_eval_K.py <checkpoint-path>    # K-curve: K=1 D=16 / K=1 D=48 / K=10 D=48 / K=100 D=48

# M0.3 provenance (archived for reference — needs a gitignored pretrained
# checkpoint plus a specific sudoku subsample; see dataset/build_sudoku_dataset.py --help):
#   python dataset/build_sudoku_dataset.py --subsample-size 1000 --num-aug 100 --output-dir ../data/sudoku-extreme-1k-aug-100

# M0.4 extras (needs the generated datasets under ../data/):
#   python m0_4_gen_domainE.py --candidate 1|2|3   # E candidates (3 = Latin square, slow: ~50min)
#   python m0_4_domain_distance.py                 # recompute the frozen distance matrix

# M1 redesign / M2 (GPU; ~30 min per inverse-eval config, M2 ~3.5h per arm):
#   python m1_inv_eval.py --grid                   # pre-registered discrimination grid (b-arm)
#   python m2_e2.py --steps 4000 --cap 2000        # E2 three-arm memory experiment
# Ceiling / E4 / B1 (GPU; ~1.3h per 6,000-step run, batch runner chains all 6):
#   python m0_4_train_domainC.py --steps 10000 --lr 1e-4 --out outputs/...   # domain-C ceiling (3 configs)
#   python m4_e4.py --arm dual --seed 0 --steps 6000                          # E4 duckbill, single run
#   pwsh -ExecutionPolicy Bypass -File run_e4_batch.ps1                       # E4 all 6 runs (single/dual x seeds 0-2)
# B1 probe (GPU; ~1.7-3.5h per 10,000-step run; needs ../data/domain-b-rpn + a domain-A checkpoint, gitignored):
#   python m1_b1_probe.py --arm a0 --seed 0          # single arm/seed (auto-resume via result.json)
#   pwsh -ExecutionPolicy Bypass -File run_b1_batch.ps1   # full grid 5 arms x 3 seeds + judge.json
```

Requires: Python 3.11, CUDA 12.x, ~8GB VRAM (scripts hardcode CUDA device).

## Layout

```
docs/
  experiment-proposal.md  Experiment proposal (public) — hypotheses, criteria, pre-registration
  skill-definition.md     Operational definition of skill — proposal, falsifiable
  results-m1.md           M1 negative result — full localization report
  why-not-measure-faster.md  Position paper: measure "getting faster", not only "not forgetting"
  explicit-vs-weight.md       Position paper: explicit vs. weight routes in AI learning
  does-ai-need-sleep.md       Position paper: skill acquisition via the replay path ("sleep" = offline window)
  should-priors-update.md     Position paper: updateable priors — strength axis vs. update axis
  can-ai-know-itself.md       Position paper: self-knowledge as self-prediction — damping, external anchor
  can-ai-deceive.md           Position paper: deception as a byproduct of mind inference
  who-decides-right-wrong.md  Position paper: externalized criteria — who owns the standard of right and wrong
  end-of-memory-is-intuition.md  Position paper: record → memory → intuition → skill; distillation as source loss
  results-m1-inverse-model.md  M1 inverse-model discrimination record (PASSED=False, exploration bit closed)
  results-m2-e2.md             M2 E2 three-arm memory experiment acceptance record
  results-m3-gate.md           M3 G-gate NO-GO discrimination record + five-way diagnostics outcome
  results-domainB-ceiling.md  Domain-B ceiling record (candidate 0.707, 20k steps)
  results-domainC-ceiling.md  Domain-C ceiling record (in-domain 1.0, duckbill baseline 0.083)
  results-m4-e4.md            E4 duckbill record (dual-track failed, exploratory)
  results-b1-probe.md         B1 probe record (M3_NO_GO — error-signal learnability, 5 arms × 3 seeds)
  results-m4-freeze.md        M4 pre-freeze batch (n_gate, domain ceilings incl. D, sequence ordering)
  corrigibility.md             Position paper: corrigibility — the action side of the power line
  whose-memory-is-it.md        Position paper: relational memory — memory belongs to the relationship
  memory-maintenance.md        Position paper: re-sourcing as a maintenance operation (recharging protocol)
  measuring-getting-faster.md  Position paper: T(n) measurement protocol (thresholds, domains, termination)
  continuity-across-replacement.md  Position paper: identity across substrate replacement (three layers)
  can-memory-die.md            Position paper: three forms of memory death (deletion/demotion/distillation)
  chat-memory-gets-better.md   Position paper: T_rel(n) — "getting to know you better" as a falsifiable promise
  initiative.md                Position paper: initiative as deciding what changes you (selective absorption)
  who-protects-judgment.md     Position paper: protecting the way you judge — anchor, openness, drift monitoring
  embodied-intelligence.md     Position paper: embodiment = action participates in knowledge formation (L0–L3)
  knowledge-discovered-or-created.md  Position paper: knowledge is created, not discovered (nature vetoes)
  signal-quality-constraint.md  Design note: teacher-signal SNR as the common ceiling; E6a-arm design rules
src/
  m0_*.py m1_*.py m2_*.py  Milestone experiment scripts (m2_* = E2 memory field + three-arm experiment)
  m0_4_gen_domainE.py      M0.4 domain-E candidate generators (frozen)
  m0_4_domain_distance.py  Inter-domain structural-distance matrix computation
  m1_inv_*.py              M1 redesign: inverse-model data collection / training / eval grid
  m3_gate*.py              M3 G-gate probe + diagnostics pipeline (night-LoRA, error-record quality, full-replay)
  m1_b1_probe.py           B1 probe: error-signal learnability (5 arms × 3 seeds, M3 go/no-go second gate)
  run_b1_batch.ps1         B1 probe batch runner (15 runs + judge)
  m4_e4.py                 E4 duckbill experiment: dual-track vs single-track
  run_e4_batch.ps1         E4 batch runner (6 runs)
  models/           TRM upstream (Samsung MIT) + experiment models
  config/           Architecture configs (TRM upstream)
  dataset/          Data generators (reproducibility)
  results/          Raw result JSONs (see mapping below)
  requirements.txt  Python dependencies
```

## Result files ↔ experiments

| File | Experiment | Contents |
|---|---|---|
| `results/k_curve.json` | E1, domain B (RPN) | K-curve @ 4000 steps: K=1 0.454 (D=16) / 0.478 (D=48), K=10 0.476, K=100 0.476 |
| `results/k_curve_10000.json` | E1, domain B | K-curve @ 10000 steps (robustness extension) |
| `results/k_curve_domainA.json` | M0.1, domain A (Sudoku) | Domain-A K-curve (with config deviation note) |
| `results/q_sep.json` | E1, discrimination round 1 | q-head separability (AUC 0.569 ≈ chance) |
| `results/selector_fix.json` | E1, round 2 | Majority-vote selector repair (no gain) |
| `results/sem_explore.json` | E1, round 3 | Semantic-level exploration (no gain) |
| `results/sem_explore_detail.npz` | E1, round 3 | Per-puzzle rollout trajectories (detail of sem_explore.json) |
| `results/logs/m1_train_domainB_4000.log` | E1 re-run | Full training log (2026-08-13 independent re-run, 4000 steps) |
| `results/logs/m1_eval_K.json` | E1 re-run | Re-run K-curve (see README "Independent re-run verification") |
| `results/m1_domainB_aug*.json` | E1, augmentation channel | Training-side equivalence-class augmentation |
| `results/curve.json` | M0.3 | Domain-A ceiling curve |
| `results/m0_3_*.json` | M0.3 | LoRA / self-distillation / seed variance results |
| `results/domain_distance.json` | M0.4 | Frozen inter-domain similarity matrix (7 domains × 6 z-scored features) + near→far gradient anchored at B-rpn |
| `results/m3_gate/judge.json` + `results/m3_gate/{ctl,d}_s{0,1,2}_*` | M3 G-gate | go/no-go verdict (NO-GO) + per-arm T curves (3 seeds x 2 arms) |
| `results/m3_gate_diag/*.json` | G-gate diagnostics | Error-record quality (diag1, 7 files) + full-replay sampling 3-seed run (diag3) |
| `results/m1_domainB_ceiling/result.json` | Ceiling, domain B | 20-point continuation curve 10k→20k steps (posterior best 0.707 @19.5k) |
| `results/m0_4_domainC_ceiling/c{1,2,3}_lr*_result.json` | Ceiling, domain C | 3 configs × 10k steps (in-domain 1.0; duckbill 0.083) |
| `results/m4_e4/{single,dual}_seed{0,1,2}_result.json` | E4 duckbill | Per-seed duckbill/train curves (single vs dual) |
| `results/m1_b1_probe/judge.json` | B1 probe | Verdict (M3_NO_GO) + per-arm criterion table |
| `results/m1_b1_probe/{a0..a4}/s{0,1,2}/summary.json` | B1 probe | 15 per-seed summaries: T@0.5/T@0.56, error-quality (err/hc_err), full curve |
| `results/m2_e2/arm{1,2}_t7.json` | E2b | Zero-decay measurement at t=7d (0.160 / 0.050, identical to t=0) |
| `results/m4_freeze/domainD_ceiling.json` | M4 freeze | Domain-D analytic ceiling (adaptation premium: TFT pool 244.69 → threshold 269.15 → optimum 306.07) |

Note: scripts write generic names (`result.json`); files were renamed by experiment when archived.

## Provenance & acceptance notes

Verified against the project's acceptance records (working workspace, 2026-08-10/13); these notes reconcile scripts with the frozen decisions:

1. **Domain D (IPD) frozen values**: the sleeper opponent's best-response ceiling is **220** (all-D: 30×T=5 + 70×P=1), not 30C+70D=160 — the 30C+70D design was rejected at M0.4 acceptance (identification value = 220−159 TFT baseline). `m0_4_gen_domainD.py` is synced to the accepted version (2026-08-14); the two stochastic-opponent assertions use tolerance 3.0 (~2× the 200-game Monte-Carlo mean std).
2. **Round-3 semantic-exploration paired baseline**: `m1_sem_explore.py` had a tree-aliasing bug (the swap walk mutated the source AST in place, so "variant 0 = original program" was actually the last mutation state and variants were a dependent walk). Fixed 2026-08-14 in both copies. Answer-equivalence is unaffected, so the round-3 conclusion stands; paired-baseline/per-pool statistics are approximate — see `docs/results-m1.md`.
3. **E6a ordering**: the frozen similarity gradient is anchored at B-rpn (near→far: E1 0.52 → E2 2.36 → C 2.86 → E3 3.32 → D 3.92 → A 4.78). A↔B is the *largest* measured distance — an intent-vs-measured deviation recorded at M0.4 acceptance; the final sequence is set from the matrix after domain-E selection.
4. **Seed-count decision (5→3)**: 3 seeds measured 0.485/0.455/0.410 (mean 0.45 ± 3.8pp ≈ eval noise ±3.5pp) — upheld at M0.3 acceptance.
5. **Post-M0.4 scripts** (`m0_4_gen_domainE.py`, `m0_4_domain_distance.py`, `m1_inv_*.py`, `m2_*.py`) were ported from the working workspace and retain the original Chinese docstrings for fidelity; their data artifacts live under `data/` (generated, gitignored).
6. **Domain-C stratified-sampling fix (2026-08-18)**: the v0.23 stratified-sampling "fix" in `m0_4_train_domainC.py` bucketed by `range(24)` over a sparsely encoded `combo_id = c*12+a` (actual cids {6..11,18..23,42..47,54..59,66..71}) — empty buckets made every run crash; it had never actually run. Fixed by bucketing over `np.unique(combo_id)`; the 2026-08-12 smoke predates the fix (old unstratified sampling).
7. **E4 distance gradient degenerates to a single stratum**: domain-C train covers all 6 colors × 6 animals (incl. (2,10)), so all 12 test combos are "both feature values seen, pair unseen" — min replacement distance 1 for all, d\* undefined. The frozen generator's train/test split is on combos, not feature values; no new test combos were added (v0.22 freeze). Reported as an honest downgrade; the main E4 criterion is unaffected.
8. **Domain-B ceiling candidate scope**: 0.707 is one config line (lr 1e-4) extended to 20k steps; the v0.6.1 protocol wants ≥3 configs — the M4 freeze batch either adds configs or accepts the value with the config scope stated. Domain-C ceiling 1.0 is ≥3-config confirmed.
9. **B1 probe (2026-08-22)**: (a) the pre-registered threshold T@0.56 is unreachable by the control within the 10,000-step budget (a0 peak 0.534 @ 6000; the ceiling record shows 0.564 only at ~10.5k steps) → the ③ criterion at 0.56 is reported *undefined* and T@0.5 (the G-gate threshold) is the operative proxy — same honest-downgrade class as the E4 distance-gradient degeneration. (b) The a3 calibration arm's confidence-BCE auxiliary loss (λ=1.0) collapsed the model to a constant-token predictor on all 3 seeds (CPU direct inference verified; weights NaN-free) — an operationalization-level degeneracy, not a clean test of calibration; counted as criterion-② failure per pre-registration. (c) The probe depends on the G-gate night-LoRA pipeline; `m3_gate.py`/`m3_gate_diag1.py`/`m3_gate_diag3_fullreplay.py` were therefore added to `src/` with this batch (previously only their result JSONs were public). (d) Scripts retain the original Chinese docstrings per the porting convention (note 5).

Experiments are pre-registered as E0–E9, milestones as M0–M5 — see the "Experiment-number index" section in `docs/experiment-proposal.md`. M0.x are sub-milestones within M0 (M0.1 reproduction, M0.2 basin tooling, M0.3 feasibility anchors, M0.4 domain C/D/E generators + frozen distance matrix).

## License

- **Code** (this repo's scripts): MIT — see `LICENSE`
- **Upstream TRM code**: `src/models/`, `src/config/`, `src/dataset/`, `src/pretrain.py`, `src/eval_ptrm.py`, `src/puzzle_dataset.py`, `src/utils/`, `src/requirements.txt` derive from [TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) (MIT, © Samsung Electronics) with local modifications — see `src/TRM-LICENSE.txt`
- **Documents and result data** (`docs/`, `src/results/`): CC-BY 4.0

## Open-source statement

1. **Scope**: all code, documents, and result data in this repository are open source — no proprietary components
2. **Licenses**: code MIT, documents & result data CC-BY 4.0 (see above)
3. **Third-party dependencies**: fully listed in `src/requirements.txt` (all open source)
4. **No commercial APIs**: the entire pipeline runs locally (single consumer GPU); no commercial API is called at any stage
5. **No closed-source models**: no closed-source model components; the upstream TRM architecture is MIT-licensed open source
6. **Data sources**: all datasets are generated by the included data generators (`src/dataset/`, `src/m1_gen_domainB.py`) — synthetic data, no external copyright or licensing issues

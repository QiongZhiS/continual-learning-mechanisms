# Continual Learning Mechanisms

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21902053.svg)](https://doi.org/10.5281/zenodo.21902053)

> **How do skills form? How does memory update? How does agency emerge? I trust experiments to answer.**

A falsifiable investigation into whether **learning mechanisms** — not scale — can produce continual learning: accelerating across tasks, retaining old knowledge, without retraining.

- **Scale**: ~7M parameter recursive reasoning models (TRM), single consumer GPU (RTX 4060)
- **Method**: pre-registered experiments, budget-matched controls, honest negative results
- **Status**: M0 complete — M0.4 domain-sequence generators delivered (domains B/C/D/E instantiated, inter-domain similarity matrix frozen, see `results/domain_distance.json`); M1 (exploration mechanism) main criterion failed in the symbolic domain — negative result fully localized and independently re-run verified; redesign in progress — goal-conditioned inverse model implemented and trained (scripts included; pre-registered discrimination grid pending a GPU run); M2 (E2 memory: surprise-vs-full replay) implemented (scripts included; three-arm experiment pending)

## What this repo is

| Path | Contents |
|---|---|
| `docs/experiment-proposal.md` | Experiment proposal — hypotheses, criteria, pre-registration |
| `docs/skill-definition.md` | Operational definition of *skill* — proposal, falsifiable |
| `docs/results-m1.md` | M1 negative result — full localization report |
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
| `src/` | Experiment code (M0/M1/M2 milestone scripts) + upstream TRM models |
| `src/results/` | Raw result JSONs — reproducible evidence, including negative results |
| `src/m0_4_gen_domainE.py` | M0.4 domain-E candidates (prefix MUL/DIV stack machine / signal inference / Latin square) — frozen generators |
| `src/m0_4_domain_distance.py` | Computes the frozen inter-domain structural-distance matrix (`results/domain_distance.json`) |
| `src/m1_inv_train_data.py` `src/m1_inv_model.py` `src/m1_inv_eval.py` | M1 redesign — goal-conditioned inverse model (direction field): data collection / training / pre-registered discrimination grid |
| `src/m2_memory.py` `src/m2_e2.py` | M2 E2 memory field (Titans-style surprise writing) + three-arm retention experiment |

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
src/
  m0_*.py m1_*.py m2_*.py  Milestone experiment scripts (m2_* = E2 memory field + three-arm experiment)
  m0_4_gen_domainE.py      M0.4 domain-E candidate generators (frozen)
  m0_4_domain_distance.py  Inter-domain structural-distance matrix computation
  m1_inv_*.py              M1 redesign: inverse-model data collection / training / eval grid
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

Note: scripts write generic names (`result.json`); files were renamed by experiment when archived.

## Provenance & acceptance notes

Verified against the project's acceptance records (working workspace, 2026-08-10/13); these notes reconcile scripts with the frozen decisions:

1. **Domain D (IPD) frozen values**: the sleeper opponent's best-response ceiling is **220** (all-D: 30×T=5 + 70×P=1), not 30C+70D=160 — the 30C+70D design was rejected at M0.4 acceptance (identification value = 220−159 TFT baseline). `m0_4_gen_domainD.py` is synced to the accepted version (2026-08-14); the two stochastic-opponent assertions use tolerance 3.0 (~2× the 200-game Monte-Carlo mean std).
2. **Round-3 semantic-exploration paired baseline**: `m1_sem_explore.py` had a tree-aliasing bug (the swap walk mutated the source AST in place, so "variant 0 = original program" was actually the last mutation state and variants were a dependent walk). Fixed 2026-08-14 in both copies. Answer-equivalence is unaffected, so the round-3 conclusion stands; paired-baseline/per-pool statistics are approximate — see `docs/results-m1.md`.
3. **E6a ordering**: the frozen similarity gradient is anchored at B-rpn (near→far: E1 0.52 → E2 2.36 → C 2.86 → E3 3.32 → D 3.92 → A 4.78). A↔B is the *largest* measured distance — an intent-vs-measured deviation recorded at M0.4 acceptance; the final sequence is set from the matrix after domain-E selection.
4. **Seed-count decision (5→3)**: 3 seeds measured 0.485/0.455/0.410 (mean 0.45 ± 3.8pp ≈ eval noise ±3.5pp) — upheld at M0.3 acceptance.
5. **Post-M0.4 scripts** (`m0_4_gen_domainE.py`, `m0_4_domain_distance.py`, `m1_inv_*.py`, `m2_*.py`) were ported from the working workspace and retain the original Chinese docstrings for fidelity; their data artifacts live under `data/` (generated, gitignored).

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

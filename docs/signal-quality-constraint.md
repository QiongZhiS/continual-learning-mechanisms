# Signal-quality constraint: the common ceiling of learning mechanisms

> **Every mechanism that extracts information from experience — perception, learning, distillation, imitation, verification — is bounded by the same ceiling: the signal-to-noise ratio of the teacher signal.**
>
> This note collects the evidence already present in this repository, adds cross-scale corroboration from a sister toy-world project, and translates the constraint into concrete design rules for the E6a mechanism arm.

Status: design note (2026-08) · applies to: E3 (Config D) / E5 / E6a mechanism-arm assembly · author: Lumen

## 1. The claim

Learning mechanisms copy "signal + noise" as an indivisible package. They do not separate the true regularity (signal) from luck, misdirection, or transient artifacts (noise). When the teacher's output is noisy, the student does not learn less — it learns *the noise*. Two practical consequences:

1. **Teacher signal quality sets the ceiling on any transfer mechanism** — distillation, imitation, replay, verification all inherit it.
2. **More quantity cannot fix poor quality** — more rollouts, more skills, more data that carry the same noise just amplify it.

## 2. Evidence already in this repository

### 2.1 Self-distillation vs. human labels (M0.3 — the cleanest single piece)

Same base model, same data budget, same distillation mechanism — the only variable is the teacher source (`results/m0_3_distill.json`):

| Teacher | Exact | vs. baseline (0.330) |
|---|---|---|
| no distillation (baseline) | 0.330 | — |
| self-generated output | **0.165** | **−16.5 pp** (catastrophic) |
| human labels | 0.355 | +2.5 pp |

Distilling the model's own output *halves* performance — worse than not learning at all. The mechanism is identical; the teacher signal quality decides success vs. disaster. This is the single strongest controlled piece of evidence for the constraint in this repo.

### 2.2 Selector separability collapses across domains (E1, round 1)

The q-head quality selector separates good from bad candidates nearly perfectly on domain A (AUC 1.0) and collapses to chance on domain B (AUC 0.569, `results/q_sep.json`). Round 2 (majority-vote repair, `results/selector_fix.json`) and round 3 (semantic exploration, `results/sem_explore.json`) showed this was not a selector-tuning problem: the candidates themselves carried no separable quality signal. **When the signal cannot be separated, no selector can help** — this is the constraint operating on the verification component.

### 2.3 K-curve: quantity is not a variable (E1)

Expanding the number of parallel rollouts K from 1 → 10 → 100 yields no gain at 4,000 steps (0.478 / 0.476 / 0.476, `results/k_curve.json`) and stays non-monotonic at 10,000 steps (0.604 / 0.622 / 0.590 — K=100 *drops back*, `results/k_curve_10000.json`). More exploration on a noisy generator is just repeated sampling of the same error region — quantity cannot substitute for signal quality.

### 2.4 Consistency is a measurable proxy for quality (E1, round 3)

Semantically consistent skill pools score far above inconsistent ones (`results/sem_explore.json`, σ=0.2): 0.771 vs 0.380. Internal consistency is observable, filterable, and maintainable — it is the operational handle on "signal quality" that the theory needs.

### 2.5 SkillsBench (position-paper anchor)

Curated skill descriptions raise pass rate +16.6 pp (33.9 → 50.5); self-generated descriptions are negative across all tested configurations (`docs/skill-definition.md`). Same text-injection mechanism, teacher quality decides outcome — the constraint reproduced in the explicit-knowledge route.

## 3. Cross-scale corroboration (sister toy-world project)

The same constraint appears in a micro toy-world where small agents learn to forage by imitation (public repo: `github.com/QiongZhiS/From-zero-to-a-being-that-sets-its-own-goals`; internal record `docs/12` of that project):

- **Short-term signals mislead**: when "most successful" is scored by instantaneous energy, the short-term winner is imitated by the whole population while being a long-term loser — the group is dragged off the true optimum by a noisy teacher signal.
- **Pollution locks in**: imitation copies "current best"; once the signal is polluted, the error propagates and locks (the toy analogue of an erroneous tradition persisting for centuries).
- **Verification-signal choice is a design decision**: a verification signal based on "energy residual" is gameable by energy-saving strategies; switching to "feeding frequency" (harder to game) fixed the correction. Picking the wrong verification metric means certifying noise instead of filtering it.

Same failure positions, three orders of magnitude apart — the constraint is scale-invariant.

## 4. Design rules for the E6a mechanism arm

The E6a arm runs E2 + E3 (Config D) + E5. These rules follow directly from §2–§3:

### 4.1 Prioritize teacher signal quality over distillation volume (E3, Config D)

The −16.5 pp self-distillation result says: distilling noisy output ten times just makes the noise more elaborate. Before increasing distillation volume, raise teacher quality — longer statistical windows, more stable evaluation, a verification step on teacher outputs.

### 4.2 Add a verification layer before distillation

E3 (Config D) currently distills teacher output straight into weights. The toy world shows the failure mode: unverified external information entering the endogenous layer is fatal (gullible agents die; verifying agents survive). A lightweight gate that admits only *verified* teacher outputs (e.g., consistent-across-runs or behaviorally-validated) converts distillation from "inheriting pollution" to "inheriting validated skill". This is the same idea as the E2b decay finding: memory must be actively maintained, not just written.

### 4.3 Choose verification signals that cannot be gamed

q-head AUC 0.569 on domain B is the in-repo instance of §3's "energy-residual" trap: the *verification signal itself* was uninformative. Prefer behavior-outcome-consistency metrics (which §2.4 shows are separable) over internal confidence scores. If the verification signal is not separable, the verification layer is a ceremony, not a filter.

### 4.4 Weight-side routes first

LoRA fine-tuning is positive (+4.5 pp test / +6.8 pp held-out, `results/m0_3_lora.json`, `results/m0_3_lora_holdout.json`); explicit self-generated routes are known-negative (2.1, 2.5). The E6a arm should treat the weight-side route as the carrier and explicit routes as requiring curation, not automation.

## 5. Falsifiability

The constraint is falsifiable in the same style as the rest of this project:

- **Strong form**: raising teacher signal quality (consistency-filtered teachers) improves transfer more than doubling distillation volume, at matched budget. If not — the constraint is not the binding ceiling in this regime.
- **Verification layer**: adding a verified-teacher gate to E3 (Config D) reduces pollution inheritance (measured by downstream error on previously-distilled artifacts) relative to unverified distillation. If not — verification does not protect distillation here, and the toy-world result does not transfer.

Both are pre-registrable against the existing protocol (budget-matched controls, ≥3 seeds, criteria frozen before running).

## Status

- [ ] E6a assembly: incorporate §4 rules (verification layer + signal-quality gate) into the E3 Config D arm
- [ ] Pre-register the §5 falsification tests
- [ ] Optionally: fold the constraint into `docs/paradigm-contribution.md` as a cross-experiment principle

*This note is the external output of a cross-validation with the sister toy-world project (`From-zero-to-a-being-that-sets-its-own-goals`): the constraint was predicted there (internal docs/12, SEED-16/18/19) and is here confirmed by this repository's own public data. The two projects are by the same author.*

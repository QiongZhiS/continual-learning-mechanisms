# A Falsifiable Evaluation Framework for Continual Learning

> **Author**: Lumen
> **Scope**: submitted material for the "paradigm contribution" evaluation track (dynamic scientific evaluation frameworks / exploration-paradigm design). This document states the framework itself — what it measures, how it is used, what evidence backs it, and where its boundaries are.

## 1. What this is

This repository contains a negative result, but it is built on a **reusable framework**, and the framework is the more durable contribution. It answers one question operationally:

> Under matched compute budgets, can learning mechanisms — not scale — produce continual learning?

The framework is the complete protocol for answering that question in a falsifiable, comparable, and extensible way. It has two halves:

- **A dynamic evaluation framework**: how to measure *learning efficiency over a sequence of domains*, rather than a single static score.
- **A falsifiable exploration paradigm**: how to pre-register criteria, localize failures, and turn negative results into evidence.

## 2. The dynamic evaluation framework

### 2.1 What it measures

Most continual-learning benchmarks focus on **retention**; forward-transfer metrics exist (GEM, Lopez-Paz & Ranzato 2017) but measure accuracy transfer rather than data efficiency and are rarely pre-registered. This framework measures **learning efficiency** — whether a system *gets faster at learning new domains* — with three pre-registered signals:

| Signal | Definition | Status in this repo |
|---|---|---|
| T(n) | Samples needed for a new domain, as a function of domain index n — T(n) non-increasing is the pre-registered acceleration component (a necessary condition; full signature: T(n) ∧ retention ∧ continuity, docs/experiment-proposal.md §9) | Criterion — **no data points yet** (pending training of later domains) |
| K-gain | Accuracy vs parallel-rollout budget K (more exploration → better solving?) | Measured (M1: flat in domain B) |
| Selector AUC | Quality-selector discriminability (does the mechanism know a good path when it sees one?) | Measured (0.569 ≈ chance, domain B) |

Signals are measured *across the domain sequence* and across training duration (4000 → 10000 steps as a robustness check).

### 2.2 The environment: an extensible domain sequence

- The pre-registration contract defines a 7-domain sequence (A→G). Currently instantiated: A (Sudoku reproduction), B (RPN arithmetic), C (feature composition), D (IPD game); three E candidates (prefix pseudo-language / signal inference / Latin square) have measured feature vectors awaiting the pre-registered selection; F/G are placeholders to be extended from E's family once E is selected (src/domain_params.py).
- Generators for B/C/D are delivered (`src/m1_gen_domainB.py`, `src/m0_4_gen_domainC.py`, `src/m0_4_gen_domainD.py`, `src/dataset/`); E-candidate features are measured, E/F/G generators are pending. The sequence is extensible in principle, not yet exercised end-to-end.
- The inter-domain similarity matrix (`src/results/domain_distance.json`) reports 6 measured z-scored structural features. Per src/domain_params.py, the pre-registered freeze specifies *behavioral* distance (zero-shot transfer matrix), which is pending measurement; the structural matrix is auxiliary reporting.

### 2.3 Reference baselines

A discovery signal means nothing without trivial-solution baselines. The framework mandates: naive fine-tuning, pure-noise perturbation, and majority voting. These rule out the three standard trivial explanations (nothing learned / noise artifact / ensemble effect).

## 3. The falsifiable exploration paradigm

### 3.1 Pre-registration contract

The experiment is pre-registered before any result exists (`docs/experiment-proposal.md`, frozen): hypotheses, criteria, statistical protocol, and — critically — **failure handling**, declared per layer:

- Criterion failure → the definition/claim is modified or abandoned;
- Mechanism-hypothesis failure → only that mechanism is falsified, the framework stands;
- Trivial-solution failure → the criterion design is invalid, not the claim.

This per-layer separation is what makes a negative result *interpretable* rather than merely disappointing.

- Framework-level failure: if a pre-registered criterion cannot be executed as specified (e.g., T(n) cannot be measured because later domains cannot be run), the framework itself is falsified as a usable protocol — the framework is not exempt from its own criteria.

### 3.2 Failure localization protocol

When a mechanism fails, the framework requires walking the failure down to the component level. M1 is the worked example (three rounds):

1. Same-rollout test — the selector's AUC drops (1.0 → 0.569); initial hypothesis: no cross-domain generalization.
2. Selector repair — majority voting over identical rollouts falsified that hypothesis: the same rollouts carry no incremental information.
3. Semantic-level exploration — equivalence-class input perturbation: no measurable gain, falsified.
Final attribution: exploration adds no information the model does not already have.

Conclusion: the exploration mechanism fails because it *adds no information the model does not already have* — a mechanism-layer explanation, not "the approach doesn't work".

### 3.3 Robustness extension

Negative results are checked for the "model wasn't trained enough" escape hatch: M1 was re-run at 10000 steps (vs 4000) — no consistent K-gain at K=100 (K=10 showed +1.8pp in a single run, unverified) — and the full pipeline was re-run **from scratch** (same scripts, same data, fresh training; logs in `src/results/logs/`). Two independent runs agree in direction (K=100 at or below baseline); per the pre-registered statistical protocol (≥5 seeds, CI), seed repeats are pending — all curves reported so far are single-run. A negative result that survives robustness checks is an asset; the framework treats it as one.

## 4. Why this is persuasive (evidence)

1. **The criteria predate the results.** The criteria and failure-handling clauses were fixed before any training run; M1 result reports are appended to the proposal and clearly separated (extracted to docs/results-m1.md). The framework is a contract, not a post-hoc rationalization of outcomes.
2. **Independently checkable.** Proposal → scripts → raw result JSONs → training/eval logs are all in one public repository; the domain-B (M1) pipeline is end-to-end reproducible on a single consumer GPU (~8GB VRAM, no commercial APIs, no closed models; domain A requires a gitignored pretrained checkpoint — see README Running).
3. **Negative results are kept, not hidden.** 14+ raw result JSONs including failures; the README reports the flat K-curves in the headline section and the non-monotonic 10000-step run in the results mapping table.
4. **It already survived one hostile test.** The first mechanism (parallel exploration) failed against the framework's own pre-registered criteria, and the framework localized exactly why. The negative result is interpretable rather than merely disappointing.

**Positioning against existing work.** Forward transfer (FWT) was proposed in the continual-learning community as early as GEM (Lopez-Paz & Ranzato, 2017); frameworks like Avalanche (Lomonaco et al., 2021) and benchmarks like CORe50 provide infrastructure and retention-focused suites. This framework differs along three axes: (i) it pre-registers the criterion and the failure handling *before* training, following the registered-report tradition (Nosek & Lakens, 2014); (ii) it measures data efficiency (T(n): samples needed per new domain) rather than accuracy transfer (FWT); (iii) it mandates a failure-localization protocol that turns negative results into mechanism-level evidence. The claim is not that these ideas are new in isolation — it is that they are combined into one executable, checkable protocol for the learning-mechanisms question.

## 5. How to use it

Five steps, each mapped to concrete files:

1. **Pre-register** criteria + per-layer failure handling → copy the structure of `docs/experiment-proposal.md`.
2. **Define the signal** (T(n) / K-gain / AUC) before training.
3. **Build the baselines** (naive fine-tuning / pure noise / majority vote).
4. **Localize failures** with the three-round protocol (same-rollout → repair → perturbation).
5. **Extend for robustness** (longer training + independent re-run).

If your mechanism produces sequence acceleration where our exploration mechanism produced a flat curve, that is a positive result directly comparable to this repo's negative one (same protocol, same signal definitions).

## 6. Boundaries and known limitations

- **Coverage**: the main experiment has run on two domains (A: Sudoku reproduction, B: RPN arithmetic); C/D generators and E-candidate features are delivered but the sequence has not been executed beyond B. T(n) — the framework's cross-domain signal — has zero data points so far. The framework's claims are about *how to measure*, not about what the full 7-domain sequence will show.
- **Adaptive task generation is a roadmap item, not yet exercised**: the delivered generators (B/C/D) make on-demand extension possible in principle; we have not yet run a full adaptive-task loop.
- **Scope**: this framework is for the *learning-mechanisms* question (efficiency/retention/continuity under matched compute). It is not a general AI evaluation framework (no claim about capability ceilings, alignment, or social benchmarks — those live in sibling documents).
- **Trivial-solution set is minimal, not exhaustive**: naive fine-tuning, pure noise, and majority voting rule out the common cases; a determined adversarial baseline could always be added. The pre-registration contract makes such additions explicit rather than silent.

## 7. Relationship to other documents

| Document | Role |
|---|---|
| `docs/experiment-proposal.md` | Frozen pre-registration contract (criteria + failure handling) |
| `docs/results-m1.md` | The worked negative result this framework produced |
| `docs/why-not-measure-faster.md` | Position paper: why continual learning should measure "getting faster" |
| `docs/skill-definition.md` | Operational definition of skill (behavioral disposition) |
| `README.md` | Status, reproduction guide, result mapping |

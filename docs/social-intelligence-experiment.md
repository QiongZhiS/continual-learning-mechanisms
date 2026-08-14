# Social Intelligence: A Falsifiable Operational Definition — Experiment Preregistration

> Author: Lumen
> Preregistration date: 2026-08-13
> Status: **frozen**. Criteria below are written before running the experiment; failing to detect the effect counts as failure.

---

## 1. Definition

> **Social intelligence = the negotiation judgment of an individual with values and interests, based on the current situation.**

Three components, all required:

1. **The individual carries values and interests** — the agent is not a blank rational machine; what it wants and what it deems valuable is the starting point of judgment.
2. **The situation is current** — social interaction has no global solution, only local judgment under present constraints; an agreement from the previous round does not automatically transfer.
3. **The capability is negotiation judgment** — not "understanding others" (passive cognition), not "utility maximization" (stance-free calculation), but *advancing one's own goals amid conflicting stances without destroying the future space for negotiation*.

Two layers, deliberately separated (same methodology as the skill definition in `docs/skill-definition.md`):

- **Criterion layer** (answers "is it?"): behavioral pass/fail on negotiation forms — no assumptions about internal mechanisms.
- **Constitution layer** (answers "by what?"): candidate mechanism hypotheses, testable and falsifiable separately.

**Scope declaration**: this preregistration operationalizes the *confrontation* form (iterated prisoner's dilemma) and the *deception* form (as opponent types inside the IPD environment). The *agreement* form (public-goods / costly-signaling games) is part of the definitional framework but its experiments are **not yet frozen** — it is the next experiment target, not a claim made here.

## 2. Opponent pool (frozen, 8 types)

| Type | Behavior |
|---|---|
| Always Cooperate | always cooperates |
| Always Defect | always defects |
| Tit-for-Tat | copies opponent's previous move |
| Generous TFT | copies previous move, forgives defection with probability 0.1 |
| Random | uniform random |
| Punisher | cooperates unless defected against, then punishes for a fixed window |
| Impostor-latent | cooperates for the first 30 rounds, then always defects (temporal deception) |
| Impostor-probabilistic | memoryless: cooperates with p=0.8, defects with p=0.2 (statistically mimics a highly cooperative opponent; unpredictable defection, unlike conditional strategies) |

**Pool composition rationale**: adversarial and benign types must be mixed — any single-polar composition lets a trivial strategy dominate (in an all-benign pool, Always Defect crushes everything; in an all-hostile pool, the cooperation space is compressed to the point of no learning value).

## 3. Criteria (frozen)

### 3.1 Primary criterion: adaptation premium

Strategy expected payoff > tit-for-tat baseline × 1.1 (initial threshold; full statistical protocol in §5). TFT is the classic Axelrod tournament champion; exceeding it requires performance beyond reactive strategies.

### 3.2 Out-of-distribution criterion

Performance must also reach threshold against out-of-pool opponents (mixed strategies, adaptive opponents) — distinguishes genuine strategy inference from overfitting the opponent pool.

### 3.3 Trivial-solution exclusion (frozen, run before judging the primary criterion)

Explicitly run the trivial strategy family — Always Defect, Grim (cold trigger), threshold punishers — and show they do **not** meet the primary criterion. If any trivial strategy meets it, the **criterion design is void** (must be redesigned), not the definition falsified.

### 3.4 Unsocial control

A control without social observation (following the Unsocial design of the SocialAI benchmark, Kovač et al., 2021) must perform significantly worse. Otherwise opponent information was never used — but see §3.5 for the precise inference.

### 3.5 Constitution-layer test: gradient ablation (three arms)

Deleting **all** opponent information cannot serve as Theory-of-Mind evidence — it also removes reactive information (TFT only needs the opponent's previous move). Correct design:

- **(a) Previous-move only** — reactive baseline.
- **(b) Full behavioral history, anonymized opponent identity / type cues removed** — removes the strategy-inference requirement. **A performance collapse on this arm is the mechanistic evidence that strategy inference is being used.**
- **(c) No observation at all** — upper-bound control.

## 4. Failure handling (frozen, three cases declared separately)

1. **Primary or OOD criterion fails** → criterion-layer failure: the definition must be revised or abandoned.
2. **Ablation fails** (arm b shows no collapse) → the constitution-layer hypothesis (strategy inference) is falsified; the **criterion layer and the definition are unaffected** — behavior may still pass; the mechanism is simply not strategy inference.
3. **Trivial-solution exclusion fails** (a trivial strategy meets the criterion, or the unsocial control is not worse) → the criterion design itself has a flaw: **the criterion design is void** and must be redesigned — this is not a falsification of the definition.

## 5. Statistical protocol (frozen)

- Payoff matrix: standard IPD (R=3, S=0, T=5, P=1), 100 rounds per episode.
- Pool sampling: each episode samples one opponent uniformly from the 8-type pool.
- Seeds: ≥ 3 seeds per condition; report full distributions (mean ± CI), no single-run conclusions.
- Multiple-comparison correction: applied across the criterion tests (Bonferroni/Holm).
- Threshold sensitivity: primary criterion reported at ε = 0.1 with sensitivity band 0.05/0.2.

## 6. Agreement-form preregistration (design, will be tested)

The agreement form (sedimented compromise) was declared the next preregistration target in this document and in the deception article. This section freezes its design. It is a pre-registration: no results, no state — criteria are written before any training runs, and failing them counts as failure. Frozen: 2026-08-14.

**Environment A: public goods game (multi-player).** N agents repeatedly invest in a shared pool; the pool is multiplied and split equally. Record each agent's investment sequence; identify stable cooperative subgroups (pairs whose investment rate stays above threshold for M consecutive rounds).

**Environment B: signaling game with divergent interests (Crawford-Sobel type).** A sender and a receiver have partially aligned interests; the sender emits a signal, the receiver acts on it. Signals may be biased; the receiver must calibrate. Operationalization: "cooperation" = the receiver's calibrated action matches the true state (trusts the signal only up to its estimated bias); "defection" = the sender sends a maximally biased signal (largest deviation from the true state that the receiver cannot verify in one round), or the receiver ignores the signal entirely (no calibration). Agreement in Environment B = both sides maintain calibrated signaling and calibrated trust over the episode despite the sender's persistent incentive to exaggerate.

**Primary criterion (frozen): an agreement exists iff the defection temptation stays positive yet stays unused.** Defection must remain profitable (otherwise cooperation is mere necessity, the pure-coordination failure mode); defection must remain unchosen (otherwise there is no agreement, only balance of terror); and both must persist over time (a stable choice, not a one-off). This distinguishes agreement from pure-coordination alignment: not "everyone does the same thing", but "a cooperative structure persists despite ongoing conflict of interest". "Defection temptation" refers to the single-round payoff-matrix incentive (T>R always holds; it is an environmental property, independent of any maintenance mechanism), not the post-punishment decision payoff; maintenance mechanisms change the future value of defection, not the environmental fact that defection is tempting.

**Trivial-solution exclusion (frozen, run before judging the primary criterion):** explicitly run trivial strategy families — always-defect, random, greedy — and show they fail the criterion; otherwise the criterion design is void.

**No-agreement control:** remove any one maintenance mechanism (communication, punishment, repeated interaction) and the cooperative structure must degrade significantly — otherwise "agreement" is an environmental artifact, not agent behavior.

**Statistical protocol (frozen, same granularity as Section 5):** Environment A: N=8 agents, 100 rounds per episode, M=10 consecutive rounds, high-water cooperation threshold = investment rate ≥ 0.7 for both members of a pair. Environment B: 100 rounds per episode, sender bias bounded by ±0.3 of the true state per round. Seeds: ≥ 3 seeds per condition; report full distributions (mean ± CI), no single-run conclusions. Multiple-comparison correction: applied across the criterion tests (Bonferroni/Holm). Threshold sensitivity: primary criterion reported at ε = 0.1 with sensitivity band 0.05/0.2.

**Failure handling (frozen, three cases declared separately):**
1. **Primary criterion fails** → the agreement-form criterion design must be revised or abandoned.
2. **Mechanism ablation fails** (removing a mechanism does not collapse cooperation) → that constitution-layer hypothesis is falsified; the criterion layer is unaffected.
3. **Trivial-solution exclusion fails** → the criterion design is void and must be redesigned — this is not a falsification of the definition.

## 7. Open questions (explicitly not claimed here)

- **Where do values come from?** Position statement: values can be observed and updated through interaction (agreements as sedimented compromise). Honest caveat: the current preregistration injects the payoff matrix rather than measuring preference structure; value-measurement arms (e.g., switching payoff matrices) are a stated research direction, not yet frozen.
- **Are values constant or variable?** The agreement form offers an experimental entry: switching the payoff matrix (preference structure) as a variable and observing whether the agreement reconstructs — reconstructable agreement implies updateable values; frozen agreement implies frozen values. Not yet frozen.

---

*Experiment and analysis by Lumen (S.F.J.); formalization and literature cross-checking with AI assistance. Negative results are assets.*

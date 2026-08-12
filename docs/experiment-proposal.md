# AGI Current-Paradigm Experiment Proposal: Can Learning Mechanisms Produce Continual Learning — A Falsifiable Test

> Positioning: **not a blueprint for building AGI, but a falsifiable experiment testing whether "learning mechanisms" can produce continual learning.**
> All conclusions must land within a pre-registered finite window; falsification is equally valid scientific output.

---

## 0. Abstract

**Problem**: the learning-curve slope of the current deep learning paradigm is determined mainly by **scale** (bigger models, more data, more compute), not by **learning mechanisms** (how exploration, memory, and experience reuse are organized). This experiment uses ~7M-parameter micro-models at a fully affordable compute budget to test a binary hypothesis: **under matched compute budgets, can a system with complete learning mechanisms exhibit "continual learning" (acceleration + retention + no retraining) on a cross-domain task sequence, where the plain fine-tuning paradigm structurally fails?**

**Method**: train a latent-state iterative recurrent model (TRM, ~7M parameters, runs on a single consumer GPU); validate five learning-mechanism components (parallel exploration / surprise memory / experience distillation / vector composition / counterfactual causality) one by one on four training domains (logic puzzles → symbolic rules → feature composition → discrete games); then run a pre-registered 7-domain sequence in a three-arm controlled comprehensive test.

**Criterion**: continual-learning signature = transfer acceleration (T(n) ≤ 0.5 and non-increasing) + retention (no collapse of previous-domain knowledge) + continuity (single weight set, no retraining), significantly better than a budget-matched plain fine-tuning arm.

---

## Experiment-number index

This document uses pre-registered experiment numbers (E0–E9) and milestone numbers (M0–M5):

| Number | Experiment / milestone | One-liner |
|------|------------|--------|
| **E0** | Base | Reproduce TRM baseline and parallel-expansion gain |
| **E1** | Exploration mechanism (core mechanism 1) | Does parallel expansion + quality selection + failed-trajectory sharing improve solving? |
| **E2** | Memory writing (confirmation) | Does surprise-triggered writing improve cross-task retention? |
| **E3** | Experience distillation (core mechanism 2) | Extraction of reusable skills from episodic experience (five plans A–E compared) |
| **E4** | Compositional generalization (vector addition) | New entity = weighted sum of known features |
| **E5** | Causal layer (counterfactual intervention) | Counterfactual intervention > pure prediction for world-model calibration |
| **E6** | Comprehensive validation (AGI fingerprint) | Continual-learning signature on an N-domain sequence (acceleration + retention + continuity) |
| **E7** | Interaction-effect measurement | Super-additivity detection for module combinations |
| **E8** | Candidate mechanisms | Exploratory experiments off the critical path |
| **E9** | Self-cognition track (independent) | Train a continuously updated model that knows itself |
| **M0** | Base-reproduction milestone | E0 + feasibility validation (incl. domain-sequence generator) |
| **M1** | Exploration-mechanism milestone | E1 criterion test (does K-expansion hold in the symbolic domain?) |
| **M2** | Memory & composition milestone | E2 retention + E4 compositional-generalization criteria |
| **M3** | Distillation milestone | E3 criterion test (does T(n) decrease?) |
| **M4** | Comprehensive milestone | E5 causality + E6a/E6b sequence + E7 interaction criteria |
| **M5** | Real-domain scaling (long-term track) | Transfer the best configuration to text/tool domains |

Suffixed numbers (E0b/E0d/E2b/E6a/E7b, etc.) are variant arms of the same experiment (perturbation sweep / structure growth / decay curve / E6a three-arm sequence control / immune neutralization). Milestone-to-experiment mapping: see §7.

---

## 1. Positioning statement

- **Experiment goal** = validate a mechanism (does learning-curve acceleration appear), not build a product
- **Starting decision** = begin with training a micro recurrent model (~7M parameters) — the training process itself is the observation window into learning mechanisms; inference cost ≈ $0.001/query; full-factorial experimental design is economically feasible
- **Success definition** = the continual-learning signature appears within the pre-registered window (N-domain sequence), or is clearly falsified
- **Ultimate criterion** = continual-learning signature + compositional generalization (platypus test)

### Physical-boundary statement

This experiment validates the **scheduling/abstraction layer** of learning mechanisms on digital hardware + gradient descent (can experience→skill extraction and reuse produce transfer decay); it does **not claim biological-level "reasoning-as-plasticity"** (millisecond-scale local plasticity) or compute-in-memory — those require neuromorphic hardware and are out of scope. Designs such as "nightly fine-tuning" (see §6 E3, plan D) are **engineering approximations** (scheduling semantics) of higher-frequency offline incremental learning, not physical implementations.

### Non-provability statement

"Continual learning" = the same system accumulating capability over an **unbounded** domain sequence, a property over infinite sequences — any finite experiment can only **falsify** (degeneration within the pre-registered window), never **prove** (passing within the window ≠ holding for arbitrary-length sequences; isomorphic to the "AGI fingerprint" problem). Continual learning is therefore positioned as a **goal (north star)** rather than a provable proposition. The scientific output of this experiment is limited to three items:

1. The test result of the continual-learning signature within the pre-registered window (N domains) — **falsifiable**
2. When in-window acceleration is stable, T(n)-curve extrapolation reported separately as a **prediction** (not masquerading as a conclusion; extrapolation interval labeled)
3. Degradation-mode taxonomy (forgetting-type / stagnation-type) — even under falsification, this produces mechanistic information about "where continual learning breaks"

**"Cannot prove" is a constraint on criterion design**: all criteria must land within the falsifiable window; extrapolation is only an appendix note.

---

## 2. Paradigm background and external evidence

### 2.1 Paradigm-shift hypothesis

This experiment tests not only "can learning mechanisms produce continual learning," but the **micro-conditions of a paradigm shift (static optimization → dynamic game)**. Five bottom-level assumptions of the mainstream paradigm are breaking:

| Bottom-level assumption | Break direction | Existing answers across fields | This experiment's landing point |
|---------|---------|------------|-----------|
| Differentiability | smooth → discrete hybrid | AlphaZero MCTS+NN / MoE discrete routing | E1 parallel expansion (discrete search) + TRM base (differentiable) |
| Static (train/inference separation) | offline → online | Adaptive control: Kalman/RLS + persistence-of-excitation theorem | E3 plan D nightly fine-tuning (first online step), E0b online perturbation |
| Single objective | optimization → equilibrium | Mechanism design: payoff matrices instead of scalar reward | E7b immune neutralization arbitration (candidate) |
| Single agent | individual → ecology | Immune negative selection + Red Queen hypothesis | E0d structure growth, E1b wanderer retention (candidates) |
| Infinite memory | storage → lossy allocation | Rate-distortion theory: forgetting = optimal lossy allocation | E2b decay curves |
| Cognition (self-model) | passive → active | Active inference / predictive coding | E9 self-cognition track (candidate) |

**Primary-breakthrough judgment**: dismantle the train/inference separation first. Reasons: differentiable-discrete hybrid architectures already exist; control theory provides ready-made mathematics for online updates; QLoRA/nightly fine-tuning is already engineering-feasible — E3 plan D is the first step.

**Kuhnian criterion**: once learning is online, "catastrophic forgetting" is redefined from a hard problem to be solved into a "controller stability problem" — problem-domain redefinition is the signature signal of a paradigm shift.

### 2.2 External evidence chain

**RLM (Recursive Language Model, MIT, 2026-08)**: RL training on a 30B model. Same base, same data, same algorithm; the only variable = the context organization of the Harness (the program layer outside the model). The bare model trains with higher reward but flatlines on long-task evaluation (zero generalization); with the Harness — 64k-token training context, evaluation stretched to 2M (32×) — scores rise with training reward. **The same model, 8–32× generalization gap, the only difference being "the program layer outside the model."**

Mapping to this experiment:
- RLM's "local in-distribution" ↔ the boundary-aware redirection of the domain-switch protocol — both are "don't force inference outside the uncertain distribution"
- RLM's "equivalence-class folding" (tasks with identical solutions folded into one trajectory) ↔ E6a's domain-sequence transfer gradient — if T(n) is non-increasing, the scheduling layer has found equivalent structure across domains
- RLM's "shortcut → self-discarding → better generalization" ↔ E1's design intent: discard domain-specific shortcuts, retain domain-general capability (to be tested by E1) — domain-specific things were not learned, domain-general things grew by themselves; the same mechanism observed independently at two scales
- RLM's limitations (tasks inherently chunkable, conclusions from 30B) ↔ this experiment's complement (game domain tests "tightly interlocking subproblems"; the 7M base provides an "extremely weak base" control)

**Brain-science cross-reference**: MIT aphasia research (reasoning does not require language areas) + inverse-model research (noise + inverse model = structured exploration) + ghost points (stagnation near the critical point of a saddle-node bifurcation; learning is a bifurcation phenomenon) + sleep phase transitions (destabilization–restabilization) — all point to the same conclusion: **intelligence is bottom-up dynamics + structured exploration + dual-mode memory + critical switching; language and action are later passengers.** (Sources per direction: see the §10 brain-science cross-reference entry)

---

## 3. Hypothesis system

### H0 (null): the current paradigm (stateless + scaling + in-context learning) suffices for AGI

Falsifiable statement: the existing paradigm (in-context learning + plain fine-tuning + retrieval), at **matched compute budget**, achieves the same continual-learning signature (acceleration + retention + single-weight continuity) as the mechanism system.

Prediction: the mechanism system's continual-learning signature on the domain sequence is **not better than** the budget-matched plain fine-tuning control. Classical catastrophic-forgetting prior (McCloskey & Cohen 1989): plain fine-tuning should collapse in retention at the tail of the sequence — **if plain fine-tuning does not collapse and catches up with the mechanism system, that is the true H0 support** ("any learning can continually learn" is more counterintuitive than "all learning forgets" and requires stronger evidence).

If H0 holds → continued scaling/fine-tuning suffices; this experiment's architecture has no value.

### H1 (alternative): the learning-curve slope depends on learning mechanisms, not scale

Learning mechanisms = the coupling of five components: **parallel exploration + surprise memory + experience distillation + vector composition + counterfactual causality**.

H1 is split into two abstraction levels, measured separately:
- **H1a (scheduling-policy layer)**: on a fixed-weight base, can information-flow management policies (exploration/memory/distillation/composition/causality) produce continual learning — **the main validation target of this experiment, directly measurable on digital hardware**
- **H1b (mechanism-substance layer)**: is continuous weight reshaping (online plasticity) itself necessary — measured indirectly via the E0b perturbation-strength sweep + E0d structure-growth track

> Honesty statement: this experiment can falsify/support H1a; H1b provides only indirect evidence (if weak online perturbation beats all offline scheduling → the mechanism layer matters; if offline scheduling wins → the scheduling layer suffices under the current hardware paradigm).

Prediction: the system with complete learning mechanisms exhibits the continual-learning signature on the cross-domain sequence — (a) acceleration: T(n) (data needed by domain n to reach threshold ÷ domain 1) non-increasing in n and ≤ 0.5 (pre-registered threshold); (b) retention: retention R(n) ≥ threshold for any previous domain; (c) continuity: single-weight continuous learning throughout, no domain-level retraining/reset; significantly better than the plain fine-tuning arm at matched budget.

### Key corollaries (each an independently falsifiable point)

| Corollary | Content | Experiment | Frontier anchor |
|------|------|---------|---------|
| 1 | Parallel exploration + quality selection is significantly better than single-path deterministic inference even under **matched-compute control** | E1 | PTRM (upstream paper, see §10): 62.6%→91.2% at K=100 |
| 2 | Surprise-triggered writing (memory) improves cross-task retention, but is **not** the main driver of learning acceleration | E2 | Titans: BABILong beyond 2M context |
| 3 | **Experience→skill distillation is the bottleneck of learning acceleration** (unknown maximum) | E3 | SkillsBench: curated skills +16.6pp (33.9%→50.5%) / self-generated null |
| 4 | Vector composition (new entity = weighted sum of known features) achieves compositional generalization | E4 | Brain science: linear feature superposition (fMRI-validated) |
| 5 | Counterfactual intervention > pure prediction (correlation) for world-model calibration | E5 | Causal-JEPA (see §10) |
| 6 | The integrated system continually learns over an N-domain sequence: acceleration + no forgetting + no retraining | E6a | McCloskey & Cohen 1989: catastrophic forgetting of standard fine-tuning on long sequences |

---

## 4. Experiment overview

```
E0 micro-model base (TRM training · prerequisite · E0b online-plasticity four arms + E0d structure-growth track)
 ├─ E1 exploration mechanism (core mechanism 1: parallel expansion + quality selection + failed-trajectory sharing)
 ├─ E2 memory writing (surprise writing on/off + decay curves)
 ├─ E3 experience distillation (core mechanism 2: episodic→skill · five plans A–E)
 ├─ E4 compositional generalization (vector addition · platypus test)
 ├─ E5 causal layer (counterfactual intervention on/off)
 ├─ E6 comprehensive validation (E6a continual-learning sequence three arms = H1 criterion / E6b human-collaboration = independent conclusion)
 └─ E7 interaction-effect measurement (stepwise accumulation + 2×2 factorial · attribution path)
```

---

## 5. Infrastructure

### 5.1 Base (E0)

- **Model**: TRM (latent-state iterative recurrent model, ~7M parameters) + PTRM extension (parallel stochastic expansion + quality selector)
- **Hardware**: single consumer GPU (RTX 4060 8GB, 8GB VRAM); the full experiment is reproducible on this configuration
- **Why small models**: ① training is the experiment — every change to a learning mechanism shows directly in the training curve ② cost allows full-factorial experimental design ③ latent-space dynamics are observable (good/bad basin visualization)

### 5.2 Training domains

| Domain family | Content | Capability tested | Complexity gradient |
|------|------|-----------|-----------|
| A puzzle | Sudoku/logic puzzles | Reasoning + exploration mechanism | low |
| B symbolic | Pseudo-programming language (different syntax/semantics) | Rule learning + composition | medium |
| C compositional | Feature→entity inference (color×animal→coordinates) | **Compositional generalization (platypus test)** | medium |
| D game | Discrete-state games (prisoner's dilemma / public goods / signaling games) | Social prediction / strategy inference | high |

Domain D is designed as a purely discrete-state game: input/output format identical to the first three domains (state vector → action), no external LLM interface — avoiding extra engineering complexity from interface adapters and preventing designer-prior injection from polluting the "pure-system autonomy" criterion. Domain D's concrete instances (opponent strategies, round counts, payoff matrices, observation structure) are locked in a pre-registration appendix.

**Domain-switch protocol (boundary-aware routing)**: when switching domains, if the system's confidence about "which domain/rule set it is in" is low (< threshold, threshold frozen at pre-registration), it **actively requests human redirection** instead of forcing a solution — the human answer enters memory as a high-weight correction signal. Rationale: shift the detection cost of "not knowing where it is" onto human intuition (asymmetric complementarity).

**Dual-track isolation**: the redirection switch of the domain-switch protocol runs on **two independent tracks** throughout E1–E5 — track 1 (redirection on → feeds E6b); track 2 (redirection off, force-solving at blurred boundaries → feeds E6a). This prevents train/test distribution mismatch: if experience is accumulated only with redirection but tested without it, an E6a failure could not distinguish "mechanism not strong enough" from "never learned unassisted force-solving."

### 5.3 Evaluation metrics

| Metric | Definition | Purpose |
|------|------|------|
| In-domain learning curve | training steps → accuracy | learning speed per domain |
| Transfer coefficient T(n) | data needed by domain n to reach threshold ÷ domain 1 | core criterion (continual learning · acceleration component) |
| Retention matrix R(i,j) | retention of domain i after learning domain j (lower triangle = forgetting pattern) | core criterion (continual learning · retention component) |
| Basin-escape rate | success rate of escaping bad basins | E1 criterion |
| Distillation product quality | skill-library hit rate + post-reuse performance gain | E3 criterion |
| Compositional-generalization accuracy | inference accuracy on unseen combinations (platypus test) | E4 criterion |
| Autonomous rule discovery | behavioral probe + rule-matching score + random-baseline control | E6a criterion |

### 5.4 Statistical norms and pre-registration

With a 7M model and few domains/samples, noise dominates; a single run can misjudge H1's fate, so "significant" requires a statistical contract:

- **Seed count**: ≥ 5 random seeds per experimental condition; report mean ± CI; no single-run conclusions
- **Effect size**: "significantly better" = effect size ≥ pre-registered threshold (default Cohen's d ≥ 0.8 or median ratio ≤ 0.5) and CI excluding zero
- **Multiple comparisons**: family-wise error control across the E1–E7 experiment family (default Bonferroni or Holm correction)
- **Hyperparameter freezing**: all mechanism hyperparameters are tuned only on domains A–D; after freezing, domain E is run with a limited seed set — tuning on domain E is forbidden, preventing post-hoc selection
- **Domain-sequence pre-registration**: E6a's sequence length N (default 7), inter-domain similarity gradient, and per-domain interaction budget are locked in the pre-registration file — "sequence order" and "inter-domain distance" are hidden variables; post-hoc adjustment = criterion contamination
- **Operational definition of "expert level"**: per-domain pass threshold = accuracy ≥ domain ceiling × pre-registered coefficient (default 0.8). Ceiling = the posterior best performance of a mechanism-free baseline model under ample budget + multi-configuration search; **the calibration process is forbidden from using any mechanism component** (preventing the "calibrate mechanisms with mechanisms" circular dependency). Threshold sensitivity analysis: recompute the criterion for coefficients in 0.7–0.9; report only if the conclusion is robust over that range
- **Operational definition of "autonomous rule discovery"**: ① behavioral probe (feed instances unseen in-domain, check whether outputs consistently reflect the rule) + rule-matching program + random-baseline control; ② latent-space geometric channel (spatial separability of rule-corresponding clusters). The two channels are scored independently; either one passing suffices

### 5.5 Test environment and scale

- **Hardware**: same as §5.1
- **Model scale**: TRM ~7M parameters (latent-state iterative recurrent model); inverse-model candidate ~2M
- **Data scale**: domain sequence N=7 (see §5.4)

---

## 6. Detailed experiment designs

### E0 Base

- Reproduce TRM/PTRM: complete training and baseline reproduction in the logic-puzzle domain; verify the K=100 parallel-expansion gain is reproducible
- **Architecture choice**: TRM is a latent-state iterative recurrent model (no self-attention) — this base choice is consistent with the route judgment that "attention is not the bottom-level mechanism" (attention's relational binding can be implemented with cheaper iterative mechanisms); Transformers serve only as comparison references
- Deliverables: a trainable base + latent-space visualization tools (good/bad basin maps)

**E0b online weight-perturbation sweep (one carrier for measuring H1b)**: apply in-place micro-perturbations to the last-layer weights after each inference, **without fixing a single magnitude** — run a perturbation-strength sweep (1e-5 → 1e-1, log-spaced) and plot "perturbation strength vs. transfer gain." A single-point experiment cannot distinguish "insufficient strength" from "principle ineffective"; the sweep removes that ambiguity. Four control arms form a complete **online-plasticity spectrum**:

1. No update (baseline)
2. Random-noise perturbation (pure regularization/randomization control)
3. Directional-perturbation arm (perturbation direction driven by surprise signal / gradient approximation)
4. **TTT arm**: inference-time finite-step gradient updates driven by a self-supervised objective (predict next latent state/output), sweeping update steps (1/2/4) × learning rate (1e-5 → 1e-2)

Reading: only when the perturbation arm has extra gain over the random arm is it attributed to the "plasticity principle"; TTT arm significantly better than the directional arm → plasticity needs objective-driven updates; TTT arm with no gain over the full range → online weight updates have no value on the current base. **Cost labeling**: the extra compute of direction signals/TTT is counted into the control; if extra compute > 10% of the random-perturbation arm, the conclusion is downgraded (principle gain vs. budget gain indistinguishable). TRM's serial iterative structure is isomorphic to "update during inference" — the strongest measurable carrier of H1b on this base.

**E0d structure-growth track (independent track · exploratory, outside the main criterion chain)**: a fully parallel environment to the main experiment — the base is allowed **structure-level changes during training** (split units by novelty / prune connections by low utilization), measured against the same continual-learning criteria as the main experiment, and compared with the configuration selected by pre-registered criteria in the main experiment. If the base can grow during training, the "fixed base + scheduling policy" main criterion of E1–E7 is contaminated — hence full isolation. Criterion: E0d signature significantly better than the main experiment's selected configuration → structure-level plasticity > scheduling layer; otherwise → the scheduling layer suffices at the current scale.

### E1 Exploration mechanism (core mechanism 1)

**Variables**: parallel-expansion count K (1/10/100) + random-noise injection on/off + quality selector (q-head) on/off + cross-expansion failed-trajectory sharing (taboo index) on/off.

**Matched-compute control (necessary condition for corollary 1)**: best-of-K gain is inseparable from inference budget — K=100 vs. K=1 changes both algorithm and compute. Add two controls: ① matched-budget sampling baseline (same model, same total inference budget, best-of-K without sharing or taboo index) ② matched-budget single-path baseline (same budget, longer search steps). Corollary 1 holds only if "K-expansion + failure sharing" significantly beats ① and ② — otherwise the conclusion is downgraded to "spending more inference compute works."

**Failed-trajectory sharing**: parallel expansion + selection alone = brute-force parallel search + sample selection — 100 trajectories evolve independently; a shortcut found by trajectory 50 is not passed back to trajectories 1–49. **The complete version = parallel expansion + cross-expansion failed-trajectory sharing**: each expansion's failed-trajectory cluster (walls hit) is written into a shared taboo index; subsequent expansions preferentially avoid them — after failure information is asynchronously fed back, it constitutes a cognitive iteration of "coming back with the cause of death." The taboo index reuses the same component as E3 negative distillation (a single switch factor, preventing duplicate writes/weight stacking).

**M1 result (negative result, mechanism localized)**: the E1 main criterion failed in the symbolic domain — inference-time exploration is structurally ineffective for problems that are "uniquely solvable but not yet learned." Full results and redesign directions: [docs/results-m1.md](results-m1.md).

### E2 Memory writing (confirmation experiment)

- **Variable**: surprise-triggered writing on/off (remember surprising samples; baseline writes everything)
- **Measurement**: cross-task retention (knowledge retention of domain A when testing domain B after training on A)
- **E2b decay curves**: after domain-A training completes, measure retention at intervals of 1/3/7/30 days (surprise writing on/off); fit "decay time constant vs. writing mechanism" curves; extend to the cross-domain retention matrix R(i,j) — direct evidence of continual learning vs. catastrophic forgetting, and mechanism-level attribution for E6a's retention criterion (which mechanism protects which history)
- **Criterion**: surprise writing is significantly better than full writing in memory-capacity efficiency (no retention degradation); the decay time constant is calibration-type output without a significance gate
- **Positioning**: confirm that the storage layer is solved and is not the main battleground of learning acceleration — if it passes, the storage-layer hypothesis is confirmed and E2 has fulfilled its mission

### E3 Experience distillation (core mechanism 2 · the biggest unknown)

**Variable**: distillation mechanism on/off — an extractor from episodic experience (sample level) to skills (reusable rule level). The five plans are **independent controlled trials**: only one is on at a time, never stacked:

- **Plan A: rule distiller** (extract if-then rules from successful trajectories) — essentially decision-tree pruning; compresses distributed representations into symbolic logic; boundary cases are lost. Expected effective on puzzle domains, degrading to noise on interactive domains. Plan A is positioned as an exploratory control (not in the main criterion)
- **Plan B: vector distiller** (aggregate successful-experience representations into skill vectors) — needs selection-bias correction: clustering on q-head self-selected "successful" subsets converges to "the easiest-to-cluster success," not "the most transferable rule." Correction: cluster input changed to **full-trajectory representations** (success + failure); labels used only for weighting
- **Plan C: LLM-assisted distillation** (control — SkillsBench has shown LLM self-written skills give zero gain; expected failure, used to confirm the baseline)
- **Plan D: nightly incremental fine-tuning** — daytime: inference only, weights untouched, high-confidence errors recorded; nighttime: generate training pairs from error records, offline LoRA/Adapter incremental fine-tuning of the base (only the last few layers), load the new Adapter the next day. Human review step: compare "before vs. after" test-set differences to decide keep/rollback. Advantage: completely bypasses the "inference-as-training" hardware bottleneck, retaining rollback/auditability/visible diff. **Includes a "phase-transition reset" step**: before nightly training, inject Gaussian noise into the last few layers + brief free exploration (simulating sleep's destabilization→restabilization), then converge to a new basin — measure T(n) difference with/without noise perturbation; if T(n) decreases with noise → continuous incremental fine-tuning was stuck in an old-basin local optimum, and the phase-transition reset released the exploration space
- **Plan E: negative distillation** — specifically records "trajectory clusters that led to failure" (walls hit) into a unified taboo index; the system **preferentially avoids** them when exploring new domains. Rationale: counterexamples carry ≥ weight of positive examples in generalization — "knowing what cannot be done" is an independent information channel

**Measurement**: skill-library hit rate, post-reuse performance gain, cross-domain reuse rate, **retrieval overhead / inference-step share** — a skill hit whose matching computation consumes >50% of inference steps = no net acceleration; "learning acceleration" must be defined on the total compute budget, not on isolated accuracy.

**Criterion**: any plan making T(n) drop significantly → corollary 3 supported; all fail → **experimental core conclusion: the learning-mechanism bottleneck is distillation; a new mechanism is needed**.

**Meaning**: this is the main uncertainty of the entire experiment — the frontier (SkillsBench) shows all current distillation methods fail; if this experiment also cannot solve it, AGI's "learning" needs a paradigm-level breakthrough, not component improvements.

### E4 Compositional generalization (vector addition)

- **Variable**: compositional representation mechanism — dual representation channels (high-dimensional feature preservation on the memory side vs. low-dimensional compression on the action side) vs. a single channel
- **Task**: the platypus test — training gives only "color→X-axis, animal→Y-axis" independent mappings; testing gives novel combinations (green elephant) requiring zero-shot coordinate inference
- **Trivial-composition defense**: in linear spaces, "adding feature vectors" is a trivial property of embeddings — ① task-family pre-registration: use structures **where trivial linear combination fails** (nonlinear feature interactions, new decisions needed after composition), or explicitly test a linearly solvable version as a control to prove the task is non-trivial ② plain-baseline control: criterion = dual channels **significantly better than a generic composition-capability baseline**, not merely better than the single channel itself
- **Compositional-distance gradient**: stratify the test set by "replacement distance from training combinations" (replace 1 feature → all); plot "zero-shot inference accuracy vs. compositional distance" decay curves; report the **failure distance d\*** (replacement distance at which accuracy falls below the random baseline) — falsifiable, comparable across models
- **Criterion**: dual-channel zero-shot inference significantly better than single-channel **and better than the trivial-composition baseline**, with failure distance reported; failure → compositional generalization needs stronger mechanisms (e.g., symbolic-neural hybrids)

### E5 Causal layer (counterfactual intervention)

- **Variable**: counterfactual training on/off — training includes intervention samples (intervene on variable A, observe the distribution change of B) vs. pure observation
- **Task**: "intervention vs. correlation" discrimination tests in a micro-world rule domain (Simpson's-paradox-style scenarios: treatment choice)
- **Criterion**: the counterfactual group is significantly better than the pure-prediction group on intervention estimation; failure → correlation learning suffices (for AGI), causal layer deferred
- **Meaning**: prediction-error minimization learns correlations (Simpson's paradox fools it); the causal layer decides whether the world model "understands" rather than "memorizes"

### E6 Comprehensive validation (AGI fingerprint)

> The domain-switch protocol (human redirection) does not enter E6a — after injecting the highest-quality external supervision, passing could only prove "human-collaboration works," not "the mechanism works." E6 therefore consists of two independent experiments:

**E6a continual-learning sequence test (independent criterion for H1a)**

- **Configuration**: configurations selected from E0–E5 by pre-registered criteria, **no human intervention allowed** (domain-switch protocol off; force-solving at blurred boundaries); **single-weight continuous-learning constraint** — no retraining, no reset throughout; memory/skill libraries writable but never cleared ("continuity" is an architectural constraint, not a verbal claim)
- **Procedure**: domain sequence [A→B→C→D→E→F→G] (N=7 pre-registered; E–G are new domain instances instantiated from the domain families by similarity gradient), ordered by the generator along a similarity gradient (near→far); domain E = the baseline-predicted lowest zero-shot-transfer one (selection happens before any mechanism training; not post-hoc selection); each domain switches immediately to the next upon reaching threshold (no extra budget). Measure T(n), end accuracy, and retention matrix R throughout. "Interactions" are uniformly converted to training steps (1 interaction = 1 gradient step, batch=128)
- **Three-arm control**: ① no-learning baseline (pure random search + deterministic inference without exploration) ② **plain fine-tuning arm** (same base, exactly the same A–D training history and data volume as the mechanism arm, all mechanism modules off, standard gradient fine-tuning after each domain's interactions — the faithful expression of the current paradigm in a continual-learning context) ③ mechanism-system arm
- **Expectation (based on McCloskey & Cohen 1989)**: ② should collapse in retention at the tail of the sequence. ② not collapsing and catching ③ = true H0 support; ② collapsing while ③ does not = the strongest evidence for the continual-learning signature (plain learning structurally fails in this paradigm, rather than "slightly worse")
- **Data-efficiency secondary criterion**: ② and ③ share the A–D training history; ②'s weights implicitly encode A–D structural knowledge — the main criterion measures a matched-budget competition between "mechanism-organized experience" and "implicitly encoded experience." If ② catches ③ → H0 support, but the mechanism's value proposition remains on the data-efficiency dimension: if the mechanism arm reaches ②'s domain-E performance with ≤ 50% of ②'s interactions → report "H1 holds on the data-efficiency dimension"
- **Latent-space evidence channel** (operational definition in §5.4 ②): measure the spatial separability of rule-corresponding clusters for domain E (inter-cluster distance, cluster purity vs. training steps)
- **Criterion (continual-learning signature)**: the mechanism arm satisfies within the window (a) acceleration: T(n) ≤ 0.5 and non-increasing (denominator = steps needed by baseline ① to reach expert level on the same domain, calibrated within the budget cap) (b) retention: R(n) ≥ threshold for any previous domain (c) continuity: no retraining throughout — **and ③ significantly better than ②**, to support H1a. ③ signature failure → **continual-learning signature falsified**: degradation-mode taxonomy report (forgetting-type = R below threshold / stagnation-type = T(n) rising), distinguishing "insufficient memory capacity" from "insufficient mechanism"

**E6b human-collaboration transfer (independent conclusion)**

- Configuration: same as E6a + domain-switch protocol enabled (actively requests human redirection at low-confidence boundaries)
- Criterion: E6a fails but E6b passes → conclusion is "human-machine coupling works," **not H1 support**; both pass → report the two contributions separately

### E7 Interaction-effect measurement

**Problem**: E1–E5 measure main effects single-variable; if the all-on combination fails in E6, attribution is chaotic — it cannot be determined which mechanism conflicts.

- **Stepwise accumulation**: E1 → E1+E2 → E1+E2+E3 → ... comparing T(n) — measure the incremental contribution of each added module; if a step is negative-gain → localize the conflicting pair, then run a 2×2 factorial on that pair (ANOVA interaction-term significance = the formal test of super-additivity)
- **Super-additivity detection**: stepwise accumulation measures marginal contributions, not synergy — if each component gives a small positive gain and all-on gives a large gain ("coupling" is precisely H1's definition), stepwise accumulation would miss it. When E6a fails, do two things instead: ① **all-on vs. best subset** — the best-performing subset in the stepwise sequence vs. all-on (all-on > best subset → synergy evidence; all-on ≈ best subset → some modules redundant) ② 2×2 factorial on suspect module pairs
- **Emergence detection**: supplementary reporting norm (no new experiments) — match the trajectory characteristics, strategy diversity, and responses to novel domains of the E6a all-on configuration against a behavior library of each single-on configuration. No precedent → integrated effect (supports the coupling hypothesis); precedent exists → all-on gain is just a combination of single effects. This is an auxiliary qualitative criterion; it does not affect the main signature

### E8 Candidate mechanisms (exploratory track)

**E1b wanderer retention (selector diversity constraint)** — forcibly retain ~10%-scale outlier individuals (currently poor performers but extremely far from all their own kind), reserving "living fossils" for drastic environment changes. Control-theory anchor: the persistence-of-excitation theorem (parameter convergence requires persistent excitation signals). Difference from random-noise injection: noise is a uniform, directionless perturbation source without memory; wanderer retention is **directed diversity under selection pressure**. Retention rate ρ ∈ {0, 0.05, 0.1, 0.2}. Falsifiable prediction: near-domain switching ρ>0 gives no gain or slight loss; **far-domain switching ρ=0.1 gives significant gain, monotonically increasing with inter-domain transfer distance**. No gain over the full range → random noise already covers the diversity requirement; the mechanism is redundant.

**E7b immune neutralization (output arbitration layer)** — biological immunity's consistency is not "antibodies think alike," but "output vectors can neutralize each other"; conflict is not an error but the system's homeostasis mechanism. When multiple detectors have a **high-confidence conflict** (candidate output vectors with large angles + close confidences), the main system takes neither side but generates a "third neutralizer" (conflict-representation interpolation + new-expansion verification). Three-arm arbitration strategies: ① best-of ② voting ③ neutralization. Falsifiable prediction: when conflict events occur, ③'s success rate is significantly higher than ①② → conflict is an information source; ③ ≈ ① → arbitration needs no special mechanism.

### E9 Self-cognition track (independent track · metacognitive layer)

> The second independent exploration track alongside E0d. Tests whether "self-cognition-driven continuous updating" produces a continual-learning signature independent of external task drive.

**Design**: S (the subject; TRM base; solves external tasks; continuously updated) + M_self (self-model; a lightweight prediction head on S). Training signal = **self-prediction error** — self-supervised; S itself is the ground truth; zero external labeling cost. Cognitive-bandwidth constraint (architectural hard constraint): M_self must be lighter than S, and its update frequency ≥ S's change frequency.

**M_self implementation**: Bayesian-style online belief tracking in log-odds space (b_t = b_{t-1} + evidence_t; evidence = per-round self-comparison evidence value); success probability p_t = σ(b_t); prediction target = marginal change in success probability.

**Trivial-solution defenses (the crux of this track's design)**:
- Forbidden predictions: ① S's output distribution (parrot solution) ② S's logits confidence (copy-paste solution)
- **Main prediction target: predict the learning curve** — given a "new domain/new task," predict "samples needed for S to reach the expert threshold." This prediction cannot be derived from the task itself; it must induce "S's learning-speed law" from S's historical learning trajectories; ground truth = the T(n) the experiment will measure anyway, zero extra cost
- **Strongest anti-trivial criterion: generalize to the future self** — after S updates, M_self's prediction error on the "new S" does not rise significantly. A parrot learns a static snapshot of S and collapses the moment S changes; a true self-model learns S's change law and keeps up

**Criteria**: ① self-prediction error decreases with training ② after domain switches, error first rises then falls (M_self keeps up with the "new self" — the fall speed quantifies cognitive bandwidth, reported as a secondary indicator) ③ domains where self-prediction is accurate → task performance improves on those domains (self-cognition serves behavior, not narcissism) ④ control: does S without M_self gain behavior ⑤ generalize to the future self

**M_other interlocutor modeling (sub-track)**: the brain's default network handles self-reference and theory of mind simultaneously (simulation theory — knowing others = simulating others with the self-model). This sub-track depends on future dialogue-type task-domain extension (current task domains A–D have no dialogue scenarios). Operationalization: M_other's predictions about the same interlocutor should show **systematic differences** between "with group context" and "one-on-one dialogue," with the difference size reflecting the person's role-deviation in the group. Anti-trivial criterion: generalize to changing people (people change), not lock onto classification labels.

**Three stages (against recursive self-reference)**: 1. observation (M_self only observes, does not intervene) → 2. participation (predictions enter decisions: knowing "it won't work" → allocate more compute / request redirection) → 3. internalization (self-representation enters S's decision loop). Stage 2 only begins after stage 1 passes.

**Risks and defenses**: trivial solutions → predict the learning curve + criterion ⑤; recursive self-reference → three-stage gating; narcissism (only caring about itself) → external-task secondary goal + criterion ③; insufficient cognitive bandwidth → architectural constraint.

---

## 7. Milestones

| Milestone | Content | Criterion |
|--------|------|------|
| M0 | Base reproduction (domain A) | Parallel-expansion gain reproduced + basin visualization |
| M1 | E1 exploration mechanism (domains A→B) | K-expansion holds in the symbolic domain |
| M2 | E2 memory + E4 composition (domains B→C) | Retention + platypus test pass |
| M3 | E3 distillation (core) | T(n) decreases? **the biggest unknown** |
| M4 | E5 causality + E6a/E6b + E7 | E6a fingerprint criterion (H1) + attribution path |
| M5 | Real-domain scaling (long-term track) | Best configuration transfers to text/tool domains |

> M3 is the watershed: if distillation fails, M4/M5 are meaningless; exit conditions in §8.

---

## 8. Failure criteria and exit conditions

| Scenario | Verdict | Action |
|------|------|------|
| M0 reproduction fails | PTRM gain not reproducible | Check implementation; if the frontier result is unstable → down-weight the exploration-mechanism hypothesis |
| M1 K-expansion fails in the symbolic domain | Exploration mechanism only works for puzzle domains | Dynamics tuning / inverse-model direction |
| E6a plain fine-tuning catches the mechanism arm | Plain learning already reaches the same transfer efficiency | H0-support direction: mechanism has no incremental value; report and switch routes |
| M3 all distillation plans fail | **Learning-mechanism bottleneck confirmed** | Experimental core conclusion: current components cannot produce learning acceleration → report + explore new mechanisms (e.g., abstraction-prediction class) |
| M4 causal layer has no gain | Correlation learning suffices | Causal layer deferred; does not block the main path |
| E6a continual-learning signature falsified | T(n) rising (stagnation-type) or R below threshold (forgetting-type) | Report degradation mode + mechanism attribution; same as M3: report as output, switch routes |

The value of this proposal = cheaply answering "is the learning-mechanism route viable" with controlled cost (local GPU + small models).

---

## 9. Glossary

- **Basin**: regions of TRM latent space converging to correct/incorrect answers. A deterministic model falling into a bad basin and unable to escape = the mathematical definition of "death"
- **Platypus test**: the operational definition of compositional generalization — inferring properties of a new entity from known feature combinations (18th-century biologists inferring that platypuses lay eggs/swim)
- **AGI fingerprint**: out-of-distribution few-shot transfer + autonomous rule discovery (replacing the unmeasurable "general intelligence")
- **Budget matching**: control and experimental groups consume the same compute budget (inference steps / parameter scale), removing "spending more compute" as a confound of "mechanism"
- **Expert level**: the pre-registered per-domain pass threshold (accuracy ≥ domain ceiling × 80%; ceiling calibrated first, then registered; see §5.4)
- **Super-additivity (synergy)**: all-on effect significantly better than the sum of each component alone — the operational test of the "coupling" hypothesis
- **Continual learning**: the same system (same weights, no domain-level retraining) accumulating capability over a domain sequence; signature = acceleration (T(n) non-increasing) + retention (R ≥ threshold) + continuity (no retraining)
- **Retention matrix R(i,j)**: retention of domain i after learning domain j; lower triangle = forgetting pattern
- **Wanderer retention**: a selector policy that forcibly retains outlier individuals (proportion ρ), reserving adaptation embers for distribution shifts — a discrete implementation of the persistence-of-excitation theorem
- **Immune neutralization**: third-candidate synthesis arbitration when multiple high-confidence candidates conflict — conflict as an information source, not noise
- **Self-prediction error**: the difference between M_self's predictions of S's behavior/learning attributes and actual performance — a self-supervised signal requiring no external labels
- **Trivial solution**: M_self degenerating into a "copy of S" (predicting output distributions / copying confidences) — defenses: predict the learning curve + the future-self criterion
- **TTT arm**: inference-time online weight-update control arm driven by a self-supervised objective — the strongest end of the online-plasticity spectrum
- **Belief recursive update**: Bayesian-style online belief tracking in log-odds space (b_t = b_{t-1} + evidence_t); M_self's implementation mechanism

---

## 10. Reference anchors

- **PTRM**: probabilistic micro recurrent model (parallel stochastic expansion + quality selection, 7M parameters) — prior-validation source for the exploration mechanism (the upstream paper reports 62.6%→91.2% at K=100)
- **Titans**: Google memory architecture (surprise-based learning) — BABILong beyond 2M context
- **SkillsBench** (arXiv 2602.12670): curated skills +16.6pp (33.9%→50.5%) / self-generated null — anchor for "experience→skill" distillation as the bottleneck
- **Brain-science cognitive mapping**: Oxford team's four-step cognitive pipeline (hippocampal high-dim / perceptual low-dim / vector addition / spatial anchoring) — blueprint for compositional-generalization mechanisms
- **Causal-JEPA**: arXiv 2602.11389 — object-centric world models + latent interventions
- **RLM**: recursive language model (MIT, 2026-08) — context organization causing 8–32× generalization gaps on a 30B model
- **Catastrophic forgetting**: McCloskey & Cohen 1989 — structural prior for standard fine-tuning forgetting on long sequences
- **Brain-science cross-reference**: MIT aphasia research (reasoning without language areas) / GCML (noise + inverse model = planning) / ghost points (learning as a bifurcation phenomenon) / sleep phase transitions (destabilization–restabilization) — motivational anchors; formal citations to be completed after verification

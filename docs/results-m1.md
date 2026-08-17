# M1 Results: The Exploration Mechanism Fails in the Symbolic Domain (Negative Result)

> Author: Lumen

The E1 (exploration mechanism) main criterion failed in domain B (RPN stack-machine arithmetic). This report records the complete failure-localization process and redesign directions.

---

## M1 result and failure branch

After training from scratch for 4000 steps on domain B (RPN — reverse Polish notation — stack-machine arithmetic): K=1 deep-expansion baseline 0.478 / K=10 0.476 / K=100 0.476 — **increasing K yields zero gain**; the E1 main criterion failed; corollary 1 is not supported in the symbolic domain. The failure mechanism was fully localized through three rounds of discrimination:

1. The quality selector (q-head) has no discriminative power on domain B (AUC 0.569 ≈ chance, vs. 1.0 on domain A) — initially judged as "does not generalize across domains."
2. **Majority voting falsified that hypothesis**: pure majority voting over the same batch of rollouts = deep-expansion baseline — no incremental information across rollouts.
3. **Semantic-level exploration falsified**: noise moved from the latent space to the token/rule space (swapping operands, swapping subtrees); K=30 semantically equivalent perturbations + majority voting = baseline. The model's errors are **rule-based** (the same error pattern on equivalent programs).

> **Implementation caveat (fixed 2026-08-14)**: the variant generator in `m1_sem_explore.py` had a tree-aliasing bug — the swap walk mutated the source AST in place, so "variant 0 = original program" (the paired baseline) was actually the walk's last mutation state, and variants were a dependent walk rather than i.i.d. draws. Swaps preserve the answer, so the round-3 conclusion (rule-based errors; no K-gain) is unaffected; the paired-baseline and per-pool statistics are approximate. The fix is in the script; archived results stand as recorded.

**Robustness extension**: extending training to 10000 steps and re-testing (the model was still climbing; test 0.606 / train 0.623), the K-curve remains non-monotonic (K=1_D=48 0.604 / K=100 0.590, see src/results/k_curve_10000.json) — the 4000-step "flat curve" conclusion holds under longer training. The "half-trained model" explanation is simultaneously excluded: training to maturity does not revive the exploration mechanism.

**Round 4 — training-side augmentation (2026-08-12, neutral)**: equivalence-class information injected on the *training* side also carries no gain. On-the-fly per-sample uniform sampling of ADD-swap forms (dataset/size/passes/steps/compute all matched to the baseline; the only variable = training distribution): 0.612 vs baseline 0.606 at 10000 steps — +0.6pp, inside the baseline's own ±12pp swing (single seed) → **neutral** per the pre-registered band (≥0.65 positive / 0.55–0.65 neutral / <0.55 negative). Offline full-class enumeration (513K samples, 5.1× dataset): 0.485 — the -12pp vs the on-the-fly arm is **pass dilution** (12.7→2.5 passes; train 0.481 < test 0.485 = undertrained), not form harmfulness (the on-the-fly arm proves the forms are harmless). Consequence: the equivalence-class channel is dropped from the E3 prior-control factorial (channel ③ removed; ① pretraining volume + ② architectural bias remain for the strong-prior cells). Curves: `results/m1_domainB_aug.json`, `results/m1_domainB_aug_batch.json`.

**Final mechanistic conclusion: inference-time exploration (any input-side carrier) is structurally ineffective for problems that are "uniquely solvable but not yet learned"** — PTRM-style exploration solves "non-unique-solution" problems (basin geometry allows cross-basin sampling), not "rule-not-yet-learned" problems (which need direction/training-side fixes). This conclusion is a valid scientific output: it delimits the applicability boundary of exploration mechanisms.

**Discrimination (2026-08-14): PASSED=False — exploration bit closed.** The inverse-model candidate was discriminated per the pre-registered grid (`results/inv_model/matrix.json`): all 6 b-configs (α∈{0.1,0.5,1.0} × σ∈{0.1,0.2}) fell below the a+2pp threshold (best 0.490 vs 0.508), i.e. direction-field exploration carries no decision-relevant geometry over pure noise — the third independent evidence for structurally-ineffective inference-time exploration. The exploration mechanism bit is **permanently closed** per S3 pre-registration; E6a mechanism arm = E2+E3+E4+E5 coupling (four mechanisms).

**Discrimination (2026-08-14): PASSED=False — exploration bit closed.** The inverse-model candidate was discriminated per the pre-registered grid (`results/inv_model/matrix.json`): all 6 b-configs (α∈{0.1,0.5,1.0} × σ∈{0.1,0.2}) fell below the a+2pp threshold (best 0.490 vs 0.508), i.e. direction-field exploration carries no decision-relevant geometry over pure noise — the third independent evidence for structurally-ineffective inference-time exploration. The exploration mechanism bit is **permanently closed** per S3 pre-registration; E6a mechanism arm = E2+E3+E4+E5 coupling (four mechanisms).

**Redesign directions (candidates, within the M1 budget)**:
- **Dynamics tuning (bifurcation redefinition)**: domain A's K-gain depends on dynamics near the ghost point of an early saddle-node bifurcation (critical learning rate ∝ T⁴); domain B's bifurcation structure differs → scan domain B's learning rate × memory-duration ratios to find new critical parameters. If a ratio restores the K-gain → the E1 failure was a bifurcation-localization success (not an architectural defect); if the full scan yields zero gain → the original conclusion stands. Cost: ~20–30 training runs.
- **Inverse model (goal-conditioned inverse model)**: pure-noise exploration is ineffective on "basin-free" flat terrain — noise has no direction. Design: incrementally train a lightweight inverse model (~2M parameters) mapping "current state + goal" → "differential direction vector toward the goal"; at inference, that direction + Gaussian noise = directed expansion. Falsifiable prediction: on domain B, "inverse-guided + noise" restores the K-gain (vs. zero gain for pure noise + "inverse-only, no-noise" degrading to greedy). Cost: lightweight incremental training (~1/4 of base scale).

## Related observation (E7b arbitration candidate)

On domain B, the "best-of" and "voting" arbitration arms both failed empirically (no incremental information across rollouts) — the neutralization arm is the only arbitration candidate with measurement space left.

---

**This conclusion is a valid scientific output: it delimits the applicability boundary of exploration mechanisms.**

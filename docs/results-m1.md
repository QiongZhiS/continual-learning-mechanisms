# M1 Results: The Exploration Mechanism Fails in the Symbolic Domain (Negative Result)

> Author: Lumen

The E1 (exploration mechanism) main criterion failed in domain B (RPN stack-machine arithmetic). This report records the complete failure-localization process and redesign directions.

---

## M1 result and failure branch

After training from scratch for 4000 steps on domain B (RPN — reverse Polish notation — stack-machine arithmetic): K=1 deep-expansion baseline 0.478 / K=10 0.476 / K=100 0.476 — **increasing K yields zero gain**; the E1 main criterion failed; corollary 1 is not supported in the symbolic domain. The failure mechanism was fully localized through three rounds of discrimination:

1. The quality selector (q-head) has no discriminative power on domain B (AUC 0.569 ≈ chance, vs. 1.0 on domain A) — initially judged as "does not generalize across domains."
2. **Majority voting falsified that hypothesis**: pure majority voting over the same batch of rollouts = deep-expansion baseline — no incremental information across rollouts.
3. **Semantic-level exploration falsified**: noise moved from the latent space to the token/rule space (swapping operands, swapping subtrees); K=30 semantically equivalent perturbations + majority voting = baseline. The model's errors are **rule-based** (the same error pattern on equivalent programs).

**Robustness extension**: extending training to 10000 steps and re-testing (the model was still climbing; test 0.606 / train 0.623), the K-curve remains non-monotonic (K=1_D=48 0.604 / K=100 0.590, see src/results/k_curve_10000.json) — the 4000-step "flat curve" conclusion holds under longer training. The "half-trained model" explanation is simultaneously excluded: training to maturity does not revive the exploration mechanism.

**Final mechanistic conclusion: inference-time exploration (any input-side carrier) is structurally ineffective for problems that are "uniquely solvable but not yet learned"** — PTRM-style exploration solves "non-unique-solution" problems (basin geometry allows cross-basin sampling), not "rule-not-yet-learned" problems (which need direction/training-side fixes). This conclusion is a valid scientific output: it delimits the applicability boundary of exploration mechanisms.

**Redesign directions (candidates, within the M1 budget)**:
- **Dynamics tuning (bifurcation redefinition)**: domain A's K-gain depends on dynamics near the ghost point of an early saddle-node bifurcation (critical learning rate ∝ T⁴); domain B's bifurcation structure differs → scan domain B's learning rate × memory-duration ratios to find new critical parameters. If a ratio restores the K-gain → the E1 failure was a bifurcation-localization success (not an architectural defect); if the full scan yields zero gain → the original conclusion stands. Cost: ~20–30 training runs.
- **Inverse model (goal-conditioned inverse model)**: pure-noise exploration is ineffective on "basin-free" flat terrain — noise has no direction. Design: incrementally train a lightweight inverse model (~2M parameters) mapping "current state + goal" → "differential direction vector toward the goal"; at inference, that direction + Gaussian noise = directed expansion. Falsifiable prediction: on domain B, "inverse-guided + noise" restores the K-gain (vs. zero gain for pure noise + "inverse-only, no-noise" degrading to greedy). Cost: lightweight incremental training (~1/4 of base scale).

## Related observation (E7b arbitration candidate)

On domain B, the "best-of" and "voting" arbitration arms both failed empirically (no incremental information across rollouts) — the neutralization arm is the only arbitration candidate with measurement space left.

---

**This conclusion is a valid scientific output: it delimits the applicability boundary of exploration mechanisms.**

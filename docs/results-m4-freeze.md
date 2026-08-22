# M4 pre-freeze batch — n_gate roster + domain ceilings + sequence ordering

> Pre-registration basis: `docs/experiment-proposal.md` §3.3 (v0.7/v0.22 frozen distance matrix + v0.23 ordering rules) and the §0.0 gate rule (ceiling < 0.6 → domain exits T(n)/R signature statistics).
> Status: **partially determined (A/B/C/D fixed) · final freeze pending domain-E selection** (E selection requires an A→B→C→D continuous-trained baseline → zero-shot transfer measurement on the three E candidates → mechanical selection of the lowest).
> Date: 2026-08-22. Predecessor: B1 probe = M3_NO_GO (distillation axis archived — orthogonal to this batch; domain-E selection and the prior axis are both independent of the distillation axis).

## Domain ceilings (n_gate inputs; calibration class, no significance gate)

| Domain | Ceiling | Source | Gate (≥ 0.6 → enters T(n)/R signature stats) |
|---|---|---|---|
| A puzzles (Sudoku) | **0.48** (frozen, M0.3) | M0.3 acceptance | ✗ excluded (< 0.6; reported only; its E1-foundation role unaffected) |
| B symbolic (RPN) | **0.707** (candidate; continuation to 20k steps, best @19.5k) | `docs/results-domainB-ceiling.md` | ✓ retained (v0.33) |
| C composition | **1.0** (final; 3 configs × 10k steps, all converge) | `docs/results-domainC-ceiling.md` | ✓ retained |
| D game (IPD) | **adaptation-premium ceiling 306.07** (vs TFT pool mean 244.69; criterion threshold ×1.1 = 269.15; premium headroom **+13.7%**) | analytic best-response rollout, `m0_4_gen_domainD.py` ceiling() (frozen 2026-08-12, values in `results/m4_freeze/domainD_ceiling.json`) | ✓ retained (criterion signal measurable: analytic ceiling exceeds threshold by 13.7% ≫ noise floor; game-domain ceiling is analytic — no mechanism-free baseline needed, domain-D design) |
| E | **to select** (three candidates measured, mechanical lowest) | needs A→B→C→D baseline + zero-shot measurement | pending (E retained → n_gate = 4 domains) |

**n_gate = {B, C, D} (+E pending) → 3–4 domains ≥ the §3.3 lower bound of 3** ✓ (the E6a signature conclusion does not degrade to "exploratory case study" on domain count).

## Domain-D ceiling detail (analytic, IPD 8-opponent pool)

- Criterion: adaptation premium = strategy expected payoff > tit-for-tat pool mean × (1+ε), ε = 0.1 fixed (sensitivity 0.05/0.2 reported); base = TFT pool mean.
- TFT pool mean = **244.69** (matches the M0.4 acceptance baseline) · threshold ×1.1 = **269.15** · optimal-response pool mean = **306.07** (per-opponent ceilings: always-cooperate 500.0 / always-defect 100.0 / tit-for-tat 302.0 / generous-tft 302.0 / random 301.34 / punisher 302.0 / sleeper 220.0 / prob-deceiver 421.2).
- Premium headroom = +13.7% above the threshold → the "adapt beyond TFT×1.1" signal is measurable for a 7M learner; D passes the gate on its own criterion scale (the 0.6 accuracy gate does not map to a score-premium domain; measurability of the criterion signal is the operative test, per the domain-D design).

## Domain-sequence ordering (frozen matrix · min-total-distance near→far · start = nearest to B anchor = B)

Distance matrix (v0.22 frozen, `results/domain_distance.json`): B↔E1 **0.52** · B↔E2 2.36 · B↔C 2.86 · B↔E3 3.32 · B↔D 3.92 · **B↔A 4.78 (largest)** · C↔E2 1.59 · E3↔A 2.68.

| If domain E = | Min-total-distance ordering | Total | Start |
|---|---|---|---|
| E1 (near; B↔E1 0.52) | **B → E1 → C → D → A** | 12.71 | B |
| E2 (mid; B↔E2 2.36) | **B → E2 → C → D → A** | 13.28 | B |
| E3 (mid; B↔E3 3.32) | **B → C → E3 → A → D** | 14.02 | B |

> Note: A is excluded from signature statistics by the gate but remains in the sequence as the base/far-end member (the E6a sequence is the 5-domain re-ordering, not the original A→B→C→D→E order — the measured A↔B distance is the largest, so A sorts last). Once E is selected, the corresponding ordering becomes the final sequence, frozen together with the n_gate roster and ceiling values (§3.3 freeze point).

## Pending (before the freeze is final)

1. Domain-D trainer (`m0_4_train_domainD.py` — not yet in the public repo; data format is the same sequence-prediction protocol as B/C: 81-position obs→action, pos0 = action C/D; port the C trainer; eval = adaptation premium over the 8-opponent pool).
2. A→B→C→D continuous training (mechanism-free baseline TRM): A checkpoint → B fine-tune → C fine-tune → D fine-tune (~8h GPU) — prerequisite for E selection and for the E6a baseline.
3. Domain-E three-candidate zero-shot transfer measurement on the A→D baseline → **mechanically select the lowest** (the three observed points also support the "transfer decays with structural distance" gradient hypothesis).
4. Freeze registration: n_gate roster + sequence start + ceiling values + final ordering, registered together (§3.3 freeze point).

## Relation to the B1 probe

The B1 M3_NO_GO verdict does not block this batch: domain-E selection and the prior axis are both orthogonal to the distillation axis; the E6a mechanism arm's E3 slot is labeled "negative conclusion archived" (v0.35 backfill in the working-workspace proposal).

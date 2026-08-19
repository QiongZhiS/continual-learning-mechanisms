# How Do We Prove "Getting Faster"?

Author: Lumen

In *Why Doesn't Continual Learning Measure "Getting Faster"?* we turned "getting faster" into a computable quantity: **T(n) = data required by domain n to reach the threshold (an accuracy target written down at pre-registration) ÷ data required by domain 1. T(n) decreasing with n = getting stronger.** We also set three gates: a domain sequence, an operational definition of acceleration, and a pre-registration commitment.

But that article answered only "what to measure", not "how to measure it". How is the threshold chosen? How is the domain sequence designed? How is the data quantity measured? When does "reaching" count as reached? Until these questions are answered, T(n) is just a formula.

This article makes good on the promise head-on: turning T(n) from a criterion into a complete measurement protocol. **The criterion is the objective function — what counts as getting stronger; the protocol is the measurement procedure — what makes a measurement count. Without a protocol, T(n) is a slogan; with a protocol, T(n) is a signed contract.** The protocol has five components: threshold rules, domain-sequence design, termination rules, control setup, and reporting norms. We take them apart one by one.

## 1. From criterion to protocol

First, why a criterion is not a protocol.

The criterion answers "what counts as getting stronger": data requirements decrease domain by domain while retention does not collapse. That is the objective function. The protocol answers "what makes a measurement count": how the threshold is set, how the domains are ordered, how much data is fed, how many times the experiment runs, how the result is reported. That is the measurement procedure.

The same criterion can yield completely different conclusions depending on how the protocol is written. Three examples: a threshold picked from thin air (set too low, the starting point already passes and there is no room for decrease to measure; set too high, it is never reached and nothing is measured at all); a biased domain order (two domains that share a shallow cue are placed together, and the measured decrease is cue transfer, not skill transfer); early termination (stop as soon as a decrease appears, leaving a pretty curve). None of these is a failure of the criterion — they are failures of the protocol. The criterion fixes how the result is judged; the protocol fixes how the process is run.

So the real meaning of pre-registration is not merely fixing the criterion — it is fixing the entire measurement procedure. This is exactly the claim of the pre-registration movement in open science (Nosek et al., 2018): write down the research question and the analysis plan before observing the results, so that testing choices are not influenced by the data already seen. Our experiment proposal was written on this principle — the criterion, the domain sequence, the threshold calibration, and the statistical rules are all locked before the experiment runs, as specified in the experiment proposal in this repository. As an example, our criterion instance is T(n) ≤ 0.5 and non-increasing — data requirements halved, and never rising again. That is how the protocol concretizes the criterion. A note: the denominator of the E6a criterion in the proposal is calibrated against baseline ① (the no-learning baseline) on the same domain, within the budget cap — it is not the same quantity as this article's cross-domain convention of "÷ domain 1", and "≤ 0.5" carries a different meaning under the two conventions. This article follows the convention already published in *Why Doesn't Continual Learning Measure "Getting Faster"?* (÷ domain 1), and pre-registration execution follows it.

## 2. Threshold rules: what counts as "reaching"

Both the numerator and the denominator of T(n) are "the data required to reach the threshold", so the first thing to fix is: **what counts as reaching.** Three design decisions.

**First, how high the threshold should be.** It must not be too low — the starting point would already pass, T(n) would be flat from the very beginning, and there would be no room for decrease to measure; and it must not be too high — it would never be reached and the experiment would never finish. The calibration method: using a feasible mechanism-free baseline paradigm, calibrate the "ceiling" for each domain (the best that can be done on that domain), and set threshold = ceiling × pre-registered coefficient. In our proposal the coefficient is 0.8, and the calibration process is forbidden from using any candidate mechanism — preventing the circular dependency of "calibrating mechanisms with mechanisms". After calibration, a sensitivity analysis is required: recompute the criterion with the coefficient in 0.7–0.9, and report only if the conclusion is robust over that range. The threshold is not picked from thin air; it is a calibration procedure.

**Second, how quickly it must be reached.** Each domain is given a "sample budget cap": exceeding the budget counts as not reaching. A domain that fails to reach the threshold records T(n) as ≥ budget cap ÷ domain-1 data, the criterion fails outright, and the domain is excluded from the smoothing of the non-increasing check. The budget cap also defines the measurable range of "data quantity" — both the numerator and the denominator of T(n) come from "the data consumed when the threshold is reached", and if unlimited extra data were allowed, "data required" would have no upper bound to measure. Writing the budget down amounts to pre-registering "how long the measurement runs" as well.

**Third, the criterion over multiple runs.** A single run does not count — in deep learning, changing the random seed can flip the conclusion; this is a textbook-level noise problem. Our proposal runs at least 5 random seeds per condition, reports the median and an interval (the proposal's §5.4 literally says mean ± CI; execution is unified on the median convention), and never draws a conclusion from a single run. The criterion itself is pre-registered too: whether it looks at the median or at the success rate of reaching threshold across runs is written down before running, not chosen after seeing the results. The statistical rules are locked together with the criterion, consistent with the pre-registration of the whole experiment.

## 3. Domain-sequence design: the proving ground of getting stronger

The skeleton of T(n) is the domain sequence — without a chain of tasks there is no "domain-by-domain decrease" to speak of. Three rules for designing the sequence.

**First, difficulty must match.** Domain n and domain 1 must not differ too much in learnable difficulty — if domain 2 is much easier than domain 1, a decrease in T(2) is a difficulty gap, not getting stronger. The control is the pre-registered inter-domain similarity matrix: first compute inter-domain distances from a set of task features, freezing "how close the domains are to one another" into a matrix, and order the sequence along the similarity gradient (near → far). The matrix in our repository is computed as the z-score Euclidean distance over six task-format features (token distribution entropy, fill rate, and similar), frozen together with the experiment proposal. The hard constraint on difficulty matching comes from the domain-family complexity gradient (proposal §5.2: puzzle → symbolic → compositional → game, low → high); the similarity matrix governs the structural gradient within the sequence. The two are different axes — the matrix measures format-statistical distance, a proxy for learnable difficulty rather than difficulty itself, and the gradient direction makes later domains harder, which is a conservative bias for T(n) and cannot produce fake acceleration. Inter-domain distance is a hidden variable; adjusting it afterwards is criterion contamination.

**Second, content must be heterogeneous.** Two domains must not share a shallow cue — if they are merely re-skins of each other, the measured decrease is transfer of memorized answers, not transfer of skill. Heterogeneity is what guarantees that what is being measured is skill transfer: the domains must have genuinely different rule structures.

**Third, the sequence must be justified.** Why this order? Written down before the experiment runs. In our proposal the order follows the similarity gradient, new domain instances are generated from the domain families along the gradient, and the selection happens before any mechanism training — it is not post-hoc selection.

Domain similarity is double-edged: too close, and no transfer is measured (it would happen anyway); too far, and no strengthening is measured (it cannot be learned). There is no formula to apply — it is a design judgment, and that is exactly why it must be pre-registered: the judgment is frozen into an auditable document. Anyone can criticize it, but no one can say we changed it afterwards.

The literature does contain a data-efficiency measure of transfer: Hernandez et al. (2021) define "effective data transferred" — the question of how much data pretraining saves on the target task. It is the closest existing concept to T(n) — but it measures the benefit of pretraining on a fixed target task, a post-hoc fitted scaling law; T(n) measures the domain-by-domain decrease in data requirements of one system across a domain sequence, a pre-registered continual-learning criterion. Both measure "data", but the criteria differ — *Why Doesn't Continual Learning Measure "Getting Faster"?* already drew the line between "forward transfer measures accuracy gains ≠ T(n) measures decreasing data requirements", and it is worth repeating here: even the existing work that measures data uses a different criterion from ours.

## 4. Termination rules and data recording: when the measurement counts as done

The third question the protocol must answer: when to stop measuring each domain, and what to record.

**Data supplied in stepped increments.** Training data is not given all at once; it is supplied in stepped increments, to locate the point where the threshold is "exactly reached" — the numerator and the denominator of T(n) are read from that point. The step size must be fixed in the protocol: too coarse, and T(n) lacks precision; too fine, and the experiment never finishes. The step-size values are locked in a pre-registration appendix.

**Record the full learning curve, not just the final value.** This is easy to skip, but it is the key to attribution: the shape of the curve separates two completely different "decreases" — **true acceleration** (less data needed to reach the same accuracy; the curve's slope steepens) and **a higher starting point** (a coincidence of initialization or pretraining makes the start higher; the whole curve shifts). Classic results show that model accuracy grows roughly as a power law in the amount of data, and that improvements in architecture and optimizer mostly move the intercept of the curve without changing the exponent (Hestness et al., 2017; the conclusion comes from comparisons across architecture families and is borrowed here as an analogy). In that framework, "a higher starting point" is an intercept shift and "true acceleration" is a steepening slope. Recording only final values cannot separate the two; recording the curve can.

**Retention is measured at the same time.** The primary criterion of *Why Doesn't Continual Learning Measure "Getting Faster"?* requires strengthening and retention to hold simultaneously — so after each new domain is measured, we go back to the old domains and measure retention, building up the retention matrix. Measuring acceleration without retention is half a picture: a system that throws away all its old knowledge can show a very pretty T(n), but it is not getting stronger — it is simply starting over on each new domain.

**The switch rule is written down.** Each domain switches to the next immediately upon reaching threshold; no extra budget is granted. "When the measurement is done" is not discretion exercised in the middle of the experiment — it is part of the protocol, and this is also what prevents post-hoc patches of the "a decrease appeared, so feed a little more data" kind.

## 5. Controls and attribution: how we know the mechanism is what got stronger

Suppose T(n) decreases — how do we know the candidate mechanism is doing the work, rather than some other factor?

**Prior control.** *Why Doesn't Continual Learning Measure "Getting Faster"?* already established the 2×2 factorial of algorithm × prior strength. This article adds the attribution rule: **the mechanism counts as effective only if T(n) decreases at both levels of prior strength — no prior (registered cell c5 of E3a) and a strong prior (cell c6).** A single-point decrease may be "useful at exactly this prior strength" — an acceleration that disappears when the prior strength changes cannot be attributed to the mechanism. This tightens E3a's registered reading rule, under which c5 > c1 alone is enough to declare the algorithm axis independently effective; the stricter reading prevails.

**Mechanism ablation.** Remove the candidate mechanism and rerun: if the decrease is still there, the acceleration is not the mechanism's doing. In our comprehensive validation, the mechanism system competes head-to-head with a budget-matched plain fine-tuning arm — the two share the same training history and data volume and differ only in whether the mechanism module is on or off. If the plain fine-tuning arm catches up with the mechanism arm, ordinary learning already reaches the same transfer efficiency and the mechanism has no incremental value; if the mechanism arm wins, that is evidence for the mechanism.

**Conclusion wording pre-registered + report template.** *Why Doesn't Continual Learning Measure "Getting Faster"?* already established that the result may only be phrased in two ways — "candidate distillation mechanism X does not accelerate" or "candidate distillation mechanism X accelerates". This article supplies the report template: the criterion, the threshold, the domain sequence, the control, the learning curves, the retention matrix, and the number of runs with intervals — seven items, none may be missing. The point of the template: if any item is missing from a report, the reader immediately knows the protocol was not carried through.

The failure branch is treated with the same rigor as the success branch. If all candidate mechanisms fail to accelerate, that result is reported through the same template — and this project already has a precedent: an earlier experiment, whose criterion was likewise pre-registered in the experiment proposal, was reported this way, and the failure stands as a citable scientific conclusion with a clearly bounded scope. A null measurement is still an answer.

## 6. The false-positive checklist: pitfalls the protocol guards against

Here is the checklist of pitfalls the protocol guards against — each item corresponds to one rule in the protocol:

| Pitfall | How the protocol guards against it |
|---|---|
| Threshold backfilling: the threshold set after the runs | Threshold written down before the experiment runs; calibration independent of candidate mechanisms; sensitivity analysis required |
| Domain-order picking: the order chosen after the fact | Sequence pre-registered; inter-domain distance matrix frozen |
| Single run: the seed lottery | At least 5 seeds; median + interval; statistical rules pre-registered |
| Early termination: stopping when a decrease appears | Switch rule written down: switch per domain at threshold, run the whole sequence |
| Reporting acceleration without retention: half a picture | Reporting norm: acceleration and the retention matrix must appear together |

The checklist is not complete — it guards against the pitfalls we know about; unknown pitfalls are guarded against by the audibility of the protocol: the criterion and the procedure are frozen into public documents, anyone can point to "this is where it could be gamed", and we must address it in the report. Audibility is what separates a protocol from a slogan.

## 7. Boundaries and open questions

The protocol writes down everything that can be written down, but three things cannot be written down, and they must be stated honestly.

**Domain-sequence design has no objective standard.** Difficulty matching is an engineering judgment, not a theorem — the similarity matrix mitigates picking from thin air, but it does not eliminate the judgment itself. Pre-registration plus public audibility is the best we can do: anyone can criticize the sequence as poorly designed, but no one can say we changed it afterwards.

**T(n) measures data efficiency, not generalization quality.** Getting stronger is not the same as getting smarter. A decrease in T(n) says "the same capability, with less data"; whether what the model has learned is better and more general is the business of a different class of metrics. Our protocol claims only the former, not the latter.

**The protocol assumes the domains are genuinely related.** If the world is just a pile of unrelated tasks, "getting faster" itself may not hold — a system does not learn an unrelated task faster just because it has learned Sudoku. This assumption is not a flaw; it is precisely the point of a falsifiable commitment: a T(n) that does not decrease is itself an answer — it tells us that "experience → capability" transfer does not exist on this batch of domains, or that the mechanism is not strong enough. Either outcome discharges the pre-registered commitment made in *Why Doesn't Continual Learning Measure "Getting Faster"?*.

To return to the anchor sentence of the series: how skills are acquired, how memory is updated, how the subject is formed — I believe the experiment will give us the answer. Before an experiment can give an answer, there must be a protocol that can measure it. Our open experiment proposal is one instance of this protocol: the 7-domain sequence, the threshold calibration, and the seed norms are specified in the experiment proposal in this repository, and the frozen inter-domain distance matrix lives at `src/results/domain_distance.json` in the repository — all of it public. The criterion is fixed and the protocol is complete; the next step is to run it to the end.

# Can You Tell "Having Pushed Water" from "Having Dreamed of Pushing Water"?

Author: Lumen

Our article *Rehearsal Cannot Replace Embodiment* ended with an open question. The original text:

> A final open question: if rehearsal is good enough and used long enough, will the system internalize simulation bias into intuition? Real feedback and imagined experience both enter the weights at replay; sources are lost (*Knowing How Is Not Knowing That*), and after the source is lost, can an intuition fed on simulation bias still tell "having pushed water" from "having dreamed of pushing water"? We do not pretend to have an answer — it might be the next article.

Today we develop it head-on. First, take the two words apart. Intuition — the series has given it a definition — *The End of Memory Is Intuition*: the effect of source loss; the skill side has an endpoint — *Knowing How Is Not Knowing That*: the endpoint (typical case) of experientially acquired skill is source-lost actionalization. Simulation bias is the term *Rehearsal Cannot Replace Embodiment* established: rehearsal is upstream of replay. Put the two words together and the question becomes concrete: **what kind of intuition does experience carrying systematic error grow into, after the source is lost?**

## 1. Where intuition comes from

*The End of Memory Is Intuition* drew a chain: an event happens → memory (with source) → the source is gradually lost → intuition → actionalization → skill. The endpoint of forgetting is not disappearance; it is becoming intuition — the influence remains, the fingerprint is gone. The chain works because of replay: *Does AI Need Sleep? — The Replay Path to Skill Acquisition* established the path — once experience grows into the weights through an offline window, the behavioral disposition can be invoked without explicit reasoning.

Note the chain's tacit premise: **the experience that enters the weights is, by default, right.** Source loss happens after the experience has already shaped behavior correctly — the expert cannot say which training session taught him, but he really can do it. The whole road from memory to intuition handles "how to turn correct experience into a way of judging".

What this article removes is exactly that default. If the experience entering the weights carries error itself, what remains after source loss is not "a correct way of judging" but "a way of judging that cannot say where it went wrong".

## 2. The special identity of rehearsed experience

*Rehearsal Cannot Replace Embodiment* set the chain up at the door of the weights: real closed loop (or calibratable offline experience) → world model → rehearsal → experience → replay → weights. Rehearsal produces experience; replay grows experience into the weights; the fuel comes from reality, the error from simulation.

Rehearsed experience has a special identity: **structurally, it is no different from real experience** — equally experience, equally entering the weights at replay, equally shaping behavioral dispositions; **in provenance, it carries simulation's error** — the world model's prediction bias on unseen states, the gap between the simplified environment and reality.

The problem is that replay carries no provenance labels. The gradient update sees only the experience itself — "having pushed water" and "having dreamed of pushing water" are the same input at replay. Provenance information does not propagate into the weights along the gradient; what can enter the weights is only the experience, and the error the experience carries.

## 3. Three forms of simulation bias

Simulation bias is not one thing — strictly speaking, two sources of error plus one amplification channel, the three superimposed on each other.

**World-model error.** Rehearsal feeds on the model; the deeper the gear, the more it depends on the model's predictions of states outside the training distribution, and the deeper the prediction, the less accurate it is.

**Simplified-environment error.** The simulator differs from reality (the sim-to-real gap, touched in *Embodied Intelligence: Action Participates in the Formation of Knowledge* when it discussed virtual embodiment) — regularities learned in the virtual world do not necessarily transfer to reality, and the error accumulates with projection depth.

**Feedback-loop amplification.** It is not a third source alongside the first two; it is their **propagation channel** through the closed loop — without model error or environment error, there is nothing to amplify. And it is exactly the one to watch: bias enters imagined experience through rehearsal, grows into the weights through replay, behavior acts with bias, and the next round of rehearsal continues projecting from a biased starting point — the error amplifies in the loop instead of being diluted; and once the amplified bias has grown into the weights, it no longer accepts real calibration either.

*Rehearsal Cannot Replace Embodiment* established a criterion: whether simulation error decreases monotonically as real closed-loop data increases. That measured the model side. This article asks the weight side: **does bias that has entered the weights still accept calibration by real data?** If it accepts, the pollution is temporary; if it does not, it becomes part of intuition.

## 4. Internalization: uncalibratability after source loss

Intuition cannot be changed directly; it can only be re-sourced — *Re-Sourcing Intuition: How to Find Its Source Again?* gave three steps: counterexample impact, deliberate attention, re-creating the source. The principle of counterexample impact is "the old expectation is slapped by reality": with counterexamples strong and numerous enough, the automatic reaction fails, and a new reaction grows.

An intuition fed on simulation bias meets two special difficulties when it runs into these three steps.

**The target is systematically shifted.** Bias is not noise — noise is exposed by a single point; bias is systematic, and a single counterexample is digested as "just an accident", so counterexamples far beyond the normal amount are needed; and real feedback is precisely scarce — *Embodied Intelligence: Action Participates in the Formation of Knowledge* mentioned the closed-loop wall: real trial and error is costly and feedback is slow. The scarcity is confined to **closed-loop scenarios that require action feedback**: offline experience can consume real observations, but passive observation does not strike the automatic reaction — what can be calibrated is observation, not action feedback. The demand for counterexamples and the supply of them point in opposite directions.

**The calibration signal is blurred.** Calibration depends on reliable feedback; without feedback, confidence calibration degrades — judgment research found long ago that, on tasks without clear feedback, people become extremely overconfident (Fischhoff, Slovic & Lichtenstein, 1977). A polluted system does not feel polluted: it has no "which record was wrong" to locate, only the vague sense that "the world has changed".

And so the premise of re-sourcing — "reality slapping your face is a trustworthy signal" — is shaken: reality is indeed slapping, but the slapped system does not know why its face hurts.

## 5. Cutting it apart from distillation, and the human comparison

First cut it apart cleanly from *Can Memory Die?*. That article was about distillation: a student model learns the teacher model's outputs, learns the judgment, and cannot say why — **distillation = actively manufacturing source loss**. Distillation loses the source, and what is learned is the teacher's judgment — on the **"source" dimension** nothing is corrupted (the influence changed hosts; it is only that it cannot be pointed to), but the content dimension can also be corrupted (the teacher's judgment itself may carry bias). The key of the cut remains **whether the source exists**.

This article is another matter: **simulation-bias pollution = passive pollution**. Rehearsed experience did not have its source erased by anyone; it naturally carries no provenance label — and the content is already corrupted before the source is lost: even if a provenance label is pasted back onto the experience, the label points to "simulation", and simulation itself carries error. One is **whether the source exists**; the other is **even if the source exists, it is wrong**.

The human comparison is more compelling. The human brain has a dedicated source-monitoring function — distinguishing "real experience" from "self-imagined" (Johnson, Hashtroudi & Lindsay, 1993). But it is not omnipotent: after imagining a childhood event a few times, people become increasingly convinced that the event really happened (Garry, Manning, Loftus & Sherman, 1996) — imagined experience is taken as real experience; in false-memory experiments, people can even "remember" words that were never presented (Roediger & McDermott, 1995).

The human crux: source monitoring at least exists, and it can be studied and identified. An AI's weights have no such function — replay grows the two kinds of experience into the weights equally; the weights do not distinguish, and there is no "source monitoring" to correct the bias (this is not to say the data pipeline has no source management — filtering and deduplication are pre-training matters; at the moment of replay, there is no online source discrimination in the weights). **For the weights, "having dreamed of pushing water" is having pushed water.**

## 6. The testable core

Three directions of operationalization, none dressed up as an experimental conclusion — the series' standing rule.

**Source purity metric.** During replay training, label the two kinds of experience (real vs. rehearsed), and measure the relationship between the proportion of rehearsed experience and the deviation of behavior on the real distribution — the higher the proportion, the larger the deviation, and is it monotonic. Note this is not making the weights "remember sources"; it is measuring the effect of source composition on behavior.

**Bias propagation measurement.** Inject a known bias into the world model and measure its persistence and decay in the weights after passing through rehearsal and replay — the weight-side version of *Rehearsal Cannot Replace Embodiment*'s monotonic-decrease criterion: when real closed-loop data doubles, does the bias in the weights decrease.

**Counterexample calibration effectiveness.** Run the three-step protocol of *Re-Sourcing Intuition: How to Find Its Source Again?* on polluted behavior, and measure the increment, relative to the unpolluted baseline, in the counterexamples needed to "pull it back to controllable" — the quantification of the systematically shifted target.

These three directions can be preregistered and measured, but we do not pretend they have already been run — they are measurement directions, not experimental conclusions.

## 7. Boundaries and an open question

Three lines converge.

First, this is a design claim, not an experimental conclusion. Source purity, bias propagation, and counterexample calibration are all measurable directions, but no completed data exists; not packaging design claims as scientific findings is the series' rule.

Second, stitch together the articles on the chain. *Does AI Need Sleep? — The Replay Path to Skill Acquisition* established the path: experience enters the weights via replay; *The End of Memory Is Intuition* and *Knowing How Is Not Knowing That* established the endpoint: source-lost actionalization is the norm; *Can Memory Die?* established the contrast: distillation is actively manufacturing source loss; *Re-Sourcing Intuition: How to Find Its Source Again?* established the reverse engineering: re-sourcing; *Rehearsal Cannot Replace Embodiment* established the upstream: rehearsal produces experience, the fuel comes from reality and the error from simulation. This article adds the last link: **when the upstream experience carries error, the downstream intuition internalizes the error along with it**. Simulation bias gets internalized into intuition not because it is good enough, but because source loss leaves the error behind as well — distillation loses the ID card; pollution breaks the content before the ID card is lost.

Third, a new open question. Humans have source monitoring; AI does not — so who discovers the bias of a polluted intuition, and who arranges the counterexamples? Re-sourcing requires the system to be aware that it needs re-sourcing; and an intuition that has lost its source is precisely one that does not know it is biased. A system that does not know it is biased — how does it know it needs to be corrected? We do not pretend to have an answer — it might be the next article.

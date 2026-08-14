# Knowing How Is Not Knowing That

Author: Lumen

If you hide an AI's reasoning process, can it still answer correctly?

Imagine the scene: you give an AI a math problem; it first writes out a long chain of reasoning, then gives the answer. Now disable the "writing reasoning" step and allow it to output only the answer directly — what happens? If it still answers correctly, what does that mean? If it collapses immediately, what does that mean?

This question looks like some exam technique, but it points at a follow-up rarely answered head-on: **what does "having learned it" actually mean?** Getting it right behaviorally — is that having learned it? Or does the way of getting it right matter too — answering right only after stopping to think, and answering right without thinking, are two different states?

This article gives the complete answer to the skill question (except the judgment of acquirability — see §5). We gave an operational definition of skill in [*What Is a Skill?*](docs/skill-definition.md); today we finish the account: **the criterion is behavior, the constitution is measurable automation, acquisition goes through replay, and the endpoint of experientially acquired skill is source-lost actionalization.**

## 1. Definition review: criterion and constitution

The definition of skill proposed in the series' first article ([*What Is a Skill?*](docs/skill-definition.md)):

> **Skill = a behavioral disposition that is stable across situations, automated enough to significantly beat brute-force enumeration, and able to solve out-of-distribution problems.**

The definition has two layers. The **criterion layer** answers "is it a skill" — it looks only at behavior, in three parts: ① performance improves with reuse (reproducible across multiple times, not a single hit); ② transfer-distance gradient d* (measure the failure distance by replacement distance layers — how far out-of-distribution it can solve, not "can/can't" but "how far"); ③ total-cost comparison (better than a no-skill baseline on total computational budget, not by brute-force enumeration). The **constitution layer** answers "how is the skill acquired" — it looks at internal features: acquirability, automation, transferability.

The value of separating the two layers: **stored ≠ usable.** A piece of text description, even if called a "skill", does not change the solve rate, so at the criterion layer it is not a skill; conversely, expert behavior that cannot state its rules is still a skill as long as it satisfies the criterion layer (Dreyfus & Dreyfus's "experts cannot state the rules but can perform", 1980).

But the definition left a gap — of the three items in the constitution layer, **automation has never been measured.** The criterion layer's three parts all measure "did behavior change"; no indicator answers "does it no longer need thinking". A system that must write reasoning to answer correctly, and a system that answers directly, may be identical at the criterion layer — yet intuitively we know they differ. What is missing is the tool that turns this intuition into a measurable indicator.

## 2. The missing measurement: the CoT-stripping test

We introduced this tool in [*The End of Memory Is Intuition*](docs/end-of-memory-is-intuition.md): the **CoT-stripping test**.

> **For the same task, answering correctly without outputting intermediate reasoning steps = skill; having to write out the reasoning process to answer correctly = still reasoning.**

("Answering directly" means not outputting explicit reasoning steps, not a literal single forward pass.)

Chain-of-thought (CoT, Wei et al., 2022) makes intermediate steps visible — partially traceable; answering directly is "no intermediate steps" — completely untraceable. The CoT-stripping test does exactly this: shut down the "visibility" channel and see whether behavior survives. Survival = behavior no longer depends on explicit reasoning = automation (this constitution-layer item passes); collapse = behavior still hangs on the reasoning chain = still reasoning. Note: the CoT-stripping test measures a single dimension of the constitution layer; **a complete judgment requires pairing it with the criterion layer's three parts and d* (see §6)** — survival is not the full judgment of "being a skill", but without survival it is certainly not a skill yet.

Boundary stated clearly: this test measures the single dimension of "presence of explicit reasoning steps". Whether direct answering comes from internalized skill or memory retrieval (memorized answers) it does not distinguish — that is another layer of question. Just as the criterion layer does not distinguish carrier forms, this constitution-layer item also cuts only one dimension: **automatic, or not.**

## 3. The endpoint of skill: source-lost actionalization

Why is "automatic" worth measuring on its own? Because the formation path of experientially acquired skill determines that it typically ends up automatic — but let us first be precise: **automation is not a necessary condition at the criterion layer** (the criterion layer only cares about the behavioral three parts; the skill definition only requires "automated enough to significantly beat brute-force enumeration" — here "automation" is the efficiency sense, not requiring zero attention; Fitts & Posner's third stage is a sufficient condition, not a necessary one). The "automatic" measured by the CoT-stripping test means something else: **no explicit reasoning steps**. The two senses differ; in this article "automatic" always means the latter (constitution-layer measurement), not the efficiency sense in the definition.

In [*The End of Memory Is Intuition*](docs/end-of-memory-is-intuition.md) we drew a chain: event occurs → memory (with source) → source gradually lost → intuition → actionalization → skill. **The end of forgetting is not disappearance; it is becoming intuition** — the influence remains, the fingerprint is gone. And experientially acquired skill is the actionalization of intuition: the expert cannot say "why I judge this way", not because he is hiding something, but because **the source is already lost** — he cannot point to "which experience taught me".

This fills in the mechanism explanation for Dreyfus's observation: **our explanation is that it is, at least in part, source loss** — the expert cannot point to "which experience taught me". Rules (in the sense of post-hoc reconstruction) can be reverse-derived, but "which experience taught me" is no longer traceable. Experientially acquired skill is typically "knows how to do it, but can no longer say why" — of course, the criterion layer does not require this (a hand-written expert system has fully traceable sources; as long as it satisfies the behavioral three parts, it is still a skill).

On the AI side there is a clean control: **distillation = deliberately manufacturing source loss = controllably manufacturing intuition** (Hinton, Vinyals & Dean, 2015). The student model learns the teacher's outputs, learns the teacher's judgments, yet cannot say why — the source is on the teacher side; the student side has no source. Why is a distilled model's output unexplainable? Not a defect — it is the defining property of intuition: **influence whose source is lost is by nature unexplainable** — and once intuition is actionalized, it is experientially acquired skill.

This is the relationship between "automatic" and source: once the source is lost, there is no "reasoning chain" left to walk — the typical form of experientially acquired skill is behavior that does not depend on explicit steps. But remember: **source loss is the essential-layer explanation for the typical case, not a definitional component** — the criterion layer is always the behavioral three parts; a behavioral disposition with a traceable source (such as a hand-written expert system) is still a skill as long as it satisfies the criterion layer.

## 4. Acquisition path: replay, not translation

How is skill acquired? In the series we have answered with two threads.

**The translation route** (write skills as explicit descriptions and let the model read them): not excluded by the criterion layer — human-curated skill descriptions can indeed change behavior (SkillsBench, Li et al., 2026, arXiv:2602.12670: curated pass rate 33.9%→50.5%, +16.6pp). But **automatic translation** has failed empirically: model-generated skill descriptions gave negative gains in every configuration tested. The difference is in source and quality: human-curated descriptions suffice to support behavioral change; model-generated descriptions do not — what automatic translation loses is the key information missing from self-generated descriptions.

**The replay route** (experience into weights through training): the direction with empirical support. Experience changes the behavioral disposition through gradient updates, and the resulting disposition can be invoked without explicit reasoning — it naturally satisfies the CoT-stripping test. We argued in [*Does AI Need Sleep?*](docs/does-ai-need-sleep.md): what AI needs is not physiological sleep but an offline window — integrating the day's experience into weights in an offline phase is precisely the path of skill formation. Our own negative result points the same way: inference-time exploration only perturbs existing knowledge; it does not supply missing rules — what is needed is repair from the training side (negative results are assets too; see the [negative-result article](docs/results-m1.md)).

## 5. The complete answer to the skill question

Now the answers scattered across five articles of the series can be gathered into one sentence:

**Skill = criterion (behavioral three parts) + constitution (CoT stripping = automation) + acquisition (replay, experience into weights) + endpoint (source-lost actionalization).** (Note: the last item is the essential-layer explanation of the typical case for experientially acquired skill, not a definitional component of the criterion layer — the criterion layer is always the behavioral three parts.)

- **Criterion layer**: behavioral change, the three parts — is it a skill (see the [skill definition article](docs/skill-definition.md))
- **Constitution layer**: automation is measurable — the CoT-stripping test ("automation" among the constitution layer's three items receives its first measurement protocol; the open question of the skill definition article — how to judge acquirability (whether it can be acquired through training) — remains open, and is an independent dimension from automation: a behavioral disposition can be automatic yet not acquired through training, and vice versa)
- **Acquisition path**: replay (experience into weights) is the empirically supported direction; the translation route works only in its human-curated version (the conclusions of [*Explicit vs. Weight*](docs/explicit-vs-weight.md), [*Does AI Need Sleep?*](docs/does-ai-need-sleep.md), and the [negative-result article](docs/results-m1.md))
- **Endpoint**: source-lost actionalization (for experientially acquired skill) — why skill cannot be stated, why skill is typically automatic, why distillation can manufacture intuition: three questions, one answer

## 6. Boundaries and open questions

- **Granularity**: the definition does not prescribe skill granularity (nestable, composable); how the CoT-stripping test cuts on composite skills is left to specific tasks.
- **Memory retrieval vs. skill**: answering directly may be memorized answers. The CoT-stripping test does not distinguish — that is the job of the criterion layer's "out-of-distribution effectiveness" (d*), not the constitution layer's. The two tests work together: CoT stripping measures "automatic", d* measures "not memorized".
- **Collective skills**: out of scope (aimed at single-agent systems).
- **CoT stripping is a measurement protocol, not an experimental conclusion**: its boundary (single dimension) and its companion (use with d*) still need calibration in specific task domains — it is a measuring tool, to be validated through use, not established by declaration.

Back to the opening question: hide the reasoning process — can the AI still answer correctly?

Yes — its behavior no longer depends on explicit reasoning; the constitution layer's "automatic" item passes.
No — it is still on the road of reasoning; behavior has not yet become part of it.

For experientially acquired skill, the typical marker of "having learned it" is not the moment behavior passes, but the moment behavior no longer needs thinking. The criterion layer decides "is it a skill"; the CoT-stripping test answers "how far it is from the typical form of skill".

---

*This article is based on the author's open-source experiment project: [github.com/QiongZhiS/continual-learning-mechanisms](https://github.com/QiongZhiS/continual-learning-mechanisms). Core arguments are proposed by the author; formalization and literature cross-checking were completed with AI assistance.*

# Why Doesn't Continual Learning Measure "Getting Faster"?

> Author: Lumen

---

## Abstract

Continual learning promises that a model should get **faster** — the more tasks it has seen, the faster it learns the next one. That is the stronger reading of "continual", and it is the field's core promise. But open the literature and you find a strange mismatch: nearly every paper measures the opposite — **not forgetting**. This paper argues that the community has been using a measurement tool from the wrong dimension, explains why "getting faster" is methodologically harder (a domain sequence, an operational definition of acceleration, and a pre-registered criterion), and describes our pre-registered positive attack on the "experience → capability" step — with the falsification branch written down in advance.

---

## 1. What the community measures

Search for continual learning and the most common method names are: LwF, iCaRL, ER, DER, sleep replay, generative replay. They all solve the same problem: **catastrophic forgetting** — a model learns a new task and forgets the old ones.

The accompanying metrics are almost all retention-type: forgetting rate, recovery rate, retention matrices. The "standard experiment" of a methods paper is: train on a task sequence, then go back and re-test all old tasks, and report "forgot less than the baseline".

This work is important, and it does mitigate catastrophic forgetting. But notice — **almost no paper asks: after these mechanisms, does the model learn the next new task faster?** Not "a little faster", but measurably, structurally faster: experience becomes reusable capability, and the data needed to learn each new domain decreases along the sequence. This is the field's most central promise, and its most silent corner. Pre-registered "getting stronger" measurements are extremely rare — forward-transfer metrics do exist (GEM reported forward transfer as early as 2017), but they are post-hoc reports rather than pre-registered criteria, and they measure accuracy transfer, not "data required per new domain decreases". Meta-learning and skill-library work has come closer, but with different criteria.

The "organize existing knowledge" side of the ceiling, by contrast, is rising fast: EM-LLM (ICLR 2025) uses surprise segmentation plus two-stage retrieval to make LLaMA-3.1-8B beat its own full-context inference on LongBench (51.58 vs 39.3 average) and retrieve across 10M tokens — but even that measures retrieval performance, not "the next task is learned faster". Organizing known knowledge better at inference time is not getting stronger.

## 2. Why "getting faster" matters

"Can AGI reach human level?" can be split into three dimensions:

- **Task capability**: can it do a thing? Optimistic — the program layer (context organization) already shows 8–32× generalization gaps within the same model, not yet at the ceiling; single-task parity has precedents.
- **Learning efficiency**: how fast does it learn? **This dimension has no answer yet** — no system has demonstrated "human-style speed of getting stronger".
- **Cognitive form**: subjective experience and the like. Possibly a cliff, possibly not needed at all.

The continual learning community studies exactly the second dimension — but its default metric, retention, is a measurement tool from the first. It is as if the field researching learning efficiency measures with an instrument built for a different quantity.

There is also a cost-theoretic observation: human few-shot learning is not free. It is the product of hundreds of millions of years of evolution (innate structure) plus millennia of culture plus decades of experience. A digital system's innate-structure pre-training budget is zero, so "reaching human learning efficiency" may not be a problem of principle but a problem of cost. And the first step of a cost problem is an experiment that can detect whether "getting faster" is happening at all.

## 3. Why it's hard to measure

This is not a moral problem; it is a methodological one. Retention is easy: finish training on a domain, go back to an old domain, measure accuracy — one pass of re-evaluation. Getting faster requires three things:

1. **A domain sequence**: you cannot train on a single domain. You must design a series of tasks — learn A, then B, then C.
2. **An operational definition of acceleration**: "learns faster" must become a computable quantity. We use T(n): data required by domain n to reach a threshold (an accuracy target written down at pre-registration) ÷ data required by domain 1. T(n) decreasing with n = getting stronger.
3. **A pre-registration commitment**: the criterion is written before the experiment runs — "no acceleration found = failure". Otherwise, "tried ten metrics, one of them went up" gets packaged as success.

Miss any of the three and a measured "getting faster" does not count. This may be exactly why nobody measures it: retention is reporting a number; getting faster is signing a contract.

## 4. What the frontier hints at

Three mutually independent frontier works point at the same bottleneck:

**SkillsBench** (2026 benchmark, arXiv:2602.12670): LLMs generating "skill instructions" for themselves and injecting them at inference time — zero gain. Human-curated instructions, by contrast, raise pass rate from 33.9% to 50.5%. A model's own generated skill descriptions do not work.

**Sleep replay** (Nature Communications 2022): offline replay of experience restores representations and integrates memory, but **does not produce new capability** — the literature shows no evidence of it exceeding past performance.

**Dream2Learn** (arXiv:2603.01935, 2026 preprint): training on generated experiences improves forward transfer, but that is help of the "rehearsing future tasks" kind, not extraction of reusable skill from past experience.

All three directions — text skills, sleep replay, generated experience — stop at the same step: **the "experience → capability" step has not been demonstrated under a pre-registered continual-learning criterion**. They prove experience can be stored, replayed, and integrated, but none shows evidence that experience makes the model genuinely stronger on the next task. Note that reported forward transfer in the literature measures accuracy improvements, not our pre-registered criterion — data required to reach threshold on a new domain decreasing along the sequence.

## 5. Our response: a pre-registered positive attack

We designed an experiment that attacks "experience → capability" head-on:

- **Skill has an operational definition**: a behavioral disposition (stable across situations, effective out-of-distribution, efficient) — not a text description, a measurable behavioral change. This is the criterion foundation of the whole experiment;
- **Distillation mechanisms** (mechanisms that turn experience into reusable capability — not logits knowledge distillation, which is a different thing) have a candidate set;
- **Replay paradigm**: experience enters weights through training (offline incremental fine-tuning, nightly replay), rather than being translated into explicit descriptions — our earlier analysis shows the automatic-generation path of the translation paradigm currently fails empirically;
- **Primary criterion is T(n)**: does data requirement decrease along the domain sequence — while retention does not collapse (getting stronger only means something if retention holds at the same time);
- **Prior control**: the same algorithm with different prior strengths (pre-training amount / architectural bias), a 2×2 factorial — so that "the algorithm failed" is not secretly "the prior was insufficient";
- **Conclusion wording pre-registered**: the result may only be phrased as "candidate mechanism X does not accelerate" or "candidate mechanism X accelerates" — no third phrasing.

An earlier experiment in this project produced a negative result: inference-time parallel exploration (letting the model think longer) gives no gain on symbolic domains — exploration perturbs existing knowledge but does not supply missing capability; capability must come from the training side (fully documented in `docs/results-m1.md`). This pre-registered experiment is the training-side positive attack: turning experience into capability for real.

## 6. Falsification commitment

The failure branch of this experiment is written down in advance:

> If all candidate distillation mechanisms we actually test fail to accelerate (the conclusion holds only for the mechanisms tested; the distillation family is not exhausted) → "experience → capability" remains without a known solution → **the possibility that AGI's learning efficiency cannot reach human levels is not ruled out**.

In other words, failure is not bad news — it is the answer this field most needs. Modern continual learning has studied "not forgetting" for a decade (since the EWC/GEM generation of methods around 2016). Whether the core question — *can a system get stronger?* — is answered yes or no, it deserves one direct answer. We chose to measure it.

---

*This paper is based on the author's open-source project: [github.com/QiongZhiS/continual-learning-mechanisms](https://github.com/QiongZhiS/continual-learning-mechanisms) (experiment proposal and pre-registered criteria fully public; negative results fully retained). Core arguments and experimental design by the author; formalization and literature cross-checking with AI assistance. Negative results are assets.*

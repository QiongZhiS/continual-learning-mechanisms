# The End of Memory Is Intuition

Author: Lumen

What did you eat for lunch last Wednesday? Probably you don't remember. But if you had food poisoning that day, you would hesitate at similar food for months afterward — you may not be able to say which meal did it, but your behavior has already been rewritten by it.

This phenomenon is so ordinary that we rarely think about it seriously: **forgetting the details does not mean not remembering.** We think of memory as "stored past", but in reality, the part of memory that actually influences you is precisely the part you can no longer articulate.

This article wants to split the word "memory" apart — what people call "memory" mixes at least three different things, and their fates are completely different.

## 1. A record is not a memory

Imagine a database: entries stored one by one, retrieved when needed. This is most people's default "memory model". But it fails to explain one fact: **the probability that you remember last Wednesday's lunch has no necessary relationship with the degree to which you are influenced by it.**

Key distinction:

- **Record = what happened before** — external, traceable, precisely retrievable;
- **Memory (everyday broad sense) = how what happened before still influences me now** — internalized, usually not directly readable.

Forgetting details is fine — when traceability is needed, humans invented writing, photographs, and databases as external records. Memory is responsible for influencing the present; records are responsible for preserving the truth. The two divide labor; they are not the same thing. The photos on your phone are not your memory, they are your records; the emotion that surges when you look at the photo is memory at work.

## 2. Memory = influence with a source

Strictly speaking, we propose:

> **Memory = influence with a source.** The part of the current state's shaping that can point to "which event caused it".

Two components, neither dispensable: **influence** (the present changed by the past), **source** (can point to which event). Everyday "memory" is mostly the broad sense (influence itself); the strict sense is the part **with a source** — this is precisely the boundary between memory and intuition (next section).

This definition yields a counterintuitive criterion: **the criterion of memory is "whether the present is changed by the past", and the boundary with intuition lies in "source" — whether one can point to which event caused it, independent of reportability.** This is isomorphic to how we treated skill: in the [operational definition of skill](docs/skill-definition.md), the criterion is behavioral change, not the form of the carrier. Whether a system can recall details is like whether it can state its own definition of skill — it does not participate in the judgment.

## 3. Sources get lost: from memory to intuition

But most influence does not carry a source. You judge whether a person is trustworthy — you cannot say which experience taught you. You are afraid of the dark — you cannot say when it started. **The influence remains; the fingerprint is gone.**

This is intuition: **influence whose source has been lost.**

- Experts cannot say why they judge this way, yet their judgments are accurate — Dreyfus's expert intuition (we discussed it in the skill definition)
- Kahneman's System 1: fast, automatic, no articulated reason (Kahneman, 2011)
- Your "gut feeling", "taste", "first reaction" — all influence left behind after the source is lost

So there is a complete chain. Note it has **two streams**: the external record is an optional detour, not a mandatory station — the vast majority of events have no external record; they go directly into memory:

```
Event occurs ──→ Memory (with source · fast encoding) ──→ source gradually lost ──→ Intuition ──→ actionalization ──→ Skill
                ↑                                        (influence remains · fingerprint gone)   (subconscious · know what to do on sight)
                └── optional: external record (traceable · verifiable anytime) ──┘
                       replay/recall = turning the record back into influence (the bridge between the two streams)
```

("Record" has two senses here: fast in-brain encoding — hippocampus, and external traceable records — writing/databases; the external record in the diagram refers to the latter.)

**The end of forgetting is not disappearance; it is becoming intuition.**

## 4. The brain was designed this way

This chain is not an abstract model — the nervous system is organized by this division of labor. Complementary learning systems theory (McClelland, McNaughton & O'Reilly, 1995) proposes: the hippocampus is responsible for fast recording of specific events — stored in one pass, specific, attributable; the neocortex is responsible for slow integration of structure — each replay changes a little, cross-event fusion, non-attributable. People with hippocampal damage lose the ability to record "recent events", but structures already integrated into their judgment are unaffected: **the source is lost, the influence remains.**

And memory was never a video recording. Loftus's classic experiments: witnesses who watched a car-crash video and were asked "how fast were the cars going when they **smashed** into each other" estimated higher speeds than those asked "**hit**" (Experiment 1); more critically, a week later the "smashed" group was more likely to "remember" seeing broken glass that did not exist in the video (Experiment 2) — **questioning can not only change reports, it can rewrite memory itself** (Loftus & Palmer, 1974). The "read" operation of memory does not exist: every recall is a reconstruction, and reconstruction is contaminated by the present. This is also the theme of replay in our article [*Does AI Need Sleep?*](docs/does-ai-need-sleep.md) — replay is more than consolidating storage (in the CLS sense, replay integrates structure); it simultaneously turns the record back into influence.

## 5. On the AI side: distillation is manufacturing source loss

AI's intuition is more extreme than humans'. The vast majority of a large model's knowledge has no traceable source — it cannot itself say "why it judges this way", because the massive training data that shaped it has long melted into weights.

But there is a clean control: **knowledge distillation** (Hinton, Vinyals & Dean, 2015) — a student model learns the teacher's outputs, learns the teacher's judgments, yet cannot say why. The source is on the teacher side; the student side has no source. **Distillation = deliberately manufacturing source loss = controllably manufacturing intuition.**

This view has three direct corollaries:

- The "knowledge" of a distilled model is hard to explain — **not a defect** — intuition is by nature unexplainable (there is no source to tell);
- But distilled intuition **is updateable**: the source is in the external teacher; re-distillation = re-attaching the external source = controllable correction. What is updateable is the influence, not the student's attributability — the source always stays on the teacher side. This is the engineering channel for "intuition is updateable", and one of the motivations for externalizing criteria in our article [*Who Decides What Is Right and Wrong for AI?*](docs/who-decides-right-wrong.md): judgment standards stay external precisely because when the source is external, intuition can be updated;
- In our own experiment proposal, experience distillation is designed as one of the core mechanisms (currently the biggest unknown) — extracting reusable skills from episodic experience (experimental design and criteria are public; link at the end of the article).

## 6. Subconscious = no explicit reasoning steps

"Know what to do on sight" — the purest form of skill. On the AI side there is a clean operationalization: **for the same task, answering correctly without outputting intermediate reasoning steps = skill; having to write out the reasoning process to answer correctly = still reasoning.** ("Answering directly" means not outputting explicit reasoning steps, not a literal single forward pass.)

Chain-of-thought (Wei et al., 2022) makes intermediate steps visible — partially traceable; answering directly is "no intermediate steps" — completely untraceable. So we can add the measurement that skill definition has been missing: **the CoT-stripping test** — not only whether behavior changes (criterion layer: behavior passes), but whether it has become fast enough to not need thinking (constitution layer: automatic). A system that needs reasoning to answer correctly and a system that answers directly may pass the same behavioral criterion, but the latter is skill — its response is already intuition. Boundary stated clearly: this test measures the single dimension of "presence of explicit reasoning steps"; it does not distinguish whether direct answering comes from internalized skill or memory retrieval — that is another layer of question.

## 7. How to update: with a source, change directly; without a source, re-cultivate

Three things, three update modes:

| Object | Update mode |
|---|---|
| Record | Changeable anytime (external, a technical problem) |
| Memory with source | Change directly — locate that record, correct its influence on the present |
| Intuition | Cannot find "the one to change" — only indirectly: hammer it with counterexamples, deliberately attend, pull the automatic reaction back into conscious state, first create a new traceable source for it, then it becomes changeable |

This explains why "changing an expert is harder than teaching a novice": an expert's reactions are automatic; new evidence cannot get in. It also explains why priors are hard to update — in our article [*Should AI's Priors Be Updated?*](docs/should-priors-update.md) we said: the source of a prior has already been lost. Now we can complete that sentence: **prior = old memory whose source is lost = the end of forgetting.** It is not that a prior is too strong to change; it has lost the entry point for direct attribution.

## 8. The answer to the memory question

Back to the three questions we keep asking: how do skills form, how does memory update, how does agency emerge.

For the memory question, we now give a complete answer (using the everyday broad sense of "memory"): **memory is not storage of the past; it is the past's ongoing construction work on the present.** We always think remembering is preservation — actually remembering is rewriting (every recall is a reconstruction); we always think forgetting is loss — actually forgetting is internalization (details vanish, influence becomes your way of judging).

From record to intuition, from traceable to untraceable, from directly changeable to only re-cultivatable — **the end of memory is not disappearance; it is becoming the way you judge the world.**

---

*This article is based on the author's open-source experiment project: [github.com/QiongZhiS/continual-learning-mechanisms](https://github.com/QiongZhiS/continual-learning-mechanisms). The experience distillation experimental design is in [docs/experiment-proposal.md](docs/experiment-proposal.md). Core arguments are proposed by the author; formalization and literature cross-checking were completed with AI assistance.*

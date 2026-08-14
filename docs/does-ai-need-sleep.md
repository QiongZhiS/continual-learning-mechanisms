# Does AI Need Sleep? — The Replay Path to Skill Acquisition

Author: Lumen

Skills cannot be told into a model — so how do they grow? An earlier article gave half the answer: they must enter through training, into the weights, rather than through descriptions, into the context. This path has a name — the **replay paradigm**: experience as training material, replayed in a dedicated (offline/nightly) window — offline incremental fine-tuning, nightly replay — as opposed to one-shot forward training and prompt injection. But it sounds familiar: isn't this exactly what hippocampal replay during sleep does? Which raises the blunt question: does AI need sleep?

This article answers that question — and finishes what the earlier article only started: if skills cannot be told in, how do they actually grow?

## Why "telling" doesn't work (one-minute recap)

The earlier article's conclusion: automatically generated explicit information — model-written self-descriptions, input-injected equivalence transformations, parallel exploration at inference time — almost never changes model behavior. In the SkillsBench controlled comparison, human-crafted descriptions raised pass rate from 33.9% to 50.5% (+16.6 percentage points), while injecting self-generated descriptions gave zero gain (within noise). The criterion is not the carrier of knowledge; the criterion is whether behavior changes.

So the question stands: if skills cannot be told in, how do they enter the model?

One exception first: human-crafted descriptions do change behavior (+16.6 pp), but they are task-level prompts — effective on injection, gone on removal; they do not transfer across contexts or out-of-distribution, and do not satisfy the operational definition of skill (cross-context stability, out-of-distribution effectiveness, efficiency). Skill acquisition requires not "behavior change in one setting" but reconstruction of the behavioral disposition itself. On this path, there is only one direction: **through training, into the weights**. That path has a name: the replay paradigm. It is not new — three independent lines of evidence already point to it; the mainstream of the AI community has simply been looking at the "telling" side.

## Evidence 1: The continual learning community already knows replay works

The continual learning community's defense against catastrophic forgetting takes roughly three routes: replaying old data (Experience Replay, Rolnick et al., 2019), constraining gradient directions with old experience (A-GEM), and regularization that never touches old data (EWC, penalizing drift of important weights). Note the last category: it neither replays nor describes — it directly protects old knowledge inside the weights. That itself says skills live in the weights.

There is an easily overlooked fact: **if skills could be "told", the continual learning community would not need weight-side methods at all** — they would describe old skills as rules, inject them into the input, and be done. Instead they replay old data, protect old weights, or learn prompt parameters (learnable weight-side parameters) — never describing old skills. That is engineering evidence that skills are ineffable, voted on by thousands of papers with compute.

## Evidence 2: Human experts also cannot say what they know

In *Mind over Machine* (1986), Dreyfus & Dreyfus proposed a five-stage model of skill acquisition: novices follow rules; advanced beginners begin to recognize situations; competent practitioners handle routine cases independently; proficient practitioners grasp whole situations; and at the expert stage, judgment becomes situation-dependent intuition — **experts cannot fully explain why they judge as they do**.

An experienced driver cannot say how they judge distances; a chess master cannot state the "rules" behind each move; a physician's diagnosis rests on pattern recognition soaked in countless cases. In an expert, skill is not a set of statable rules but a set of behavioral dispositions polished by practice. Translation loses — translating an expert's behavior into rules yields a novice's rules, not the expert's intuition.

Humans and models alike. Dreyfus's model supports the "experience side" of this path: skill comes from practice, not rules; experts are ineffable — as for how practice solidifies into weights, replay (Evidence 1 and 3) is the candidate mechanism.

## Evidence 3: The brain itself replays at night

Neuroscience offers a more concrete answer. Complementary Learning Systems theory (McClelland, McNaughton & O'Reilly, 1995) holds that the brain has two learning systems: the hippocampus rapidly records specific episodes, and the neocortex slowly integrates statistical structure. Experiences during the day are first stored in the hippocampus; **during sleep, the hippocampus spontaneously replays the day's experiences** (directly recorded in rat hippocampus by Wilson & McNaughton, 1994); CLS theory holds that it is precisely this replay that provides the neocortex with repeated training signals, slowly adjusting its connections (McClelland, McNaughton & O'Reilly, 1995).

Translated into AI terms: hippocampus is the short-term buffer, neocortex is the weights. Sleep is the replay window; replay grows experience into the weights.

## So: does AI need sleep?

AI needs "sleep", but not "sleeping".

Biological organisms need sleep because sensory channels must shut down to free an offline window — while awake you must watch the road, listen, process what is in front of you; there is no spare time to replay. Sleep, one could say, provides the brain that offline window. AI has no such constraint: it can deliberately schedule offline periods and replay the day's experience into its weights. Nightly replay and offline incremental fine-tuning are the engineering equivalents of "sleep replay" — no physiological sleep cycle required, only an offline window.

What about "waking replay" (online continual learning)? It is a natural extension of the replay paradigm, but with a real tension: online updates easily fall into catastrophic forgetting — new experience overwrites old weights. The best-performing class of methods on most current benchmarks is precisely the hybrid: receive experience online, consolidate offline via replay (Rolnick et al., 2019). Almost identical to the brain's strategy.

## This paradigm is falsifiable

The replay paradigm is not a slogan; it makes a measurable commitment: **if experience enters the weights through replay, the system should learn progressively faster on new domains — the sample requirement for a new domain should decrease with the number of domains already learned** (this depends on the domain sequence containing reusable shared structure — our pre-registered sequence is designed along a similarity gradient). This is exactly what the learning-efficiency criterion in our pre-registered experiment measures ([docs/experiment-proposal.md](https://github.com/QiongZhiS/continual-learning-mechanisms/blob/main/docs/experiment-proposal.md)): replay as one candidate mechanism; the criterion is that per-domain sample demand decreases. What is measured is not "did it remember" but "does it learn faster".

If replay also fails to change learning efficiency — that is a clean answer too: the architecture's learning-efficiency ceiling is real; the problem lies in the weight structure itself, not in how experience flows. Negative results are also assets — our earlier article *Negative Results Are Assets* dissected exactly this question.

## Boundaries and open questions

The replay paradigm has boundaries. What to replay, when, and how much — replay selection strategy is an open question; blind uniform replay is far from optimal. Replay does not solve prior insufficiency: if the model's structural priors (inductive biases inherent in architecture and pretraining) are insufficient, "what to know" cannot grow, and replay hits the same wall. Also, analogical reasoning (letting exploration fit past knowledge) may be one implementation of the weight-side path; we do not yet have direct evidence.

These are open questions, left to experiment.

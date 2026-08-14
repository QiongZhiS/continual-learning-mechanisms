# Should AI's Priors Update?

Author: Lumen

Pretraining gives an AI a pair of glasses through which to see the world — but the world changes. Should the glasses be replaced?

If you use AI regularly, you may have noticed: once a model is released, it is largely "frozen". What it knows is fixed at the moment training completes. Subsequent updates are mostly reskins, added memory, and prompt engineering — the underlying "how it thinks" rarely changes.

Machine learning has a word for this underlying "how it thinks": the **prior** — from Bayesian statistics: the assumptions you already hold about the world before data arrives. It is the assumption the model carries before seeing any data — a basic stance toward the world. (Strictly speaking, pretraining yields "starting assumptions learned from large-scale data", but it plays the role of a prior for downstream tasks.)

This article examines a counterintuitive question: is a stronger prior always better? And should it be updated at all?

## What a prior is: two examples

Start with a non-AI example. A doctor with thirty years of experience always begins diagnosis with the disease patterns he knows best. That experience is an asset in a rural clinic: common diseases are handled quickly and accurately. But place him in a hospital with many rare diseases, and his first judgment is often wrong — not because he has grown stupid, but because the "prior" he carries does not match the world in front of him.

AI pretraining is an amplified version of this logic. Pretraining floods a model with vast text and code, letting it internalize "what the world roughly looks like" — syntax, common sense, reasoning patterns. That is its prior. No matter how it is later fine-tuned or prompted, this foundational assumption remains.

**The strength of a prior is how deep and how fixed this foundational assumption is.** The larger the pretraining budget, the stronger the prior usually is.

## Prior is an asset: why the community piles it on

A prior is not optional. One of the clearest lessons of recent years: **a model cannot learn what it does not "already know"**.

We ran a series of experiments on small models (full code and data in the public repository): adding rules to the input side, letting the model explore more steps at inference, having it write its own skill descriptions — almost all ineffective. The model simply did not learn. But once the corresponding structure (prior) was given in the training data, the same task became easy immediately.

A more widely known piece of evidence: large language models show zero gain from self-generated skills (within noise), while human-crafted skill descriptions improve performance from 33.9% to 50.5% — same carrier, same injection method, only the source differs, and the results differ completely. Note: this is still prompt-level behavior change, not skill acquisition in the weights; for capability to truly grow into the weights, training remains the only path.

So the community's logic in piling on pretraining budget is correct: **prior insufficiency is the root cause of many failures.** The problem lies in the next inference.

## The liability of strong priors: the world drifts

"Stronger priors are always better" has an implicit premise: **the world does not change**.

But it does. In machine learning this is called **concept drift** (Gama et al., 2014): the regularities present when you train a model and those it faces after deployment may no longer be the same.

At that point a strong prior turns from asset into liability. An old prior processing a new world errs **confidently** — it is not ignorant; it carries an outdated but internally consistent explanatory framework.

Humans are the best example. A person's understanding of society largely takes shape before adulthood — that is his "strong prior". When social structure changes, his cognition often lags by many years. It is not lack of effort: **lag is a structural consequence of strong priors** — the old framework actively digests new information, explaining away whatever does not fit.

Models are the same. Forcing the regularities of context A onto context B is negative transfer — the other face of overgeneralization.

Why is a prior so hard to update? Beyond "the old framework digests new information", we offer a source-level explanation: **the prior's origin has been lost**. The old doctor cannot say which cases shaped his diagnostic habits — not that he forgets details, but that the shaping process is no longer identifiable. Untraceable origin means there is **no "entry point" to modify**: you can argue against a specific claim that carries its source, but you cannot argue against a framework that has no source to cite. We therefore hold that prior updates can basically only proceed indirectly — through repeated impact by counterexamples and new evidence, letting it regain traceable origins. This is why prior updating requires a dedicated mechanism: **the deep root of difficulty is that the prior has lost its directly addressable entry point.** Whether this explanation is correct is left to experiment.

## The community is already "using priors" — in the wrong direction

The continual learning community has an interesting pattern: they have long used priors, but for *retention*, not for *growth*.

The classic EWC (Elastic Weight Consolidation, Kirkpatrick et al., 2016) has a Bayesian interpretation: the parameter distribution after learning an old task is the prior for the new task — weighted by Fisher information, telling the model "these parameters should not move". More explicitly, VCL (Variational Continual Learning, Nguyen et al., 2017) directly treats the old task's posterior as the new task's prior, updated iteratively.

These are engineering instances of "updateable priors" — more precisely, "updateable *task-parameter* priors" (what "updateable" means is distinguished precisely in the next section). But their direction is uniform: **use the prior to fight forgetting** — preserve old knowledge. They update the prior over task parameters, not the prior-update mechanism itself (the learning structure). Nobody systematically asks: should prior updating be designed to track a changing world, rather than weld old knowledge in place?

This is isomorphic to the phenomenon we criticized earlier: the continual learning community spends most of its effort on "retention", almost none on measuring "growth". Prior updating is exactly the ignored half of "growth".

## Updateable priors: the strength axis and the update axis are two different things

Now a precise distinction:

- **Strength axis**: how deep and fixed the prior is (pretraining budget, architectural bias)
- **Update axis**: whether the prior can be modified by new evidence, and at what rate

The community optimizes only the strength axis. What AGI needs, arguably, is the update axis.

Neuroscience has a ready-made term: **meta-plasticity** — the plasticity of plasticity (Abraham & Bear, 1996) — not "can it learn" but "can the learning mechanism itself be learned". The lag of human cognition behind social-structure change shows that the human prior-update mechanism is slow (this is an update-axis problem — lag means slow updating, distinct from the "strong prior actively digesting new information" mentioned above, which is exactly what this article wants to measure separately); but slow ≠ non-updateable. If AI inherits only the "strong prior" without designing a "prior-update mechanism", it inherits the human lag as well, and more thoroughly — after release, the prior simply never updates.

In our own approach, there is a candidate mechanism: a **self-model** — a lightweight module that continuously monitors "how well it is learning", recursively updating beliefs from new evidence (each round: belief = old belief + new evidence). It does not manage learning; it manages **the update of the prior**.

## This is falsifiable

"Strong priors are a liability in far transfer" is not a slogan. We pre-registered a falsifiable criterion (pre-registration = fixing criteria before running experiments, to prevent post-hoc redefinition):

The same learning algorithm, with four levels of prior strength (weak base / pretrained / architectural bias / both combined), is tested on near- and far-transfer domain sequences. **If the "strong prior" group is significantly worse than the "weak prior" group on far-transfer domains, the strong-prior liability receives direct evidence; if there is no difference, this inference is wrong.**

Criteria and the full experimental design are in the pre-registration document in the public repository — a commitment that can be run and refuted.

## But not all priors should update

One necessary boundary: **not every prior should follow the world.**

"The world's regularities" should update — when new evidence arrives, old assumptions must yield. But if everything drifts with the current, the system loses its bearings. Our own design has a two-tier structure: **the core layer is almost non-updateable (requires an extremely high threshold); the edge layer can be updated at low cost (one user confirmation)**. The core layer does not hold "beliefs about the world" but the criterion of "what should be updated" itself — the content beliefs carry the far-transfer debt (they should follow the world), while the criterion is the benchmark of update operations (it must be stable). The layering governs the update axis, not the strength axis. The reason is simple: if the criterion "what should be updated" itself follows the updates, then updating has no benchmark at all — prior drift becomes value drift.

The true form of the updateable prior is not "changeable at any time", but **"knowing what should change and what should not, with this layering itself stable"**.

## Boundaries and open questions

This article leaves more open questions than answers:

- How should the prior update rate be set? Too slow = lag; too fast = instability (we have a time-scale layered design in pre-registration)
- What is the relation between prior updating and forgetting? Is updating a prior itself a form of "directed forgetting"?
- Is the human prior-update mechanism (slow, layered, lagging) a defect or a feature? If AI learns even this, what will AI become?

These are experimental questions. We are approaching them one by one, in falsifiable ways.

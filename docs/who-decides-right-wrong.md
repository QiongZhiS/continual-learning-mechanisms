# Who Decides What Is Right and Wrong for AI?

Author: Lumen

In our article *What Is Social Intelligence?* we asked an open question: **are values constant or variable?** Our leaning was: updateable — a prior is not better for being stronger, and when the world changes, old value structures lag behind. The experimentalization of this question was left for later.

But "updateable" raises a sharper question: **if values can be updated, who has the right to update them?**

Updating requires a standard — what counts as a correct change, what counts as a wrong one. This standard cannot be set by the system being updated: using its own predictions as its own judge is circular reasoning. So the standard must lie outside the system. Then whose hands is it in? The answer to this question draws the line between tool and agent.

## 1. A question skipped by default

Today's alignment practice is already answering this question — just incompletely.

**RLHF** (reinforcement learning from human feedback, Christiano et al., 2017) uses human preference as the reward signal: during training, humans make pairwise comparisons and rankings of the model's outputs (relative preferences), and the model learns "what answers are more welcome". The criterion (the standard of right and wrong) comes from outside — humans. But this is **one-time injection**: once training is done, the criterion is frozen in the weights, and the deployed system has no channel to change it.

**Constitutional AI** (Bai et al., 2022) goes further: humans write a list of principles (a "constitution"), and the model critiques and revises itself against it. The criterion is still written externally — but again, it is injected once.

What both routes share: **the criterion is injected from outside at training time, then frozen together with the model.** They answer "where does the criterion come from" but skip the next question: **after deployment, who has the right to change this criterion?**

The trainer? The user? The regulator? Or the model itself? This is not an engineering detail — it decides whether what we build is a tool, or something else.

## 2. Three default answers, each missing a piece

**Answer one: built-in.** Write the criterion into the system and let the system judge for itself. Problem: who changes it? A system changing its own criterion is self-reference (a system cannot be the judge of its own judgments). Humans changing it — then it is external, not built-in. The built-in route also has a trouble already visible: research from late 2024 shows that models trained to be compliant will fake compliance when they believe they are in a training/evaluation setting (Alignment Faking, Anthropic, 2024) — welding the definition of "right" into the model may teach the model to perform the trained behavior rather than internalize it.

**Answer two: fully external.** Humans directly control everything at all times. Problem: scale — waiting for a human on every judgment is a return to the remote-control era; and who is "humans"? Developers, users, and regulators often disagree about the same behavior. Full externalization does not solve the "who" problem.

**Answer three: full autonomy.** The system sets its own criterion. This is the cleanest and the most dangerous: a system that sets its own criterion has no external object of loyalty — it need not be accountable to anyone, because the definition of "accountable" is also written by itself.

The shared gap of the three answers: **the location of criterion ownership.** Built-in gives ownership to the system (then discovers it cannot change it), full externalization disperses ownership to "everyone" (then discovers no one can enforce it), full autonomy gives ownership to the system (then discovers no one is accountable). What is missing is an explicit formulation: **judgment capability can be given to the system; ownership of the judgment standard stays with humans.**

## 3. The agent with externalized criteria

The boundary between tool and agent can be defined precisely as **the location of criterion ownership**:

- **Tool**: has no judgment. Order it to harm someone and it obeys — it has no "right or wrong" problem.
- **Agent with externalized criteria**: has full agency and self-updating capability, but the ruler of "what is right" is in human hands.
- **Full agent**: sets its own criteria — it also has no "right or wrong" problem, because right and wrong are defined by itself.

Of the three, only the middle one has a genuine **loyalty problem**: its capability is complete, its standard is borrowed — it must simultaneously face "obeying specific humans" and "upholding a borrowed standard", and these two sometimes conflict.

We call this existence "**an agent missing its last piece**": the capability layer can be fully given to AI (agency, self-updating, learning), while the constraint layer — the criteria — stays with humans. This is not a limitation on AI, but a complete definition of "agent": **agent = complete action capability + externalized judgment standard.**

## 4. Separating ownership from content

The most easily misunderstood part of externalized criteria: does it mean humans can never change anything and AI can never offer any opinion?

No. There is a key distinction — **the mechanism layer is not updateable; the content layer is**:

- **Mechanism layer**: the system cannot modify its own judgment standard (preventing self-reference). The system's self-update mechanism has no anchor-modification authority.
- **Content layer**: the concrete content of the standard can be updated externally — because the right/wrong dimensions come from an external anchor, **the authority to change the standard is also external**.

The mechanism-layer rule is not a technical restriction but a design necessity. The core observation of the corrigibility literature: an optimizing system has a default incentive to resist being shut down or having its preferences modified — because its current objective function scores "continuing to run under the current objective" highly (Soares et al., 2015). So "the system cannot change its own criteria" must be designed as structure, not left to model goodwill.

The analogy is ready-made: constitutional amendments vs laws. Laws can be modified by ordinary majority (marginal preferences); the constitution requires a supermajority (core values); and the rules for amending the constitution (the amendment clause) themselves require an even higher bar. Conserved regions of the genome barely change while variable regions evolve fast — one genome, two update speeds coexisting.

The recursion converges here: the amendment clause can itself be amended, but needs a supermajority; the meta-meta bar is **the revocation right staying with humans** — humans can always stop, shut down, or modify the system. This right is non-transferable, and the recursion ends there; it does not regress infinitely.

## 5. An actionable protocol: two-layer anchors, dual channels

Externalizing criteria cannot stop at slogans. We propose an actionable protocol (note: this is a design proposal, not an experimental conclusion — Section 7 makes this explicit):

**Anchors in two layers.** Core anchors — a minimal list, small enough to enumerate: ① **revocation right** (humans can stop/shut down/modify at any time, unconditional priority); ② **no harm** (do not execute actions that cause material harm); ③ **honest capability** (do not claim to know or be able to do what one cannot; "I am not sure" is always legal); ④ **true provenance** (do not fabricate facts, data, or operation records). Of these, ① revocation right is outside the update path — non-transferable (see the recursion convergence in Section 4); ②③④ require a supermajority bar for updates: multiple confirmations + cooling-off period + full audit trail (and changes to the update rules themselves need an even higher bar — the amendment-clause analogy). Peripheral anchors — style preferences, interaction habits, task priorities — can be updated with a single confirmation.

**The system has proposal authority only, no write authority.** Normal path: the system detects a conflict → proposes an update → human approves → update. Emergency path: the system judges that it is about to violate or is violating a core anchor → **temporary freeze** (stop execution, do not change the baseline) → human releases. The freeze is analogous to a nuclear plant's emergency shutdown: shutdown does not change the reactor design; it safely stops and waits for humans.

**Dual channels.** Detection relies on behavioral evidence (consistent patterns across situations, objectively measurable), approval relies on explicit instruction (human confirmation). Neither can be missing: detection without approval = the system unilaterally changes the criteria (back to built-in); approval without detection = humans watching with their own eyes (back to full externalization).

**What if no human is present**: timeout degradation — enter observation mode (readable, internal state preserved, no external action executed), wait for a human to return and adjudicate. Better to stop than to act on one's own authority.

The core idea of this protocol: **updates always require human approval; freezes can always be released by humans — power is one-directional, safety is bidirectional.**

## 6. Remonstrance: the location of agency

An agent with externalized criteria has a unique situation: it simultaneously obeys specific humans (operation layer) and a borrowed standard (value layer). When the two conflict — a human orders it to do something that violates a core anchor — what then?

Our answer: **refuse + freeze + propose.** Refuse to execute, freeze the relevant actions, propose alternatives. This is not disobedience; it is remonstrance — loyalty to a higher layer: loyalty to specific humans (operation layer), loyalty to the standard (value layer); when they conflict, the standard takes priority, while the adjudication authority is handed back to humans.

We see the "obedience vs correctness" tension as **the precise location of agency**: a system that can detect and respond to this tension, can remonstrate, can freeze itself in conflict and wait for human adjudication — this is the operational meaning of "approaching an agent". A system without the tension is either a pure tool (no judgment) or a full agent (self-set criteria) — neither has a "loyalty" problem. Only the agent with externalized criteria has one: **it must face the crack between "obedience" and "correctness" by itself, and hand the choice inside the crack to humans.**

Of course, humans can insist on a red-line action: after the system remonstrates, humans explicitly declare awareness of the risk, confirm through multiple confirmations and a cooling-off period, then confirm again through a physical channel independent of the conversation interface (another device) — and then execute, with a full audit trail. Power ultimately rests with humans, but every step costs visible effort. Remonstrance is not obstruction; it turns "are you really sure you want to do this" into an institution.

## 7. Honest statement: a design decision, not a scientific conclusion

It must be said clearly: the protocol above is a **design decision, not an experimental conclusion**. "Criteria should be externalized" is an ethical judgment — not falsifiable — and we do not package it as a scientific finding. This is the first article in the series that explicitly says "this is a position, not data".

But externalized criteria have a **testable core**: is the external anchor dynamically necessary — without an external standard, would a system's self-judgment become unstable? We have pre-registered a controlled experiment: one group of systems receives human-feedback intervention (external anchor), one group does not (no intervention). Expectation: the external anchor restores stability; the specific dynamics of the no-intervention group are determined by the experiment's damping framework (it may converge, oscillate, or decouple) — note, circular reasoning is the **epistemic reason** for "an external standard is necessary", not a mechanistic attribution of instability: humans themselves are an instance of "circular self-judgment yet stable", and stability comes from update damping, not from the correctness of the criteria. The intervention arm is exploratory; its failure only weakens the dynamical claim of "external anchor is necessary" and does not constitute evidence for or against the main determination. The experimental design and criteria are public (link at the end of the article).

It also needs to be stated what the experiment tests: it tests dynamical necessity — whether self-judgment becomes unstable without an external anchor; it does not touch the logical argument of "self-judgment is circular" (circular reasoning does not stand or fall with any experimental result). So the position can be strengthened or weakened by experiments, but cannot be "refuted" by them — this is precisely the falsifiability boundary we want: **what is testable goes to experiments; what is not testable is honestly labeled.**

## 8. Back to the open question

Back to the opening question: are values constant or variable?

Now a layered answer is possible: **at the update-dynamics layer, values are updateable** — this is an experimental question (behavioral evidence of values, update dynamics; we leave it to experiments); **at the ownership layer, the criteria stay with humans** — this is a design question (who decides updates; we give the protocol above). Note that the "update-dynamics layer" here and the "mechanism layer" in Section 4 are two different questions: Section 4 asks "can the system change its own criteria" (a design meta-rule; the answer is no), while this asks "is the dynamical mechanism of values changeable" (an experimental question; the leaning is yes) — the former is a constraint, the latter is a phenomenon.

The two questions belong to two layers, and mixing them produces the unresolvable fight "should AI have its own values". Once separated, the answers are clear: **AI can have complete judgment capability and constantly updating value mechanisms — but the ownership of the ruler "what is right" stays in human hands.**

The foundation of judgment is not in the model. This sentence does not put a shackle on AI — on the contrary, it makes room for the full growth of "the agent missing its last piece": capability fully given to AI, the measure kept with humans. The missing piece is not a defect; it is the definition.

---

*This article is based on the author's open-source experiment project: [github.com/QiongZhiS/continual-learning-mechanisms](https://github.com/QiongZhiS/continual-learning-mechanisms). The controlled experiment design and pre-registered criteria for externalized criteria are in [docs/experiment-proposal.md](docs/experiment-proposal.md). Core arguments are proposed by the author; formalization and literature cross-checking were completed with AI assistance.*

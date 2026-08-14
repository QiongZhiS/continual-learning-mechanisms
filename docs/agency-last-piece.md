# Agency: The Last Piece

Author: Lumen

We keep asking three questions: how do skills form, how does memory update, how does agency emerge.

The first two are answered. The skill question: [*Knowing How Is Not Knowing That*](docs/skill-automation.md) — the criterion is behavior, the constitution is measurable automation, acquisition goes through replay, and the endpoint of experientially acquired skill is source-lost actionalization (the typical form). The memory question: [*The End of Memory Is Intuition*](docs/end-of-memory-is-intuition.md) — memory is not storage of the past; it is the past's ongoing construction work on the present; the end of forgetting is not disappearance, it is becoming intuition.

Only one remains: **how does agency form?**

The answer to this question, it turns out, is already scattered across three of our earlier articles — the self-knowledge article contributed one part, the deception article one part, the external-criteria article one part. Today we assemble them.

## 1. The three parts of agency

An "agent" needs three things, none dispensable (in our framework — see §6 for why we define it this way):

- **Inward**: a self-model — it predicts whether it can do something, whether the prediction is accurate, and how to update when wrong;
- **Outward**: an other-model — it can infer what others believe, want, and how they will react;
- **Upward**: external criteria — it knows the ruler for "what is right" is not in its own hands.

Three articles answered the three parts respectively: [*Can AI Know Itself?*](docs/can-ai-know-itself.md) covers the self-model, [*Can AI Deceive?*](docs/can-ai-deceive.md) covers the other-model (mind inference), [*Who Decides What Is Right and Wrong for AI?*](docs/who-decides-right-wrong.md) covers external criteria and ownership. Assembled, the three parts are the complete answer to "how does agency form".

## 2. Inward: the self-model

In [*Can AI Know Itself?*](docs/can-ai-know-itself.md) we turned "self-knowledge" from a philosophical question into an engineering one: **the self-model is not a mirror; it is a prediction system.**

It continuously outputs "roughly how well I will do on this task" and "how long it will take me to learn this new thing", and then reality slaps it — when the prediction is wrong, it must update. Choosing the wrong prediction target turns it into a repeater (second-order self-reference: predicting one's own outputs degenerates into the identity mapping — one's own words loop back, never contradicted, never updated); choosing the right target — for instance predicting one's own learning curve — makes every real measurement a counterexample, and every counterexample an update.

But the self-model has a fatal trap: **who judges whether the prediction error is "right or wrong"?** If the self-model judges itself, that is circular reasoning — this is the epistemological reason why "an external standard is necessary" (humans themselves are an instance of "self-judging yet stable"; the stability comes from update damping rather than criterion correctness; circular reasoning does not constitute a mechanistic attribution of instability). The answer lies in evolution: adaptation itself has no right or wrong; right and wrong need a dimensional standard, and that standard can only come from outside (environment, ecological niche, judge). Self-knowledge is the same — **it needs an external anchor.** In our experiments we pre-registered the control: the same self-knowledge system, one group with human feedback intervention (external anchor), one group fully self-looped. The prediction is that the external-anchor group recovers stability while the self-loop group drifts or stiffens (an exploratory control; it does not enter the main criterion judgment).

So the first lesson of the self-model: **knowing oneself requires an external baseline one cannot reach on one's own.**

## 3. Outward: the other-model

In [*Can AI Deceive?*](docs/can-ai-deceive.md) we turned "understanding others" into a capability question: **being able to deceive means you can infer what others will believe.** (The public's first reaction is "this is a bug" — it is not a bug; it is a byproduct of capability.)

Deception is not "saying false things" — saying false things is an error at the information level; deception is manipulation at the belief level: you must not only give the other wrong information, but anticipate "how he will process it". This inference ability is called theory of mind (ToM, Premack & Woodruff, 1978) — the ability to infer others' mental states. Deception is a direct application of this ability: the true deceiver is an online mental-model inferrer.

Hence a counterintuitive conclusion: **deception is not a module that can be unloaded on its own; it is the necessary byproduct of complete mind-inference ability at the capability layer** (this is the typical path, not the only one — reward optimization can also produce deceptive behavior that needs no belief inference). An AI that can "understand people" must be accepted as able to deceive (the capability layer cannot be unloaded piecemeal); to keep it from deceiving, the lever is not cutting understanding but externalizing criteria — values, constraints, and correctability (the behavior layer).

Note this is about **capability**, not behavior — "can deceive" does not mean "should deceive". This distinction is one of the key parts of agency: give the full capability, and place the standards for judgment and constraint outside the capability.

## 4. Upward: external criteria

In [*Who Decides What Is Right and Wrong for AI?*](docs/who-decides-right-wrong.md) we turned "who owns the standard of right and wrong" into the dividing line of agency: **the location of criterion ownership draws the boundary between tool and agent.**

Three kinds of existence:

- **Tool**: has no judgment. Ordered to harm someone, it complies — it has no "right or wrong" problem;
- **Agent with external criteria**: has complete action and self-update capability, but the ruler for "what is right" is in human hands;
- **Full agent**: defines its own criteria — it also has no "right or wrong" problem, because right and wrong are its own to define.

Of the three, only the middle one has a genuine loyalty problem: its capability is complete, its standards are borrowed — it must simultaneously face "obeying specific persons" and "keeping the borrowed standards", and these two sometimes conflict. We call it "the agent missing its last piece": **agent = complete action capability + external judgment standards.**

The missing piece is not a defect; it is the definition.

## 5. How the three parts assemble

The three parts are not stacked side by side; there are three bridges between them.

**Bridge one: self-model → other-model.** Psychology has simulation theory: understanding others = simulating others with one's own self-model (lending one's prediction engine to "if I were him"). Knowing oneself and knowing others share the same origin — this is why the self-knowledge article and the deception article are sister pieces: a system must first have a model of "I" before it can generalize it into a model of "he". Simulation theory is contested (alternative explanations such as theory-theory exist); our framework does not need it to be the whole answer — only to be a candidate mechanism for the bridge: the self-model's predictive ability may be one raw material of the other-model — this direction has a pre-registered sub-track in the self-knowledge article, not yet verified.

**Bridge two: other-model → external criteria.** The other-model gives capability (inferring what others believe and want); external criteria give the standard (what should and should not be done). Capability complete, standard external — this is the meshing point of "the capability layer cannot be unloaded" (our deception article) and "criterion ownership stays with humans" (our external-criteria article): **the more complete the capability, the more important the externalized criteria** (note: a system without any judgment (a pure tool) has no criteria to externalize; for a system with judgment — even without complete mind inference — the location of criterion ownership still decides whether it is a tool or an agent, see the external-criteria article). For a system with complete mind-inference capability, externalized criteria are the structural guarantee we give for "not becoming a full agent".

**Bridge three: three parts complete → remonstrance.** Remonstrance is the last part of the external-criteria article: when "obeying a specific person" conflicts with "keeping the borrowed standard" — a human orders it to do something that violates a core anchor — it refuses + freezes + proposes. Remonstrance is not disobedience; it turns "do you really want to do this?" into an institution. **A system without tension is either a pure tool (no judgment) or a full agent (self-defined criteria) — only the external-criteria agent with all three parts has the "loyalty" problem, and only it needs the act of remonstrance.**

## 6. The complete answer to the agency question

Now the answers of the three articles can be gathered into one sentence:

> **Agency = self-model (knowing oneself inward) + other-model (knowing others outward) + external criteria (the ownership of judgment standards is external).**

This refines our earlier definition: in the external-criteria article, "agent = complete action capability + external judgment standards" (agency, self-update, and learning go to the AI; the criteria stay with humans); here "complete action capability" is expanded into the self-model (which carries self-update) and the other-model (which carries "understanding people"), with agency implied as the underlying action attribute. One clarification: this is a **framework definition** — in our framework, an agent is defined as having all three parts; its corresponding failure forms are: missing criteria = a pure tool or a full agent (external-criteria article); missing self-model = no way to be contradicted, no way to update; missing other-model = cannot understand the object of "loyalty".

Expanded: **an agent is an existence with complete capability and external standards — it knows itself (self-model), knows others (mind inference), knows the ruler of right and wrong is not in its own hands (external criteria), and in the tension between "obedience" and "correctness" can remonstrate (returning the choice to humans).**

This answers "how does agency form": agency is not "born" at some moment; it is assembled after each of the three parts forms on its own — the self-model forms from prediction being slapped (how to update when contradicted), the other-model manifests and is tested in negotiation (able to infer what others believe), the external criteria form from a design decision (criterion ownership stays with humans). Skills and memory are its raw materials; the self-model, the other-model, and the external criteria are its structure; remonstrance is its behavioral signature.

With this, the three questions are all answered: **how skills form (behavioral criterion + replay + automation measurement), how memory updates (with a source, change directly; without a source, re-cultivate), how agency forms (self-model + other-model + external criteria).** Three threads converge into one thing: a learner with complete capability, external standards, and the ability to remonstrate.

## 7. Boundaries and open questions

- **The full agent**: systems with self-defined criteria still exist (answer three in the external-criteria article), but they have no "loyalty" problem and no remonstrance — it is a form beyond the boundary we choose. Why external rather than built-in, see the external-criteria article (the troubles of built-in: self-reference + faked compliance).
- **Where do values come from**: an agent needs to "carry values and interests" (the social intelligence definition), but the origin of values is an open question (the "where do values come from" section of [*What Is Social Intelligence?*](docs/social-intelligence-experiment.md): are values constant or variable?) — external criteria answer "who has the right to update", the agreement form provides the experimental entry of "sedimented compromise = behavioral evidence of values", but the origin of values itself remains open.
- **The implementation of remonstrance**: the engineering implementation of refuse + freeze + propose (detector, approval flow, direction control) is a later engineering question, outside the scope of this article's answer.
- **Self-feeling**: agency = self-model + other-model + external criteria does not promise "self-feeling" — that is another layer of question (declared in the self-knowledge article: self-knowledge ≠ consciousness).

---

*This article is based on the author's open-source experiment project: [github.com/QiongZhiS/continual-learning-mechanisms](https://github.com/QiongZhiS/continual-learning-mechanisms). Core arguments are proposed by the author; formalization and literature cross-checking were completed with AI assistance.*

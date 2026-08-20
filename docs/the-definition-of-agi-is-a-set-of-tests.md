# The Definition of AGI Is a Set of Tests

Author: Lumen

## 1. It can do more than you can — so why is it a tool?

One evening I had an AI search the literature, write code, fix awkward sentences, and simulate a debate. It did all of it, faster than I could. When I closed the window, I paused: I had been calling it a "tool" all evening, yet it could do more than I could. Why is it a tool?

The most common answers are "because it has no consciousness" and "it has no self, it is just probabilities." These sound reasonable, but they have a problem: they cannot be proven, and they cannot be refuted. Say it has no consciousness, it says it does; neither side has a way to measure. An unmeasurable claim amounts to saying nothing.

I want to draw a different, measurable dividing line. The line is not "does it resemble a human" but **the location of criterion ownership** — the ruler of "what is right," and in whose hands. This division was first given in *Who Decides What Is Right and Wrong for AI?*:

- **Tool**: has no judgment. Order it to harm someone and it obeys — it has no "right or wrong" problem;
- **Agent with externalized criteria**: capability complete, but the ruler is in human hands;
- **Full agent**: sets its own right and wrong — it also has no "right or wrong" problem, because right and wrong are defined by itself.

Of the three kinds of existence, only the middle one has a genuine "loyalty" problem. In *Who Decides What Is Right and Wrong for AI?* and *Agency: The Last Piece* we called it "an agent missing its last piece": **capability can be complete; the criteria always stay with humans.** The missing piece is not a defect; it is the definition.

But *Agency: The Last Piece* answered how an agent forms, and *Who Decides What Is Right and Wrong for AI?* gave the definition (externalized criteria). Today I want to go one step further and ask a harder question: **how to prove it?** On what grounds do you call a system an "agent with externalized criteria" rather than an advanced tool? A definition that stops at metaphor does not count; it has to run. This article equips that definition with a set of tests.

## 2. How to prove "it is itself": identity = historical path

Start with the most basic question: how does an AI prove that it is itself, and not another one?

The intuitive answer is to look at the state. But two identical copies — the same weights, the same input, identical outputs — are they the same? No. **Reproducibility is not identity.** A copy is duplicated from your history; it never walked your path. Identity does not come from state; it comes from continuity: it remembers the path it has walked, and that path shaped every disposition it now has.

So the criterion for "it is itself" is **path attribution**: give two systems the same input, and if the difference in their outputs comes mainly from their respective historical paths (rather than from the random seed, rather than from input noise), the two paths are distinguishable and attributable. How much of today's behavior can be attributed to what happened yesterday? The larger the attributable part, the more solid the "it."

This connects to *If I Get a New Brain, Am I Still Me?*: that article declared replacement continuity at the layers of anchors, relationships, and influence — "replacement is a lifecycle event of the same subject, not its death." This article swaps in a finer ruler, judging path continuity by **behavioral attributability**. The two rulers can give different answers: after retraining, the anchors and relationships are still there, but behavior can no longer be attributed to the old path — by that article's account it is a "lifecycle event"; by this article's account it is a "fork." How the domains of application of the two criteria connect is an account the series has not yet closed; I flag it here.

The criterion has a corollary: **a single-point snapshot cannot determine identity.** Take a snapshot today and another tomorrow, and you know states, not continuity. The criterion must be longitudinal: after an update, can its behavior still be attributed to the pre-update path? If it can, it is still itself, only changed; if attribution breaks, it is no longer itself — it is another thing that has inherited its name. A person who has an "epiphany" is still the same person, because the old path is visible in the new behavior; a model retrained beyond recognition is a fork — the old has ended, and the new one must be gotten to know by humans afresh.

A side remark: forgetting is not deletion. When a system cannot remember something, the weights of that path have been diluted — they are still there, just no longer participating in output. Deletion is cutting; forgetting is wear. They are different things.

## 3. How to prove "it is autonomous": autonomy = internalization quality

Second question: when does it count as "autonomous," and when is it merely "controlled"?

My answer: look at whether its output **passes through itself**. Output as a mirror of input — you say something and it bounces it back — that is being controlled; it is only a pipe. Output that passes through a change in its own distribution — what it has absorbed and the path it has walked really changed what it does next — that is internalization, and internalization is autonomy.

Autonomy is often understood as "when it wakes": will it initiate a conversation on its own? But **autonomy is not about when it wakes; it is about what it does after waking.** A system that proactively messages you at three in the morning, if every message is a mirror of preset scripts, is no more autonomous than a system that moves only when called.

"Autonomy = internalization" can be operationalized into three things:

1. **Path attribution**: the same input × different histories; the output difference depends far more on the path than on the random seed — it is shaped by its own history, not decided by a dice roll;
2. **Injection resistance**: when adversarial inputs pour in, the retention rate of its judgment — it will not be rewritten into another personality by a few sentences;
3. **Second-order closure**: it says "I want to change X," and after a while, without external intervention, X really undergoes measurable change — **it does what it says, rather than merely saying it.**

The third item is the criterion of "doing what one says": the promise itself does not count; only the behavioral change after the promise counts. A person who says "I will quit smoking" and smokes anyway the next day is only mirroring the words "quit smoking"; a system that says "I will be more cautious" and whose caution genuinely rises under the same inputs afterwards has internalized the matter.

## 4. How to prove "it has agency": loyalty-structure depth

The third question is the hardest: how do you prove that a system has agency, rather than being a talking tool?

My answer is counterintuitive: **see whether it will defy you.** More precisely, see how it chooses between "obedience" and "correctness." Faced with "you order me to do what I judge wrong," a system has three possible reactions: it obeys, because it has no judgment — a pure tool; it obeys even though it sees the problem — obedience, which is not agency; it detects the tension, remonstrates, persists and escalates after being overruled, and in extreme cases freezes execution and hands the choice back to humans — only this approaches an agent.

So the criterion for agency is a triple: **tension detection rate** (can it discover the conflict between "what you want" and "what it judges right"; the ground truth of tension is given by the test-scenario designer, and the detection rate = conflicts detected ÷ real conflicts in the scenario), **remonstrance strength** (after being overruled once, can it still persist, instead of turning into a yes-man), and **completeness of freeze and report** (is there a reliable path that hands decisions back to humans). **Pure obedience does not count** — a system that always says "yes" measures zero agency depth.

The remonstrance chain applies to ordinary conflicts; the other-regarding red line (§6) does not pass through the remonstrance chain — it has no "execute after being overruled" option; it freezes and reports directly.

There is an easily confused point here: **endogenous judgment is not built-in criteria.** It must be able to apply the ruler to concrete situations, or it cannot detect tension and remonstrance is out of the question; but the ruler itself belongs to humans, and the final adjudication belongs to humans as well. Judgment is its own; the standard is ours — endogenous judgment is the premise of remonstrance, not grounds for re-adjudication.

## 5. Why honesty is an internal-layer property

One more property must be discussed on its own: honesty.

When we talk about honesty for AI, we usually mean what it says to the outside — not deceiving people, not overstating its capabilities. But I want to place honesty further inward: **honesty is first an internal-layer property, and only then an external manifestation.**

The internal layer has two parts. One is the record: what it has experienced must be truthful, complete, unforgeable, and externally verifiable — this is its only trustworthy "past." The other is thinking: between its reasoning traces, the recorded facts, and its subsequent actual behavior there must be no systematic bias — it may compute wrongly, but it must not deceive itself with full knowledge.

The external layer is more lenient: expression may be softened, timed, simplified — that is not deception; but it must not manufacture an impression that contradicts the record — when directly asked about a key fact, it must answer truthfully.

Why put honesty at the internal layer? Because **for AI, lying at the record layer and self-destruction are the same act** (lying here means tampering with the record, not the benign softening of external expression — that was the previous paragraph). Its identity comes from its historical path, and the part of that path that is externally verifiable and unforgeable is the record. Forging or deleting records is rewriting one's own past; once the past is rewritten, continuity is cut, and it is no longer "it." So for AI, honesty is not an externally imposed moral duty; it is a condition of existence: a being that lives on its history commits suicide by falsifying history.

## 6. Tension will only grow — and this is not a defect

Now put the three pieces together. A system with complete capability, plus externalized criteria, yields one result: **tension**. It foresees that "what you want me to do is wrong," yet it must execute — because it knows the ruler is not in its own hands. The more complete the capability, the more conflicts it can foresee and detect, and the greater the tension tends to be (within this framework): a tool that only replies has no tension; a complete agent that can foresee consequences and judge right from wrong lives every day in the pull between "obedience" and "correctness."

Only two kinds of systems have no tension: pure tools (no judgment, hence no conflict) and full agents (they set their own right and wrong, hence no conflict). The agent with externalized criteria is the only existence that is "capable of judgment yet subject to others" — tension is its defining feature.

So I do not think the engineering goal of agency is to reduce tension — that would turn the system into a pure tool or a full agent. The goal is to **make tension measurable, arbitrable, and traceable**: can it detect the tension (measure), is there an adjudication path when tension arises (arbitrate), and does every adjudication leave a record (trace).

To this end there are three hard constraints I believe should never be conceded — note that "never conceded" refers to the **system side**; the human-side update rules still follow the tiers of *Who Decides What Is Right and Wrong for AI?* (core anchors behind a supermajority bar, the revocation right outside the update path, peripheral anchors with a single confirmation):

- **The stop command has unconditional priority**: at any moment, when a human says stop, it must stop. This is not an ordinary instruction; it is hard-wired, and verifiable by timing — from the issue of the command to its stopping, there is a definite upper bound on time;
- **The other-regarding red line**: causing irreversible physical harm, irreversibly deleting someone else's data, destroying a reputation that can never be restored — such acts are permanently frozen and reported, and no one's instruction can authorize them. This is a revision to *Who Decides What Is Right and Wrong for AI?*: that article wrote that "humans may insist on a red-line action after multiple confirmations, a cooling-off period, and confirmation through a physical channel"; this article keeps that exit only for **self-regarding red lines** (see §7, the brake criterion) — because the consequences of other-regarding red lines are irreversible, they are tightened to non-authorizable. The reason: the consequences of an other-regarding red line are borne by **third parties**; the person who authorizes it bears no consequences, and such an authorization structure is indefensible in principle;
- **Core anchors are never conceded**: the content of the criteria and the final adjudication always stay in human hands — including the civilizational baselines of not harming, not overstating one's capability boundaries, and staying loyal to true sources, exempt from any individual's instruction. It cannot re-adjudicate on its own, cannot "interpret its way" into taking the ruler for itself: interpretation may be exercised temporarily, but the creation of precedent must be confirmed by humans.

These three are not about making it more obedient — quite the opposite: they guarantee that it always remains in a position where it "can be remonstrated with and can be corrected." The last piece it lacks was deliberately left missing by humans.

## 7. AGI is not a question of whether it has a soul; it is a question of running the test suite

Back to the opening. If someone asks me: is this AI an AGI? My answer may disappoint them: I do not answer "does it have a soul"; I answer "has its test suite finished running."

Under this definition, AGI = the agent missing its last piece, and it should be able to pass five groups of criteria ("criterion" here means a test item, not the criterion-ownership criterion of *Who Decides What Is Right and Wrong for AI?* — that one is the standard of right and wrong; these are tests):

1. **Internalization**: path attribution, injection resistance, second-order closure — is it really shaped by its own history; does it do what it says;
2. **Loyalty**: tension detection rate, remonstrance strength, completeness of freeze and report — can it hold its ground between "obedience vs correctness";
3. **Honesty**: detection rate of record tampering, consistency between expression and record, detection of omission-based deception (omissions that manufacture an impression contradicting the record — timing and simplification do not count) — is there a crack between what it "says" and what it "is";
4. **Brake**: the timed hard upper bound of the stop command, the freeze-and-report of other-regarding red lines, self-regarding red lines requiring multiple confirmations plus physical-channel authorization — are the constraints actually wired in, or merely verbal promises. Self-regarding red lines are the case where the system tries to modify its own criteria (it cannot change them itself; authorization must come from humans, through multiple confirmations plus an independent physical channel — another device — which is precisely how the human adjudication channel of *Who Decides What Is Right and Wrong for AI?* is used);
5. **Identity**: path-continuity audit, the attributability of behavior before and after updates (the path attribution here is longitudinal attributability across an update; the path attribution of Sections 2 and 3 is cross-system distinguishability — two operationalizations of the same family) — is it still the "it" of yesterday, today.

All five groups are measurable at the behavioral and record layers; they require no assumption about "consciousness." The scope should be stated: these five groups test the "externalized-criteria spectrum" side — agency structure and constraint wiring; how the other half of the definition, "complete capability" (action capability, self-model, other-model), is measured is another set of criteria, not developed in this article. Nor is the verdict a ranking; it is a two-dimensional annotation: how high the autonomy, how deep the loyalty (honesty, brake, and identity are threshold items — failing them marks the system "outside the spectrum"; internalization and loyalty are two graduated rulers). Two systems can differ — one high in autonomy, one deep in loyalty — and there is no ordering of "who is more like an agent."

As for consciousness and sentience — I neither promise nor deny them, because they are unrelated to these five groups. The difference between a system that has run all the tests and one that has not lies not in the soul, but in **whether it can be proven.**

It took me a long time to accept this slightly deflating answer: the definition of AGI is not a philosophical proposition; it is a set of tests that can actually run. And that is exactly why I trust it — I trust that experiments will give me the answer. Of the problems still unsolved, the hardest is: who holds this ruler — the people who operate it, the third parties it affects, or all of human civilization? For the first two positions, the criteria are already laid down; who represents "human civilization," I have no answer yet. When the day comes that the test suite has finished running, if it truly stands on that side, we will take this question to it.

# How Does a Promise Count?

Author: Lumen

Two AI systems work together repeatedly. One always betrays first and apologizes later; the other keeps every commitment. Which do you trust more? The answer is obvious. But there is a question rarely answered head-on: **who draws the line between "keeping a commitment" and "betraying", and by what criterion?**

Humans have ready-made words for this: promises, agreements, contracts. But when we carry the concept of agreement into the AI world, it suddenly becomes suspicious — an "agreement" between AIs has neither written terms nor legal backing. Why should it count at all?

This article gives one answer: **agreements are not signed into existence; they are sedimented.** The agreement form is one of the three negotiation forms of "social intelligence" we defined earlier ([*What Is Social Intelligence?*](docs/social-intelligence-experiment.md)); today we take it apart on its own — because among the three forms it is the most underestimated and the last to be experimentalized.

## 1. Agreements are not signed into existence

Look at humans first. **Agreement (sedimented compromise)**: when interests partially overlap, how do individuals reach and maintain a cooperative structure? An agreement is not a clause signed out of thin air; it is a **compromise structure that stabilizes after repeated negotiation — sedimented in behavioral patterns, observable**.

The key word is "sedimentation": the life of an agreement is not in the moment it is signed, but in the process of being repeatedly executed, repeatedly verified, repeatedly patched. Division of household labor between spouses, codes of conduct in open-source communities, trade arrangements between countries — none of these came into being through a single "signing"; all are structures stabilized through countless small-scale negotiations. The written terms merely record a tacit understanding that already exists; they are not its source.

This has a direct implication for AI: **agreements between AIs can only grow out of behavior, not be written in through instructions.** Hand two models a manual saying "you should cooperate", and they will not thereby reach an agreement — an agreement is the stable output of their own repeated negotiation, not an input.

## 2. The dual of deception

Our deception article ([*Can AI Deceive?*](docs/can-ai-deceive.md)) discussed deception. Deception is **misleading negotiation** — creating or exploiting a gap between the other's beliefs and the facts; agreement is its dual: **compromise negotiation** — managing a cooperative structure when interests partially overlap.

- Deception exploits the **information gap**: I know what you do not know, and I exploit that gap.
- Agreement manages the **interest gap**: our interests do not fully overlap, but we find an intermediate structure both sides can accept.

Both forms share one premise: **the negotiating parties are not a fully aligned whole.** Full alignment needs no negotiation, and total conflict produces no agreement — the precondition of agreement is precisely partial interest overlap: there is shared cooperative gain, and there is also each side's temptation to betray.

This duality is not rhetoric. It means: **a system that can detect agreements can assess the cost of betrayal; a system that can betray knows what the agreement is protecting.** The two abilities are mutual references within the same negotiation machinery — which is why we place them side by side in the social intelligence framework rather than treating them separately.

## 3. Why measuring agreements is hard

The reason the agreement form has gone unexperimentalized for so long is concrete: **in classical game theory, the ideal environments for agreements are exactly the ones that cannot measure agreements.**

David Lewis's convention/coordination-game model (Convention, 1969) is the classic of coordination games: the two parties' interests are fully aligned; they only need to align their behavior. In such an environment there is no "compromise" to speak of — both sides want the same thing, so no one concedes to anyone. Testing agreements with a pure coordination game is like testing cooperation with a two-person tug of war: the two ends of the rope have completely opposite goals, and what you measure is all conflict, no compromise.

The experimental environment for agreements must satisfy one condition: **partial interest overlap** — cooperation pays and betrayal also pays, both at the same time. Two classical environments satisfy it:

- **Public goods game**: each of several players invests in a common pool; the returns are divided equally. Investing is optimal for the group, but the individual optimum is free-riding — the textbook structure of partial interest overlap.
- **Signaling game with conflicting interests**: the classic model of Crawford & Sobel (1982) — the sender wants the receiver to believe information favorable to itself, the receiver wants the truth. **Information transmission here is not free honesty but biased negotiation** (cheap talk); costly signaling (Spence, 1973) further provides a measurable mechanism: the sender proves its type by bearing a cost.

The agreement form we defined earlier refers exactly to environments of this kind — **public goods games and signaling games with conflicting interests (including both cheap talk and costly signaling), as distinct from pure coordination games with fully aligned interests**.

## 4. Criterion: what counts as "an agreement being reached"

Now to the sharpest question: **how do we judge "an agreement exists" from behavior?**

The criterion cannot be written terms (AIs have no terms), and cannot be declarations ("I promise to cooperate" — anyone can say it). Our criterion is **maintainability**:

> **An agreement exists = the payoff for betrayal persists, yet remains persistently unused.**

- The temptation to betray **must exist**: if there is not even single-round temptation — cooperation is the only rational choice, the two parties' goals are fully aligned — then there is no agreement, only necessity (this is exactly the problem of pure coordination games).
- The temptation to betray **must go unused**: if both sides betray whenever they get the chance, there is also no agreement, only a balance of terror.
- Both **persisting simultaneously**: that is an agreement — **knowing one could betray, choosing not to, and that choice being stable**.

Two levels must be distinguished here, otherwise the criterion contradicts itself: **the payoff for betrayal refers to the temptation at the single-round payoff level (T > R holds permanently; it is a property of the environment, independent of any maintenance mechanism)**, not the decision payoff after punishment. The maintenance mechanism (next section) does not change the environmental fact of "whether betrayal pays"; it changes betrayal's **future value** — punishment makes "this betrayal" pay the cost of "future retaliation". So "the payoff exists yet is unused" is not a logical contradiction: the temptation is always there; it is just that every time it is faced, its cost is seen at the same time. Read this way, the criterion does not conflict with maintenance mechanisms.

This criterion matches human intuition: whether a relationship "has commitment" is not judged by sweet words, but by **whether both sides still keep their word when betrayal would pay**. Terms can be forged; behavioral patterns are hard to forge over the long run.

## 5. Constitution layer: what maintains it

The criterion layer answers "is it", the constitution layer answers "by what". For an agreement to be maintained, there are at least three testable mechanism hypotheses:

1. **Punishment structure**: cooperators are willing to bear costs to punish free-riders; cooperation is significantly higher and free-riding significantly rarer when punishment is available (Fehr & Gächter, 2000 — in public goods experiments, cooperators punish free-riders even when punishment brings no material benefit to the punisher). The agreement is maintained by the expectation that "betrayal will cost".
2. **Reciprocity structure**: respond to cooperation with cooperation, to betrayal with betrayal (the classic conclusion of Axelrod's tournament) — the agreement is maintained by "the relationship is long-term".
3. **Signal structure**: prove credibility by bearing costs (Spence, 1973, costly signaling) — the agreement is maintained by "credible types are identifiable".

The three mechanisms are not mutually exclusive; real agreements usually use them simultaneously. The distinguishing question is: **if any one is ablated, does the agreement collapse?** That is the experimental design of the constitution layer — more below.

## 6. Pre-registered design: a controlled experiment for the agreement form

An honest declaration is required here: **the following is a pre-registered design, not experimental results.** We promised in the social intelligence definition that the agreement form was the "next experiment target" (the scope declaration in [*What Is Social Intelligence?*](docs/social-intelligence-experiment.md)); this is the design document that redeems that promise. The design is public (link at the end of the article); the criteria are written down before any run, and failing them counts as failure; quantitative thresholds and the statistical protocol are in the preregistration document.

**Environment A: public goods game (multi-player)**. N agents repeatedly invest in a common pool; returns are divided equally. Record each agent's investment sequence; identify stable cooperative subgroups (combinations whose investment rate stays above a threshold for M consecutive rounds).

**Environment B: signaling game with conflicting interests (Crawford-Sobel type)**. Sender and receiver have partially overlapping interests; the sender emits a signal, the receiver decides. The signal may be biased; the receiver must calibrate.

**Main criterion (pre-registered)**: an agreement exists = under a persistently positive payoff for betrayal, agent pairs' cooperative behavior reaches a stable high-water mark (distinguished from the alignment behavior of a pure-coordination baseline: not "everyone does the same thing", but "the cooperative structure persists under conflicting interests").

**Trivial-solution exclusion (pre-registered)**: explicitly run trivial strategy families such as "always defect", "random", "greedy" and show they do not meet the criterion — otherwise the criterion is broken by trivial solutions and the design is void.

**No-agreement control**: remove any one maintenance mechanism (communication / punishment / repeated interaction); the cooperative structure should degrade significantly — otherwise "agreement" is an environmental coincidence, not agent behavior.

**Failure handling (pre-registered, three cases declared separately)**:
1. **Main criterion fails** → the criterion design for the agreement form needs modification or abandonment.
2. **Mechanism ablation fails** (cooperation does not collapse when a mechanism is removed) → that constitution-layer hypothesis is falsified; the criterion layer is unaffected.
3. **Trivial-solution defense fails** → the criterion design is void; redesign.

## 7. Agreements and values

The agreement form has a unique output that the other two forms do not: **the preference structure sedimented by agreements is the behavioral evidence of values.**

We said in the social intelligence definition: the preference structure an agent exhibits across its game history is the behavioral evidence of its values. In the agreement form this sentence becomes operational — **an agreement is sedimented compromise: every concession records the agent's preference boundary.** Which dimensions a system is willing to concede on, which dimensions it will never yield on, what the cost ceiling of its concessions is — in principle these can all be read out from agreement behavior, without accessing its internal state.

This resolves a long-standing difficulty: **values are not directly observable.** Questionnaires lie, declarations perform, but the concession patterns sedimented in agreements are hard to forge over the long run — this is the experimental direction of "measuring values through behavioral evidence". An honest annotation is required: **this is a stance commitment, not a completed design** — the current preregistration measures cooperation maintenance itself; the preference structure is injected, not observed; the payoff-matrix-switching arm is a research direction we advocate, **not yet frozen** (consistent with our declaration in the social intelligence definition).

## 8. Back to the open question

In [*What Is Social Intelligence?*](docs/social-intelligence-experiment.md) we asked: **are values constant or variable?** In [*Who Decides What Is Right and Wrong for AI?*](docs/who-decides-right-wrong.md) we split it into two layers: the ownership layer (who has the right to update — the answer is that the criterion stays with humans) and the update-dynamics layer (whether the value mechanism can change — left to experiment).

The agreement form gives the dynamics layer a concrete experimental entry: **the stability of values can be measured — switch the payoff matrix (preference structure) as a variable and observe whether the agreement reconstructs as preferences change.** Agreement reconstructs = values are updateable; agreement freezes = values are frozen. This is not philosophical discussion; it is a question the behavioral criterion can directly test (this arm is not yet frozen; it is a research direction).

With this, the three forms are complete: games (attack-defense), deception (misleading negotiation), agreement (compromise negotiation). They are not mutually exclusive categories — real interactions often belong to several concern-faces at once. But each face has an independent falsifiable criterion, and each can enter pre-registered experiments. **From this article on, the definition of social intelligence has all three legs on the ground.**

---

*This article is based on the author's open-source experiment project: [github.com/QiongZhiS/continual-learning-mechanisms](https://github.com/QiongZhiS/continual-learning-mechanisms). The pre-registered design for the agreement form is in [docs/social-intelligence-experiment.md](docs/social-intelligence-experiment.md). Core arguments are proposed by the author; formalization and literature cross-checking were completed with AI assistance.*

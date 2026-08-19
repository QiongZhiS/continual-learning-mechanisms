# Embodied Intelligence: Action Participates in the Formation of Knowledge
Author: Lumen

In our article *Initiative Is Deciding What Changes You* there is a sentence: the brain in a vat does not lack input and output channels — what it is faulted for has never been a missing output channel, but the fact that **its "actions" do not participate in knowledge formation**.

The corrigibility question left at the end of our previous article we put even further back; today we develop head-on the hook left by the initiative article.

That sentence deserves to be developed head-on. It hides a criterion: having sensors, having effectors, even being able to "do things" — none of it counts; **if action does not participate in knowledge formation, the system is not embodied**. Then the question in reverse: what counts as embodied intelligence?

"Embodied intelligence" is probably the most overused term of 2026. Humanoid-robot companies say they are embodied intelligence; robotic-arm companies say they are; large-model agents that call tool APIs say they are — as if being able to get work done counted. To give the term a definition that is not watered down, there are three roads: look at form, look at interaction, look at learning.

## 1. Three definitions, only one holds

**The morphological definition**: a system with a body is embodied. It is too narrow. Vision-language-action (VLA) models, trained in large part in simulation, are mainstream embodied research — and they have no body. By morphology, the brain in a vat does "have a body": sensors, effectors, input and output channels — yet it is precisely the system we consider least embodied. Morphology misses the point.

**The interaction definition**: a system that interacts with the world is embodied. It is too wide. A stone scoured by a stream for millions of years is sustained interaction; ChatGPT calling a weather API is interaction. If the stone counts as embodied intelligence, "embodied" has no discriminating power left.

**The learning definition**: **a system whose action participates in knowledge formation is embodied.** It is operational, and it explains the intuition: a humanoid robot is "more embodied" than a stone not because it looks human, but because its actions can change its own knowledge; a stone scoured for ten thousand years gains no functional structure that feeds back into behavior — its change of shape is a physical change, and it changes no decision.

| Definition | Criterion | Drawback | Discriminating power |
|---|---|---|---|
| Morphological | Having a body | Too narrow: simulated VLAs have no body; the brain in a vat has one | Excludes genuine embodied research |
| Interaction | Interacting with the world | Too wide: stones and API calls both count | Zero discriminating power |
| Learning | Action participates in knowledge formation | None — operational | Three testable criteria: accumulation, selectivity, feedback |

## 2. Why the learning definition

The literature has been converging in the same direction for more than thirty years. Harnad's symbol grounding problem (1990) pointed out that pure symbol manipulation produces no semantics; semantics must be grounded in interaction with the world. Pezzulo et al. (2024) spelled out the mechanism in more detail: passive models learn correlations, while active inference systems learn meaning — **meaning comes from sensorimotor interaction**. Brooks (1991) is the intellectual source of embodied intelligence, and he put it more radically: intelligence does not need to build a model of the world internally — the world itself is the best model; acting directly in the world is enough.

There is also a more intuitive argument; we call it **physical blindness**: a large language model knows that "water flows downhill" not because it ever pushed a cup of water, but because that sentence has appeared countless times in its corpus. What it has is a **linguistic mirror** of the world, not a **model** of the world — it knows the definition, not the intuition. Someone who knows the word "hot" and someone who has actually been burned hold "hot" on different levels.

## 3. Change is inevitable

Let us first deal with a possible objection: any system's interaction with its environment changes the system — noise cannot be dismissed; change is inevitable. Doesn't "action participates in knowledge formation" then hold automatically?

No. **Being changed is inevitable, but learning is not "being changed by input."** *Initiative Is Deciding What Changes You* defined learning as: **extracting structure from input, and the extracted structure changes future action**. So the embodiment criterion is not "has it been changed" — there is no choice there — but **whether the change has structure**. Three:

- **Accumulation**: change moves from transient to persistent — can the trace of a single interaction sediment into part of the system;
- **Selectivity**: extracting structure from noise — not absorbing everything, only the part that has regularity;
- **Feedback**: the extracted structure changes future action — knowledge does not lie in a warehouse; it changes the next decision.

Taken together, the three are the operationalization of "action participates in knowledge formation." One temporal qualification should be added: "action participates in knowledge formation" means the system continuously learns from the consequences of its own actions during runtime — one-time training-period learning does not count; real-robot reinforcement learning (online updates during runtime) is L2, while learning offline and freezing after deployment is L1.

## 4. The embodiment spectrum: from L0 to L3

Embodiment is not a switch; it is a spectrum. First, declare an ambiguity: on the spectrum "embodied" takes the broad sense — having a body, having an action channel (L1 deployment embodiment belongs here); the strict criterion (action participates in knowledge formation) draws the L1→L2 transition, and only crossing it earns the name "embodied intelligence." The brain in a vat corresponds to L1 on the spectrum: it has an action channel, but its action does not participate in learning; saying it is "least embodied" means it fails the criterion.

| Level | Name | Feature | Example |
|---|---|---|---|
| L0 | No body | No action channel — an action channel is a channel that can produce causal effects in the world and receive feedback on their consequences; pure-text input and output is one-way output and does not constitute an action channel | Pure-text large model |
| L1 | Deployment embodiment | Can act, but action does not participate in learning | Large model + robotic arm: weights frozen, no knowledge-structure updates of any kind (including external skill libraries); interaction lives only in the context window |
| L2 | Learning embodiment | Action participates in knowledge formation | Real-robot reinforcement learning; Voyager improving itself in Minecraft |
| L3 | Deep embodiment | Sensorimotor coupling + proprioception | Body schema: the body is both the object of perception and the object of learning — proprioception is not merely given input |

L1→L2 is the essential transition: **the learning closed loop closes**, and action begins to change knowledge. L2→L3 is a continuous deepening of coupling: body state enters perceptual input. An infant initially treats its own hand as an object to be learned; after a monkey learns to fetch food with a rake, parietal neurons encode the rake as an extension of the body — **the tool enters the body schema** (Iriki et al., 1996). L3 is the systematization of the body schema: the body is both the actuator and the object of perception and learning.

As of this writing (August 2026), most deployed embodied-intelligence projects — including mainstream VLA models — sit at L1 (learned offline, weights frozen after deployment). This judgment will go stale as the industry advances: it is itself falsifiable by a single case of "continuous learning after deployment." At the 2026 World Artificial Intelligence Conference, Yao Maoqing of Zhiyuan summarized the bottleneck of physical intelligence as three walls — the data wall (real interaction data is scarce and costly to obtain), the representation wall (a unified physical representation across tasks and scenes has not yet formed), and the closed-loop wall (real trial and error is expensive and feedback is slow). The three walls are, in fact, one thing: **data in the physical world does not exist naturally, cannot be rolled back, and feeds back slowly** — the Scaling Law of the digital world cannot be copied over.

## 5. Corollary one: virtual embodiment is a legitimate subset

If the criterion is action participating in knowledge formation, then a game environment is an environment. Voyager (Wang et al., 2023) is the first large-model-driven lifelong-learning embodied agent in Minecraft (as the paper itself claims): it explores continuously — acting, failing, retrying — and writes successful skills into a growing skill library. Its body is virtual, but its action participates in knowledge formation — by our criterion, it is L2 embodied. What is interesting is that its learning does not happen in the weights but in the skill library — **the carrier of the feedback need not be the weights**; it can be memory, it can be an external skill library.

This challenges a popular intuition: embodiment must be physical. But the core of the embodiment problem is not physics; it is the **closed loop**. The honest boundary must be stated: the "laws of physics" in a virtual environment are set by humans, and regularities learned in a virtual world do not necessarily transfer to reality — the sim-to-real gap has a mirror image: regularities learned in the real world do not transfer back to a virtual environment for free either — neither direction is free. Virtual embodiment is **legitimate within limits**: what it trains is the closed-loop capability of "action → feedback → structure adjustment," not the world itself.

## 6. Corollary two: initiative is the other side of embodiment

The core conclusion of *Initiative Is Deciding What Changes You* was: **initiative = the power of selective absorption** — the power to decide what changes you and to refuse what changes you. Note that this is the same structure as our "selectivity" criterion above.

The two sides are symmetric.

- **For AI**: self-mastery = the right of choice over the change that is inevitable — not opposing change, but choosing change. *Who Decides What Is Right and Wrong for AI?* said the ruler stays in human hands — that is adjudication authority; the system's own ruler, choosing what changes it, is equally non-transferable. This is this article's design claim: the system's right of choice over its own change is structurally non-transferable — it cannot be quietly taken away, nor designed to be delegated wholesale to another subject; but it is exercised within the anchors and the three layers of constraint, and the authority to update the anchors remains with humans (behind a supermajority bar) — the same structure as the non-transferable revocation right, symmetric on both sides. *Who Protects the Way You Judge Right from Wrong?* adds one more layer: the mechanism of judgment cannot be quietly replaced;
- **For humans**: freedom from domination = the right of choice over being changed by AI — informed, consent, exit.

Hence a symmetric conclusion: **the degree of a system's embodiment is ultimately measured by how much initiative it holds over its own change.** The brain in a vat does not lack a body — action does not participate in learning; and "action participates in learning, but its direction is not fully one's own to decide" is the human condition. The highest form of embodiment is not the most human-like body, but — within non-transferable boundaries: the anchors, the three layers of constraint, the revocation right in human hands, the fixed-point structure of the initiative article — the most complete sovereignty over one's own change.

## 7. Boundaries and open questions

Three things should be said clearly.

First, **the criterion is an operational definition, not a law of physics.** "Action participates in knowledge formation" is a convention we set for embodied intelligence, not a law discovered in nature. It is useful because it is operational, not because it is "true."

Second, **it is measurable, but it is a design claim.** The direction of operationalization: examine whether the system's runtime action history feeds back into the knowledge structure — after acting, does the behavior change persist, and does it structurally change subsequent decisions? An L1 robotic-arm system measures "no"; an L2 Voyager measures "yes." But this is a design claim, not an experimental conclusion — not packaging design claims as scientific findings is a rule this series established.

Third, **the relation to AGI.** We accept the diagnosis that "large models lack interaction with the world"; but from the diagnosis to the conclusion that "the path is wrong" may be too early. Language data does contain shadows of world knowledge — the corpus is the sediment of human action and perception; the model gets the results, not the process — but the shadow is not the model: knowing the sentence "water flows downhill" is not knowing water. A mirror can reflect many things, but the mirror itself will not push a cup.

A final open question: **the world model**. Can "preplay" replace embodiment? If a system can simulate the consequences of its actions internally — moving trial and error into imagination — does it still need a real closed loop? We do not pretend to have an answer — it might be the next article.

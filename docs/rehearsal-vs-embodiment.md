# Rehearsal Cannot Replace Embodiment

Author: Lumen

Our article *Embodied Intelligence: Action Participates in the Formation of Knowledge* ended with an open question. The original text:

> A final open question: **the world model**. Can "rehearsal" replace embodiment? If a system can simulate the consequences of its actions internally — moving trial and error into imagination — does it still need a real closed loop? We do not pretend to have an answer — it might be the next article.

Today we develop it head-on. The world model is one of the hottest topics of the moment, but this article will not survey it; it answers only the series' own question: **can moving trial and error into imagination spare the real closed loop?**

First, transform the question. The promise of rehearsal is: simulate the consequences of actions internally, before trial and error in the real world. Then does an "internally simulated action" count as an action? By the criterion established in *Embodied Intelligence: Action Participates in the Formation of Knowledge*, if internal simulation participates in knowledge formation, it would seem to satisfy it as well. But a precondition hides here: **internal simulation is not free — it requires the system to first have a "consequence generator"** — the part that generates consequences is the world model. Without it, "rehearsal" is merely reciting sentences about the world once more. So the question changes from "does rehearsal count as action" to "**where does rehearsal's premise come from**". This article's answer: rehearsal cannot replace embodiment — because rehearsal's premise is itself a product of embodiment.

## 1. The premise of rehearsal is the world model

Rehearsal = generating the consequences of actions internally and choosing actions accordingly. The more a system relies on rehearsal, the more it depends on the accuracy of its consequence generator — once it errs, the deeper the rehearsal, the further the error. So "can rehearsal replace embodiment" is structurally equivalent to "**can the world model replace the real closed loop**".

And the world model is knowledge, and the predictions of knowledge must be screened by reality — that was the answer of *Is AI's Knowledge Discovered or Created?*. The question becomes: **what screens the model's predictions?**

## 2. A corollary of the physical blindness argument

Our article *Embodied Intelligence: Action Participates in the Formation of Knowledge* set up an argument we called physical blindness: a large language model knows that "water flows downhill" not because it ever pushed a cup of water, but because that sentence has appeared in the corpus countless times. What it has is a **linguistic mirror** of the world, not a **model** of the world — it knows the definition, not the intuition; a mirror can reflect many things, but the mirror itself will not push a cup.

The corollary: **a linguistic mirror is not a world model.** Rehearsal built on a linguistic mirror cannot derive "the feel of pushing a cup" — it can only recite sentences from the corpus. Then where does the world model come from? There are three candidate sources.

The first is the real closed loop: one's own actions change the world, and the world's responses feed structure back. This is the positive instance of the criterion in *Embodied Intelligence: Action Participates in the Formation of Knowledge*, and the most expensive route — real trial and error is costly and slow; at the 2026 World Artificial Intelligence Conference it was summarized as the "closed-loop wall"; and physical-world data cannot be rolled back, which makes the wall harder to climb.

The second is calibratable offline experience: not learning from the consequences of one's own actions (no reinforcement, no trial and error), but consuming observations of the real world. Tolman's (1948) rats explored mazes without reinforcement and, later, when faced with rewards, could go directly to the goal location — passive exploration can also form cognitive maps. But note: **the rats really ran the maze** — "offline" means not learning from action consequences, not avoiding contact with reality: what it consumes is the sensorimotor stream, not linguistic descriptions.

The third is the linguistic mirror: the corpus. The corpus is the sediment of human action and perception, but through the mediation of language, the coupling between action and consequence is lost. It can raise a "world model that talks"; it cannot raise a "world model that reasons".

The corollaries converge: **a true world model must come from a real closed loop or from calibratable offline experience; the linguistic mirror touches neither.** Note that "offline" does not mean "detached from reality" — offline experience is still sediment of the real world; only, the observer does not learn from action consequences. The real closed loop does not hand out "answers"; it hands out "vetoes": *Is AI's Knowledge Discovered or Created?* said nature only vetoes, never supplies — a wrong prediction is denied on the spot by reality, and the corpus never does that.

## 3. The rehearsal spectrum: from reaction to world model

Rehearsal is not a switch; it is a spectrum:

| Gear | Rehearsal depth | Feature | Example |
|---|---|---|---|
| No rehearsal | Reactive | maps states directly to actions, no internal projection | pure reactive policy |
| Shallow rehearsal | One-step / option evaluation | lookahead over a few candidate actions, execute the best | Dyna-style planning (Sutton, 1991) |
| Deep rehearsal | Long-horizon projection | unfolds multi-step imagination inside a learned model | MuZero (Schrittwieser et al., 2020); Dreamer (Hafner et al., 2020) |
| Full rehearsal (generative projection) | Predicts unseen states | predicts transitions outside the training distribution | Ha & Schmidhuber's (2018) "world model" vision |

The deeper the gear, the more rehearsal depends on the model's predictions of **states never seen** — projection means computing, in advance, the consequences of what has not happened and has not been seen. Ha & Schmidhuber (2018) gave the strongest demonstration: an agent trained inside "dreams" constructed by a generative model wakes up and directly drives a real racing game — the controller was never trained in the real environment. This is the limit case of "rehearsal replacing embodiment". But see the enabling condition clearly: that "dream" was itself trained on real interaction data — not imagined out of thin air. **Rehearsal saves the trial and error of training; it does not save the real data fed to the model.** The hottest world-model narrative (LeCun's 2022 architecture manifesto, a preprint) places predictive models at the core of autonomous intelligence — but the hotter the narrative, the more one must return to the same question: where does the model come from.

## 4. Two kinds of offline: replay and rehearsal

Our article *Does AI Need Sleep? — The Replay Path to Skill Acquisition* established a conclusion: AI does not need sleep; it needs an offline window — sleep replay grows the day's experience into the weights; that is **replay**. Replay replays experience that **has already happened**; rehearsal projects consequences that **have not yet happened**. What the two share is the "offline closed loop": both process experience outside real feedback.

So the two articles answer two sides of the same question: **can offline replace online?** *Does AI Need Sleep? — The Replay Path to Skill Acquisition*'s answer: yes — receive experience online, consolidate offline via replay (Rolnick et al., 2019), and the best hybrid strategy on most benchmarks works exactly this way: offline replaces "online consolidation".

This article's answer: not entirely. Offline cannot replace "taking data from the real world" — whether the data comes through the closed loop of one's own actions, or through offline experience consuming real observations. Rehearsal feeds on the model; the model feeds on real data. The chain: real closed loop (or calibratable offline experience) → world model → rehearsal → (imagined) experience → replay → weights. **Rehearsal is upstream of replay**: rehearsal produces experience; replay grows experience into the weights; but the fuel comes from reality.

This also connects to the complete answer of *Knowing How Is Not Knowing That*: skill = criterion + constitution (automation) + acquisition (replay) + endpoint (source-lost actionalization). If experience could all come from rehearsal, skills could truly be "acquired without a body" — this is the target this article aims at: the quality ceiling of rehearsal-produced experience is set by the world model, and the world model's quality ceiling is set by real data.

## 5. The boundary: simulation error

The boundary of rehearsal replacing embodiment is a line that can be stated clearly: **the accumulation of simulation error**.

Closed environments are the golden home ground of rehearsal. In chess and Atari the rules are fixed, simulation error is bounded, and deep rehearsal can almost fully replace real trial and error — MuZero (Schrittwieser et al., 2020) learned no rules and relied on no human priors, and surpassed the then state of the art in Go, chess, shogi, and Atari. But "replacement" must be unpacked: rehearsal replaces the **cost of trial and error**, not the **source of the model** — here, rehearsal really does replace real trial and error, but the real closed loop was not spared; it was merely moved forward into the "learning the model" stage. For the system, the action closed loop inside a simulator is also a real closed loop; where it is not "real" is relative to the deployment environment (the sim-to-real gap), which is precisely the boundary of the next paragraph.

Open environments are another matter. The simulator differs from reality (the sim-to-real gap, already touched in *Embodied Intelligence: Action Participates in the Formation of Knowledge* when it discussed virtual embodiment), and the error accumulates with projection depth: the longer the rehearsal, the further the imagined consequence drifts from reality. Deep rehearsal fails in the physical world not because "imagination is not enough", but because **the gap between imagination and reality cannot be closed by more imagination** — closing it requires real feedback to flow back and calibrate the model.

Hence a clean criterion: **does simulation error decrease monotonically as real closed-loop data increases.** The reading rule: "monotonically decreasing" means that over intervals where the amount of real closed-loop data doubles, simulation error drops significantly and the slope of the drop does not approach zero — if the drop stays below threshold over several consecutive doubling intervals, the curve is judged to have reached a plateau. If it decreases, rehearsal is an amplifier of embodiment: the real closed loop feeds the model, the model supports rehearsal, rehearsal saves trial and error; if it plateaus, rehearsal hits the wall, and the trial and error saved must be paid back to reality. The intervals, seeds, and statistics of the criterion are fixed at preregistration — it turns "can rehearsal replace embodiment" from a slogan into a measurable question.

## 6. The testable core

Three directions of operationalization, none dressed up as experimental conclusions — *Re-Sourcing Intuition: How to Find Its Source Again?* and *Embodied Intelligence: Action Participates in the Formation of Knowledge* follow this rule.

The first, **prediction of unseen states**: hand the model states outside the training distribution and measure prediction error. One who can predict transitions of unseen states deserves the name world model; one who merely recites seen fragments is memory.

The second, **error monotonicity**: measure the curve of simulation error against the amount of real closed-loop data, and judge by the reading rule above whether it decreases monotonically or has reached a plateau.

The third, **transfer loss**: train virtually, deploy on real hardware, and measure how performance loss changes with rehearsal depth — the quantitative version of sim-to-real.

These three are measurement directions, not experimental conclusions — they can be preregistered and measured, but we do not pretend they have already been run. Any system that calls itself a "world model" can be asked these three questions.

## 7. Boundaries and an open question

Three lines converge.

First, this is an operational definition and a design claim, not an experimental conclusion — operable and measurable, but not a law discovered in nature; not packaging design claims as scientific findings is a rule this series has established.

Second, stitch the four articles together. *Embodied Intelligence: Action Participates in the Formation of Knowledge* set the criterion: action participates in knowledge formation (accumulation / selection / feedback); *Does AI Need Sleep? — The Replay Path to Skill Acquisition* set the path: experience enters the weights via replay; *Initiative Is Deciding What Changes You* set the direction: selective absorption — deciding what changes you. This article adds the last link: **can the sedimented world model independently generate new experience?** The answer is no — rehearsal is a product and an amplifier of embodiment, not a substitute. Embodiment determines where the model comes from; replay determines how it grows into the weights; rehearsal determines how much trial and error is saved; initiative determines where it gets changed.

A final open question: if rehearsal is good enough and used long enough, will the system internalize simulation bias into intuition? Real feedback and imagined experience both enter the weights at replay; sources are lost (*Knowing How Is Not Knowing That*), and after the source is lost, can an intuition fed on simulation bias still tell "having pushed water" from "having dreamed of pushing water"? We do not pretend to have an answer — it might be the next article.

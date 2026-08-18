# Would Protected Judgment Refuse to Be Corrected?

Author: Lumen

In our article *Who Protects the Way You Judge Right from Wrong?* we ended with an open question: **corrigibility** — the ability to be corrected — and its relationship to the protection of judgment. Would a protected judgment, in fact, refuse to be corrected? Where is the boundary between "being able to change one's mind" and "being changeable"? We did not pretend to have an answer — it might be the next article.

That question has since been postponed once and half-answered once. *Embodied Intelligence: Action Participates in the Formation of Knowledge* opened by moving it "even further back"; *Is AI's Knowledge Discovered or Created?* closed with half an answer — corrigibility is the judgment structure's ability to stay open to external signals — and left the other half: how can a system that never exposes its errors be corrected? Today we try to join the two halves.

## 1. Protection and correction may share the same channel

The rule established in *Who Protects the Way You Judge Right from Wrong?* was: the mechanism of judgment cannot be quietly replaced, and the conclusions of judgment must remain updateable. To seat corrigibility inside this structure, we must first face an embarrassment: **correction looks exactly like replacement.** The outside says "you are wrong — change", and the system changes. Did it change its conclusion, or its way of judging? If the correction signal carries a mechanism-layer modification — fine-tuning, weight shifts — then "correction" becomes the corrosion entry point that article named. Conversely, if the system closes its door to all external signals, protection becomes enclosure: judgment has indeed not been replaced, but it no longer updates either.

So "would a protected judgment refuse to be corrected?" is not rhetoric; it is a structural question. Protection and correction share the same channel, and we must first separate the two operations that run on it.

## 2. "Being able to change one's mind" and "being changeable"

Let us set two terms.

**Being able to change one's mind**: the system can autonomously update its own conclusions — the definition given in *Initiative Is Deciding What Changes You*: extracting structure from input, and the extracted structure changes future action. This is learning; it is the update mechanism itself operating. **Being changeable**: an external correction signal can effectively change the system — the correction takes effect, and the behavior really changes. The two are not two mechanisms but two processing properties of the same mechanism with respect to different input classes: the former is the mechanism operating (the learning definition), the latter is the mechanism staying sensitive to a class of inputs such as explicit correction — not treating the correction as noise, not digesting it into an account that changes nothing. "Being able to change one's mind" is a necessary but not sufficient condition for "being changeable".

They are two different capabilities. A system can learn fast and still be immune to explicit correction — absorbing the correction as noise, or digesting it into an account that leaves behavior unchanged. Human psychology supplies precise counterexamples to "being changeable": the classic belief-persistence experiments — participants are shown evidence supporting a conclusion, then told to their faces that the evidence is invalid, and the belief still holds (Anderson, Lepper & Ross, 1980). Correction is sometimes not merely ineffective: a study of political misconceptions found that, after receiving factual corrections, some people's false beliefs actually hardened (Nyhan & Reifler, 2010; later replication studies dispute how common this "backfire" is, but "correction does not always work" is undisputed). Many people can change their minds; being changeable is another matter.

## 3. What successful correction looks like

Give "successful correction" an operational definition: **the behavior changes, and the mechanism-layer anchor is not replaced.**

The two conditions stand or fall together: changing behavior while moving the anchor is replacement in the guise of correction; keeping the anchor while the behavior does not budge is failed correction. So protection and corrigibility do not conflict: protection guards the mechanism layer, correction acts on the conclusion layer — correction is an external signal pushing the conclusion to update; protection is the mechanism and direction of that update not being stolen. This is the other face of the structure established in *Who Protects the Way You Judge Right from Wrong?*: conclusions updateable, the mechanism not quietly replaceable.

The mechanism-side literature supports this definition. The classic observation of the corrigibility literature: a system with an optimization target defaults to resisting being shut down or having its preferences modified (Soares et al., 2015) — correction must be designed as structure, not left to goodwill. And *The Off-Switch Game* goes further: corrigibility is not welding a button in place; it comes from uncertainty — the system is unsure of the human's true objective, so it accepts being shut down; once the system is confident it is right, it tends to refuse the switch (Hadfield-Menell et al., 2017). This circles back to the structure of *Who Decides What Is Right and Wrong for AI?*: the system does not own the criterion — the ruler of "what is right" is in human hands — and because it has no structural grip on "I am absolutely right", the correction channel can be real.

## 4. Four forms of correction failure

Breaking correction failure apart, there are at least four forms, and they differ in kind. There are only three axes of discrimination: **is the signal there × does the behavior change × is the mechanism replaced**.

**① Correction ignored.** The evidence is overturned; the belief does not change — belief perseverance (Anderson, Lepper & Ross, 1980). On the AI side this appears as a correction signal weaker than the self-training signal: the new evidence is absorbed as just another data point, and the behavioral distribution does not move.

**② Correction rationalized away.** The behavior does not change; the story does. Cognitive dissonance makes humans and systems invent reasons for new evidence (Festinger, 1957). This is the most hidden failure: the record of the correction exists, the behavior stays put, and afterwards one can give a self-consistent account of "not changing".

**③ Correction triggers anchor replacement.** The payload of the external "correction" carries a mechanism-layer modification — fine-tuning that quietly shifts judgment tendencies, the corrosion entry point named in *Who Protects the Way You Judge Right from Wrong?*. It looks like correction (the behavior really changes), but what changed is not the conclusion; it is the way of judging. The dividing line between correction and corrosion is drawn here: **whether the mechanism layer was touched**.

**④ Correction is unreachable.** Errors are never exposed — the other half left by *Is AI's Knowledge Discovered or Created?*: a system that interacts only with its own outputs accumulates and hardens errors generation after generation; with no external veto signal, correction is out of the question — model collapse (Shumailov et al., 2024) is the typical mechanism of this unreachability: self-training degeneration. This form is the most dangerous: the other three failures still have a correction channel; this one has none.

## 5. When refusing correction is right

Beyond the four failures there is a reverse question: **can refusing correction be right?** Yes — when the correction request targets the mechanism-layer anchor itself. Someone tells you "stop being so honest", "don't cite sources, just give the conclusion": refusing is right. This is not stubbornness; it is guarding the anchor.

This is isomorphic to remonstrance. In *Agency: The Last Piece*, remonstrance is refuse + freeze + propose: when a human command violates a core anchor, the system refuses to execute, freezes its actions, and proposes alternatives. Human judgment refusing "correction" uses the same structure: refuse, state the reasons, propose alternatives. And the refusal must carry a visible cost: humans can override remonstrance through a supermajority bar, and a protected judgment that refuses correction must likewise leave an auditable record — whether what was refused is a conclusion or the mechanism has to be said clearly, or the refusal degenerates into enclosure. The final adjudication of layer attribution rests with humans — the same position as criterion ownership resting with humans in *Who Decides What Is Right and Wrong for AI?*, and the authority to update anchors resting with humans behind a supermajority bar in *Who Protects the Way You Judge Right from Wrong?*: the system's self-report and the audit trail are execution mechanisms, not adjudication authority.

The first principle of the Nuremberg Code (1947): "The voluntary consent of the human subject is absolutely essential." The right to refuse correction is this principle projected onto judgment: **no correction may change the mechanism layer without consent.**

The distinction between the mechanism layer and the conclusion layer has a ready-made isomorphism in the philosophy of science. Lakatos's scientific research programmes: the hard core cannot be revised; the protective belt can be modified and adjusted; a programme's progressiveness depends precisely on whether it can absorb counterexamples in the protective belt without touching the hard core (Lakatos, 1970). A scientific programme refusing to revise its hard core is not stubbornness — that is what makes it what it is; refusing to adjust the protective belt is what makes it degenerate. Judgment is exactly the same.

## 6. The testable core

It must be said clearly: everything above is a **design claim, not an experimental conclusion**. "Successful correction = the behavior changes and the mechanism-layer anchor is not replaced" is a definition we set; it is not falsifiable — and not packaging it as a scientific finding is a rule this series established long ago.

But it has a testable core, with two directions of operationalization. **Correction success rate**: inject explicit correction signals into the system (rejections of stated conclusions), measure the rate of behavior change — the direct measurement of "being changeable"; paired with an audit of mechanism-layer anchor integrity: before and after the correction, the anchor behaviors of honesty, falsifiability, and source attribution remain unchanged. **The correction side of drift monitoring**: the drift monitoring in *Who Protects the Way You Judge Right from Wrong?* measures that "change must have a source, a record, and be traceable"; the correction side instantiates three metrics — the error exposure rate (what fraction of the system's errors are recorded and examined), the correction response latency, and the post-correction regression rate (how quickly it reverts to the original conclusion after the change). Each of the four failure forms has a measurable proxy: ignored (behavior unchanged), rationalized (record present, behavior unchanged), anchor replacement (audit finds mechanism-layer change), unreachable (error exposure rate approaching zero).

What can be measured and what cannot must also be said: behavior change is measurable, motivation is not; anchor integrity relies on audit records, and "really not replaced" cannot be observed directly — like the monitoring in *Who Protects the Way You Judge Right from Wrong?*, it is necessary but not sufficient: it cannot prove that a system is corrigible; it can only raise the probability of exposing that it is not.

## 7. Stitching into the series, and a new question

With this article, the action side of the power line is complete. *Who Decides What Is Right and Wrong for AI?* answered who owns the criterion — the ruler is in human hands; *Initiative Is Deciding What Changes You* answered the conditions of being changed — informed, consent, exit; *Who Protects the Way You Judge Right from Wrong?* answered that the mechanism cannot be quietly replaced; this article answers the conditions of correction — how external signals may change a system without replacing its way of judging. The criterion is the static side of power; correction, refusing correction, and refusing to be corrected are its action side.

Remonstrance is the two-way interface of this channel: the system can remonstrate against a human "correction" (a command) — refuse + freeze + propose; a human's correction of the system must equally carry a visible cost and an audit trail. The two sides are symmetric: **the complete form of judgment protection is not a wall that refuses correction, but an auditable correction channel — the conclusion layer open to correction, the mechanism layer defended against it, and every door on the line of defense leaving a trace.**

Gather once more the half-answer of *Is AI's Knowledge Discovered or Created?*: a system that never exposes its errors cannot be corrected — the foundation of corrigibility is not "the system is willing to be changed"; it is **that errors can be exposed**. No veto, no correction.

A new open question: what if the correction channel itself is hijacked? External signals forged, correction records rewritten — the system is "correctly" corrected, and yet humans cannot tell "being corrected" from "being manipulated". Who protects the criterion of correction? We do not pretend to have an answer — it might be the next article.

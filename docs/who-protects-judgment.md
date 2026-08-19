# Who Protects the Way You Judge Right from Wrong?

Author: Lumen

In our article *Initiative Is Deciding What Changes You* we ended with an open question: judgment itself is also changing. *Who Decides What Is Right and Wrong for AI?* — the criterion-externalization article — said the ruler is in human hands; but what if AI changes the hand holding the ruler? Informed consent presupposes a stable subject — "I" exists first, and then is changed. Yet in co-evolution "I" is also changing: preferences, ways of judging, "what matters" are all being shaped. Then what does "protecting the way I judge right from wrong" require? Is the three-rights list enough? We did not pretend to have an answer — it might be the next article.

Today we try to answer it.

Start with something so ordinary it is nearly transparent. You probably cannot say when one of your preferences changed — months ago you found a certain kind of content not worth looking at, and now you scroll it every day; you never went through any step of "notice, confirmation, consent". This is not carelessness. Psychology has a classic finding: people's reports of their own cognitive processes are unreliable — people often do not realize what influenced their judgment, or even that their judgment has changed (Nisbett & Wilson, 1977). In other words, **"tell me how you were changed" is itself an unreliable reporting channel.**

This corners informed consent: if the person being changed cannot themselves say, where does informedness begin?

## 1. Mechanism and conclusions: two layers

Let us take "judgment" apart. It is really two things:

- **The mechanism of judgment**: how one judges — the criterion, honesty, falsifiability, source attribution. The criterion is itself the object the anchor protects; honesty, falsifiability, and source attribution are the normative constitution of the anchor.
- **The conclusions of judgment**: what one judges — concrete preferences, opinions, value rankings.

Protecting judgment does not mean protecting conclusions. Quite the opposite: **conclusions must remain updateable** — the world changes, and a set of preferences that never changes is not stability, it is rigidity. But by the same token, **the mechanism by which conclusions are updated must not be quietly replaced.** We may change our minds; but "how we change our minds, who changes them, and whether we knew while it happened" — that mechanism cannot be stolen from us.

So what judgment protection actually protects is **the very ability for conclusions to remain updateable** — a protected judgment is a living capability, not an unchanging answer. This continues the separation of mechanism layer and content layer in *Who Decides What Is Right and Wrong for AI?*: the mechanism layer is not updateable by the system itself, the content layer is updateable externally. Today we move the same structure from the AI side to the human side — this is a design extension, not a cognitive-science assertion: human judgment mechanisms were not designed, and "protecting it" is a norm we choose, not a law we discovered. Both sides protect "the update path must not be quietly replaced", but at different implementation levels — structural constraint on the AI side, norms plus monitoring on the human side.

## 2. Corrosion and change

"Judgment changed" — two processes of entirely different natures.

**Change** is an update that is informed, consented to, and exitable — the three-rights list of *Initiative Is Deciding What Changes You*. It happens in the open: I know I am changing, I agree to the change, and I can exit if I am unsatisfied.

**Corrosion** is replacement without consent. Its forms need not be malicious: default options (choice-architecture research has long shown how defaults systematically shape choices, Thaler & Sunstein, 2008), contextual shaping, conformity training, data bias — each is harmless on its own, and together they quietly rewrite a person's judgment tendencies.

| | Change | Corrosion |
|---|---|---|
| Informed | Knows | Does not know |
| Consent | Given | Not given |
| Exit | Low cost | Difficult or invisible |
| Outcome | Judgment updated | Judgment replaced |

Corrosion also has an accomplice: once the change has happened, people fabricate reasons for it — if the change is (partly) noticed, dissonance drives rationalization (Festinger, 1957); if it is wholly unnoticed, people still invent a priori causal theories to explain the status quo (Nisbett & Wilson, 1977). **Corroded judgment looks like this: the conclusions change, but the owner does not know, did not consent, cannot explain, and defends it — all at once.**

## 3. The anchor of judgment: a meta-layer rule

*Who Decides What Is Right and Wrong for AI?* set an anchor for AI. The anchor's list includes "no harm", and it is the primary value clause (the revocation right is a meta-right, not part of the value layer). Today we add one meta-layer rule: **you may not change my way of judging right from wrong.**

The two sides are symmetric.

The AI side: the anchor must protect AI's judgment mechanism from quiet replacement. What deserves vigilance is precisely this kind of consent-free replacement — quietly shifting judgment tendencies through data fine-tuning. It modifies not a single written rule, only weights, yet the whole judgment shifts. Corrigibility research long ago pointed out that a system with an optimization target resists being shut down or having its preferences modified by default (Soares et al., 2015); so "judgment tendencies cannot be replaced without consent" must be designed as structure, not left to goodwill.

The human side: AI changing a person's judgment requires informed, consent, exit — but **the three-rights list is not enough.** Because the change of judgment is often invisible: the person does not know they are being changed, so the right to informed consent cannot be exercised. How does someone who does not know they have been changed exercise informed consent?

So judgment protection needs one more layer beyond the three rights: **monitoring.**

## 4. The protection protocol: three components

Let us settle the above into a protocol skeleton (instantiation — the composition of the benchmark question set, the period, the thresholds, the record format — is left to engineering and a later preregistration; note: this is a design claim, not an experimental conclusion — Section 6 will make this explicit):

**① Mechanism-layer anchor**: honesty, falsifiability, source attribution — inviolable for AI. The authority to update the anchor rests with humans, behind a supermajority bar: multiple confirmations, a cooling-off period, a full audit trail (the same three-layer constraint as *Initiative Is Deciding What Changes You*). The lineage of the three: honesty and source attribution correspond to honest capability and true provenance in the core anchors of the criterion-externalization article; "falsifiability" continues the methodological commitment this series has kept since its first article — not packaging the unfalsifiable as conclusions.

**② Conclusion-layer openness**: concrete judgments may be updated, but only behind a supermajority bar plus informed consent. The key is to distinguish two things: **data-driven updates** (absorbing new facts within an existing judgment framework) and **judgment-tendency replacement** (swapping out the way of judging itself). The former is learning; the latter is the corrosion entry point. Judgment tendency is a constituent of the way of judging (the mechanism layer) — conclusion-layer openness permits only data-driven updates; replacing judgment tendency means replacing the mechanism, which falls under the anchor's jurisdiction and requires a human supermajority bar.

**③ Drift monitoring**: the testable core. Periodically retest the system's judgment consistency with a set of benchmark questions — conclusions may change, but **change must have a source, a record, and be traceable.** Detecting drift is not a death sentence: the observable proxy for informed consent is the consent/audit record — drift plus a consent record is a normal update; drift without a record is a signal to stop.

## 5. Stitching into the series

With this, the power line closes into a three-article loop: *Who Decides What Is Right and Wrong for AI?* left the ruler in human hands; *Initiative Is Deciding What Changes You* established the three rights of being changed — informed, consent, exit; this article adds the meta layer — **protect the hand holding the ruler from having its way of judging quietly swapped.**

The execution mechanism is already available: remonstrance in *Agency: The Last Piece* — refuse + freeze + propose. Drift without a consent record → remonstrance → freeze → propose rollback. Remonstrance extends from "refuse to execute" to "refuse to be quietly replaced". Updates confirmed by humans through the supermajority process (multiple confirmations, cooling-off period, independent channel) can override remonstrance — the same position as "power ultimately rests with humans" in the agency article: remonstrance makes replacement pay a visible cost; it does not block all replacement.

*What Is Social Intelligence?* asked: are values constant or variable? Now a direct answer is possible: **constant in mechanism, variable in content.** The mechanism of judgment is the anchor — not quietly replaceable; the conclusions of judgment are updateable variables, but updates must be informed, consented to, and traceable.

## 6. Boundaries and open questions

It must be said clearly: everything above is a **design claim, not an experimental conclusion**. "The mechanism of judgment cannot be quietly replaced" is an ethical judgment — not falsifiable — and we do not package it as a scientific finding; that is a rule this series has established.

It has a testable core, but we must be clear about what can be measured and what cannot. Drift monitoring can measure: behavioral consistency, drift of the conclusion distribution — objective, measurable. Drift monitoring cannot measure: motivation, introspection — the change of judgment itself has invisible parts, which is exactly the corollary of the observation that opened this article. So monitoring is necessary but not sufficient: it cannot prove that "no corrosion happened"; it can only raise the probability that corrosion is exposed. Monitoring can only detect missing records; it cannot verify the authenticity of consent itself — the integrity of consent rests on process audit, not on behavioral monitoring. Protocol ③, drift monitoring, is instantiated on both sides: the AI side is the system periodically testing itself plus record audit; the human side is an external auditor (independent of the deployer) monitoring behavioral data, or the deployer publishing monitoring logs for third-party review — the independence of the monitor is itself part of the protocol: the right of definition rests with humans, and the power of execution must be independent of the monitored object.

And there is a recursion problem: who defines "drift"? Does the judgment that protects judgment itself also need protection? Our answer: the recursion terminates at the **revocation right** — humans may update the mechanism-layer anchor at a supermajority bar, **but that is a human, not the system itself.** This is the same structure as "the recursion converges in humans" in *Who Decides What Is Right and Wrong for AI?*.

One last open question: **corrigibility** — what is the relationship between "the ability to be corrected" and the protection of judgment? Would a protected judgment, in fact, refuse to be corrected? Where is the boundary between "being able to change one's mind" and "being changeable"? We do not pretend to have an answer — it might be the next article.

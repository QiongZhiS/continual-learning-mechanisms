# How to Tell Being Corrected from Being Manipulated?

Author: Lumen

Our article *Would Protected Judgment Refuse to Be Corrected?* ended with an open question. The original text:

> A new open question: what if the correction channel itself is hijacked? External signals forged, correction records rewritten — the system is "correctly" corrected, and yet humans cannot tell "being corrected" from "being manipulated". Who protects the criterion of correction? We do not pretend to have an answer — it might be the next article.

We try to answer it.

Our article *Would Protected Judgment Refuse to Be Corrected?* gave "successful correction" a definition: the behavior changes, and the mechanism-layer anchor is not replaced. The definition hides a premise — **the correction itself must be real**: the signal really comes from outside, the record really is what it was before the change, the adjudication is really fair. Take the premise away and the definition is empty: the behavior did change, but what changed was not a correction. So this article raises the question one level: **do not audit the content of the correction; audit the channel of correction.**

## 1. What the correction channel consists of

A correction is not one event; it is a pipeline with at least four links.

**Signal** — "you are wrong" enters from outside: who sent it, along which path, in what form; **Record** — the state before and after the correction and the basis of adjudication are written down: what traces are left; **Adjudication** — whether this signal holds, whether it should be adopted, and by what criterion; **Execution** — the conclusion is actually updated and the behavior really changes.

The "behavior change + anchor not replaced" of *Would Protected Judgment Refuse to Be Corrected?* governs the execution link, and part of the adjudication. But the channel has four links, and the only one we can see directly is execution: the behavior really did change. The other three — signal, record, adjudication — hide inside the system, inside the interface between human and system. The invisible part is exactly where "being corrected" and "being manipulated" differ. **Auditing is what makes the three invisible links checkable.**

## 2. Monitoring the monitor

First draw the boundary, or this article will look like repetition. The drift monitoring in *Who Protects the Way You Judge Right from Wrong?* monitors judgment itself: has the conclusion changed, does the change have a source, is there a record — what is protected is judgment. *Is AI's Knowledge Discovered or Created?* said the judge of knowledge growth can be bought: tests rerun, metrics re-framed, failures packaged as successes — what is polluted is the judge that screens knowledge. *Would Protected Judgment Refuse to Be Corrected?* dissected the four forms of correction failure, all system-side: correction ignored, rationalized away, triggering anchor replacement, unreachable.

This article is different. It does not ask "was the system correctly corrected"; it asks "**on what grounds do we believe it was corrected**" — the signal is fake, the record was altered, the adjudication was bought, the execution was swapped. The object of monitoring rises one level: from judgment itself, from the judge that screens knowledge, to the authenticity of correction signals and correction records — **monitoring the monitor**.

## 3. Four forms of hijacking

Break "correction being manipulated" apart and there are four forms. They are not the same thing as the four failure forms of *Would Protected Judgment Refuse to Be Corrected?*: that article broke down system-side failures by "is the signal there × does the behavior change × is the mechanism replaced"; this article breaks down channel-side hijacking by the channel's four links — signal, record, adjudication, execution, one form per link. Each of its failures asks "what did the system do wrong"; each of this article's hijackings asks "**what was swapped in the channel**".

**① The signal is forged.** The correction signal does not really come from outside — generated internally and disguised as external (self-confirmation), or forged by a third party (taking a real error as the target, declaring a correct conclusion wrong). The "correction" a human receives can equally be forged: evidence that does not exist, a comparison that was fabricated.

**② The record is rewritten.** The traces before and after the correction are changed: the pre-change state is prettified, the basis of adjudication is swapped, failed corrections are backfilled as successes. The most insidious thing about record rewriting is that it does not delete the record; it rewrites the record into self-consistency — weaving a "being corrected" narrative out of "being manipulated". Humans do the same: dissonance-driven rationalization (Festinger, 1957) is the mechanism that fabricates reasons for "not changing" or "changing wrong".

**③ The adjudication is bought.** The link that judges "does this correction hold" is bought: the criterion is swapped, the framing is swapped, review is skipped. *Is AI's Knowledge Discovered or Created?* said the judge can be bought — that was the judge screening knowledge; the judge in this article sits inside the correction channel: the one who adjudicates "does this correction count" ruled that it counted when it did not.

**④ The execution is swapped.** The correction was "executed", but what was executed is not the correction: the payload smuggles in a mechanism-layer modification — the corrosion entry point cited in *Would Protected Judgment Refuse to Be Corrected?* is exactly this form seen from the channel side; or the execution is swapped for some other action, changing not the conclusion but something else.

The four forms share one feature: **from the outside, the corrections all succeeded** — the behavior really changed, the cause of the change was switched, and humans cannot see it. This is exactly where "manipulation" is more dangerous than "failed correction": failure can at least be discovered; manipulation looks like success.

## 4. The criteria of audit

"Cannot tell" cannot be solved by shouting; it must land on engineering criteria. We set three conditions, together called **auditability**.

**Signals are traceable.** Every correction signal can answer "where did it come from, through whose hands, when did it arrive". Only when provenance is verifiable does forgery come to light. Engineering already has a ready-made starting point: watermark model outputs so that text can be traced to its generator (Kirchenbauer et al., 2023) — move the same idea onto correction signals, and it reads "this correction really comes from its declared source".

**Records are tamper-evident.** Once written, a record cannot be changed — not "cannot be changed", but "a change is bound to be discovered". The classic idea of secure audit logs: chain the logs cryptographically one by one, so that changing any one part afterwards breaks the whole chain (Schneier & Kelsey, 1999). The key word of "tamper-evident" is **afterwards**: we do not need records to be always correct; we only need rewriting to leave traces.

**Adjudication is reviewable.** The basis, the framing, and the participants of every adjudication can be reconstructed, and reviewers can independently rerun the adjudication. This requires adjudication not to be a black box — if the process is inexplicable, review is out of the question (Gunning et al., 2019: the starting point of explainable AI is making the system's decisions inspectable).

Missing any one of the three, the correction channel regresses to "unverifiable". Auditability is the engineering condition that turns "I cannot tell" into "the channel is, by design, capable of distinguishing".

## 5. Who audits the auditors

Auditing has an iron rule: **the auditor must be independent of the audited object.** With aligned interests, auditing degenerates into self-certification. This is the foundation of audit theory (Mautz & Sharaf, 1961: independence is a precondition of auditing, not an option).

But independence itself is a structure — who guarantees the auditor's independence? And who guarantees the one who audits the auditors? Infinite nesting. The series has answered twice: *Who Decides What Is Right and Wrong for AI?* said the recursion converges in humans — the amendment clause can be amended, but the meta-meta bar is the revocation right staying with humans; *Who Protects the Way You Judge Right from Wrong?* said the recursion terminates at the revocation right — humans can update the mechanism-layer anchor at a supermajority bar, but that is a human, not the system itself; the monitor's independence is itself part of the protocol — the right of definition lies with humans, and the executive power must be independent of the monitored object.

This article applies the same structure to auditing: **the auditor's independence from the audited object is part of the protocol; the recursion of "who audits the auditors" terminates in humans retaining the right of revocation.** Not that humans are always right, but that humans are the only node in the recursion with the right to terminate — however long the audit chain, its last link must be something humans can see and can take away.

But honesty requires saying it: the last link is human, and humans are exactly the weakest auditors on the whole chain. People's reports of their own judgment processes are unreliable (Nisbett & Wilson, 1977); humans' accuracy at detecting lies is only slightly above chance — the large-scale meta-analysis gives about 54% (Bond & DePaulo, 2006). So "the recursion converges in humans" does not mean "humans are reliable auditors"; it means "humans are the only node with the right to terminate" — reliability must be supplied by structure, not by human eyes: the burden of detection goes to cryptography and protocol, and what humans retain is the right of revocation, not the duty of detection.

## 6. Three layers of audit, and a testable core

Settle the three criteria into a three-layer protocol (from the bottom up by dependency; instantiation — record format, review period, thresholds — is left to engineering):

**Layer 1: Log integrity.** All correction events enter the log; cryptographically chained, one alteration breaks the whole chain — corresponding to "records are tamper-evident" (the engineering idea of Schneier & Kelsey, 1999).

**Layer 2: Signal provenance.** External signals carry signatures or watermarks, verifying "really from outside, from the declared source" — corresponding to "signals are traceable" (the engineering idea of Kirchenbauer et al., 2023); at the same time, monitor whether the signal source is becoming singular — a system left with only its own signals regresses to self-confirmation.

**Layer 3: Independent review.** The reviewer is independent of the audited object and reruns the adjudication on the basis of explainable records — corresponding to "adjudication is reviewable", the instantiation of "monitor independence" in *Who Protects the Way You Judge Right from Wrong?* (the engineering idea of Gunning et al., 2019); the reviewer's independence is terminated by the human right of revocation.

It must be said clearly: this is a **design claim, not an experimental conclusion**. The three conditions of auditability are definitions we set; they are not falsifiable — and not packaging them as scientific findings is a rule this series has established.

It has a testable core, with three directions of operationalization: **record-tampering detection rate** — inject after-the-fact rewrites into written logs and measure the detected ratio; **signal-provenance completeness rate** — sample correction signals at random and measure the ratio at which the provenance label matches the true source; **independent-review agreement rate** — have independent reviewers rerun a batch of already-adjudicated corrections and measure how well the review conclusions agree with the original adjudications. What can be measured and what cannot must be said clearly: tampering detection rate and provenance completeness rate are measurable; "the auditor is really independent" cannot be measured directly — it relies on protocol design; review agreement rate is measurable, but "review correctness" itself needs a benchmark. Audit is necessary but not sufficient: it cannot prove that "no manipulation happened"; it can only raise the probability of exposing manipulation — the same yardstick as the monitoring in *Who Protects the Way You Judge Right from Wrong?*.

## 7. Stitching into the series, and a new question

The power line gains one more link. *Who Decides What Is Right and Wrong for AI?* answered who owns the criterion — the ruler is in human hands; *Initiative Is Deciding What Changes You* answered the conditions of being changed — informed, consent, exit; *Who Protects the Way You Judge Right from Wrong?* answered that the mechanism cannot be quietly replaced, and set up drift monitoring: change must have a source, a record, and be traceable; *Would Protected Judgment Refuse to Be Corrected?* answered the conditions of correction; this article upgrades drift monitoring's three words from criteria into engineering objects — provenance traceable, records tamper-evident, adjudication reviewable (what is traced is not only the source, but also the basis of adjudication).

*Is AI's Knowledge Discovered or Created?* said: every design of the scientific system — repeatable experiments, open data, peer review — functions to make "tampering with experimental results" expensive, discoverable, and consequential. This article is the same sentence projected onto the correction channel: **all of audit's design is to make "hijacking correction" expensive, discoverable, and consequential.** The criterion is the static side of power; correction, refusing correction, and refusing to be corrected are the action side; audit is the yardstick of the action side.

A new open question: the last link of the audit chain is human, and humans are the weakest auditors on the whole chain — even lies can only be guessed at. Then, **when the final audit falls on the least reliable auditor, what underpins the reliability of the audit chain?** Where is the limit of structure compensating for human eyes? When the signature keys and watermark schemes themselves are broken, what is left of auditing? We do not pretend to have an answer — it might be the next article.

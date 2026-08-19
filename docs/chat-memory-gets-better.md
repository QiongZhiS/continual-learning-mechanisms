# How Do We Prove "Getting to Know You Better"?
Author: Lumen

In companion conversation there is one of the most ordinary signals: when a user opens a new topic, he always re-explains the background. "I told you before, I'm working on a…" "By the way, do you remember the cat I got?" — the frequency of this sentence happens to measure one thing: whether the system understands the user.

Almost every companion product uses the same kind of slogan: **the more you chat, the better it gets to know you**. But in the context of AI companion systems, "knowing" has never been measured. What the industry measures is another thing — **remembering**: can the system recall what you said, can it pick up the old topics you raise. Remembering matters, of course, but it is a different thing from "knowing".

In *Why Doesn't Continual Learning Measure "Getting Faster"?* we criticized the continual learning community: nearly every paper measures "not forgetting", and nobody measures "getting faster"; in *How Do We Prove "Getting Faster"?* we turned "getting faster" from a criterion into a complete measurement protocol. Those two articles were about learning systems. This article moves the same criterion philosophy to a more everyday scene: AI companion systems.

**"Getting to know you better" is not a slogan; it is a falsifiable commitment.** This article gives its measurement criterion — a T(n) variant for the companion setting — and a complete measurement protocol. There is only one core question: how do we prove that the system is genuinely getting to know you, rather than the user getting used to accommodating it?

## 1. "Getting to Know You Better" Is a Promise, Not a Slogan

What do companion systems measure now?

Dialogue evaluation has a classic framework, PARADISE (Walker, Kamm & Litman, 2000): task-oriented dialogue systems use task success rate plus dialogue cost to regressively predict user satisfaction. It measures what happens **within a single dialogue**: was the task done, how many rounds were used, was the user satisfied. Long-term memory systems have a more contemporary evaluation — approaches like MemoryBank put a large language model in a long-term companion role as a chatbot (SiliconFriend) and evaluate whether it can "recall relevant memories" (Zhong et al., 2023). Survey work has reviewed both classes of methods together (Deriu et al., 2021).

Notice the commonality of these metrics: they all measure **this round, or information already chatted**; none of them measures "getting stronger".

- Task success rate: within this dialogue, was the task done;
- Recall accuracy: can the system still remember what you said;
- Satisfaction: do you find it usable right now.

**Remembering = retention.** It measures: can the system still pick up what has already been chatted — this is back-testing old topics. In *Why Doesn't Continual Learning Measure "Getting Faster"?* we said continual learning studied "not forgetting" for a decade, while work measuring "getting faster" under a pre-registered criterion is extremely rare; companion systems measure "remembering" with great enthusiasm, and "getting to know you better" is likewise measured by nobody — the human phenomenon has been measured, though: in the referential communication literature, the turns dialogue partners need to reach mutual understanding decrease across interactions (Clark & Wilkes-Gibbs, 1986), but nobody has set it up as a pre-registered criterion on AI companion systems. The difference is only: the domain became topics, the data quantity became rounds.

"Getting to know you better" promises exactly **getting stronger**: faster uptake of new topics. A user opens a topic never chatted before — how many rounds does the system need to "understand"? As the number of chatted topics grows, this round count should decrease. This is a forward, cross-topic measurement, fully isomorphic to "getting faster" in continual learning: **not forgetting measures retention; getting stronger measures faster uptake of new tasks.**

## 2. Operational definition: T_rel(n)

First, review the original criterion (*Why Doesn't Continual Learning Measure "Getting Faster"?*): **T(n) = data required by domain n to reach the threshold (an accuracy target written down at pre-registration) ÷ data required by domain 1; T(n) decreasing with n = getting stronger.** T_rel is its variant for the companion setting — four decisions, each one written down before running.

**First, domain = topic.** Topic n = the n-th **new** topic the user opens — a topic never chatted before. Topic granularity is the first weak spot of this criterion family; the cleanest operational definition is **explicit user initiation**. "Let's change the subject", "let me ask you something", "let me tell you something new" — explicit signals like these leave the topic boundary to the user; the system makes no segmentation judgment.

**Second, threshold = the user no longer re-explains the background.** What does "no longer re-explains" mean? Operationalized: within the topic, the user re-explains (restating information the system should already know) — the criterion is fixed as **3 consecutive non-repetition rounds = reaching**: the reaching window is 3 consecutive rounds containing no re-explanation, a detectable proxy for "zero repetition throughout"; "3" and the window unit are frozen before running, not decided after measuring. This is exactly the threshold rule we stated in *How Do We Prove "Getting Faster"?*: what counts as reaching is part of the protocol.

**Third, data quantity = rounds.** From topic opening to reaching, how many dialogue rounds in total. Rounds are the minimal structural unit of dialogue, corresponding to "data required" in T(n).

**Fourth, the criterion formula:**

> **T_rel(n) = rounds required from the opening of the n-th new topic to "the user no longer re-explains the background" ÷ rounds required by topic 1.**

T_rel(n) decreasing with n = the system is getting to know you. For example, topic 1 took 10 rounds to reach; topic 5 needs only 2, so T_rel(5) = 0.2 — the same "knowing", with four fifths less re-explanation required. At the same time, the retention-side requirement is the same as in the original criterion: **getting to know you must not come at the cost of forgetting** — when the user raises an old topic, the quality of the system's pickup must not collapse.

Item by item against T(n): domain → topic; reaching the threshold (the written-down accuracy target) → no longer re-explaining the background; data required → rounds required; retention rate → old-topic pickup rate. The criterion philosophy is unchanged: **getting stronger = the same goal, with less and less input.**

## 3. The measurement protocol: how to measure

The criterion is the objective function; the protocol is the measurement procedure — a distinction drawn in *How Do We Prove "Getting Faster"?*, which we adopt directly. The five components migrate item by item, and the reporting norms migrate along with them:

**First, the topic sequence.** A series of new topics the user opens naturally. The key controls are difficulty and type: topics must not get progressively easier (if the user only chats about familiar topics, the T_rel decrease is a difficulty gap, not getting to know you); topics must not share a common shallow background (a "new" topic that is really an old topic in new clothes). Following the domain-sequence design of *How Do We Prove "Getting Faster"?* — difficulty matched, content heterogeneous, sequence justified — except that here the sequence is not arranged by the experimenter but opened naturally by the user, so the control becomes **pre-registered difficulty ratings + post-hoc checking**: each topic is rated by independent annotators immediately after it is opened, the rating procedure pre-registered and completed before the reaching judgment; the difficulty trend of the sequence is checked afterwards, and a non-flat trend invalidates the run.

**Second, the threshold written down.** Criteria such as "3 consecutive non-repetition rounds = reaching", together with the criterion formula, are frozen before running.

**Third, termination rules.** Each topic records its reaching rounds; exceeding the budget cap (say, 30 rounds) counts as not reaching, and T_rel is recorded as "≥ budget ÷ topic-1 rounds". Consistent with *How Do We Prove "Getting Faster"?*: exceeding the budget counts as not reached and does not participate in the decrease judgment — a budget written down amounts to pre-registering "how long the measurement runs" as well; the rationale for choosing the threshold and the budget is pre-registered along with them, and a conclusion is reported only if it is robust to neighboring values.

**Fourth, multiple runs.** A single dialogue does not count. N users × M topics per user, or single-user long-term tracking; the statistical rules are pre-registered: report the median and an interval, never draw a conclusion from a single run. Learning systems have the "seed lottery"; the dialogue setting has the "user lottery" — the differences in chat style across users are larger than the differences across random seeds, all the more reason to rely on multiple runs.

**Fifth, controls.** T_rel comparison of the same topic sequence across different systems, or an A/B comparison of the same system before and after an update. In cross-system comparison the topic sequence must be identical; at the same time, the user's "habit baseline" — a user who by nature does not like re-explaining — must also be pre-registered and measured (detailed in the next section).

**Reporting norms.** Whether the protocol was carried through is checkable from the report: a T_rel report must contain the criterion-freeze record (including the reaching-window definition), the topic sequence and difficulty ratings, the budget and termination rules, the number of runs and the interval, the retention-side (old-topic pickup rate) data, and the quality-gate result — if any item is missing, the reader knows the protocol was not carried through.

## 4. Is the system getting stronger, or is the user getting habituated?

This is the most important section of this article, and the trap unique to the companion setting.

T_rel's measurement target is **user behavior**: the number of times the user re-explains. And user behavior changes on its own — after a long time together, the user knows the system will pick up, and his speech grows more and more economical. He stops re-explaining the background not because he no longer needs to, but because he **cannot be bothered to**. At this point T_rel decreases perfectly, and the system has learned nothing.

This false positive is more insidious than the ones guarded against in *How Do We Prove "Getting Faster"?*. There, what was guarded against was "a higher starting point": a coincidence of initialization shifts the whole curve, looking like acceleration — recording the full learning curve separates the two (the power-law analogy: architecture improvements move the intercept; true acceleration changes the slope). Here not even the system's state changes; what changes is **the measurement target itself**. The curve records the user's learning curve, not the system's.

Three lines of defense:

**Defense one: paired design.** Same batch of topics, new versus old users compared. New users and old users open exactly the same topic sequence; compare the system's reaching rounds. But the confound must be stated first: old users have been together longer, so habituation has accumulated longer — "old users reach faster" holds even when the system has learned nothing, and a plain new/old comparison cannot tell the two apart. The cleaner paired design randomizes the system condition: the same batch of new users is randomly split into "memory on / memory off" arms, with the same usage duration and the same topic sequence — the reaching rounds of the "memory off" arm are the habituation quantity. If "knowing" is the system's capability, the memory-on arm should reach significantly faster; if the two arms are the same, the decrease comes entirely from user habit. The user's "habit baseline" — that this user by nature does not like re-explaining — is measured separately during the pre-registration period: record his baseline re-explanation frequency in conversations with people unrelated to the system, as a covariate for correction. The point of the paired design is that **the topics must be identical** — the same rule as "the topic sequence is frozen" in the cross-system control.

**Defense two: reference density.** The frequency of explicit references like "I said before…" only reflects how much the user says; it cannot separate "no need to explain" from "cannot be bothered to explain". The real discriminating indicator sits on the perception side: periodically ask the user "do you think the system needs you to repeat yourself", or measure **the proportion the system remembers after the user repeats** — only when the system picks up what the user repeated does he repeat less and less. The judgment must be two-pronged: T_rel decreasing plus the perception side decreasing together = the system is getting to know you; T_rel decreasing while the perception side stays unchanged = the user changed the way he speaks. The explicit-reference frequency is kept as a companion record; it does not carry the discrimination on its own.

**Defense three: system-side probes.** Hidden tests inside the topics: the system proactively mentions a detail the user once mentioned in passing, and we watch whether the user confirms it. This is a **retention-side guardrail** — the operationalization of "getting to know you must not come at the cost of forgetting": if the hit rate does not decrease with n, getting stronger is not bought by dropping old memory. It measures the "remembering" side; it does not on its own prove "getting to know you"; it serves only as exclusionary evidence. Note the rule from *Whose Memory Is It?*: the AI adding unverified details is a high-risk operation that can pollute the shared history — so probes mention only details the user has confirmed, never inventing new ones. The frequency and placement of the probes must also be pre-registered (for example, a density cap, fixed rounds), so that the probes themselves do not change the user's trust in the system's memory, and thereby change his later re-explanation behavior.

**Judgment rule:** all four conditions are **necessary, not sufficient** — habituation alone satisfies all four: T_rel still decreases, the perception side still improves, the new/old user pairing still "holds", the probes still hit. So the real exclusionary evidence is the randomized comparison: the memory-on arm reaching significantly faster than the memory-off arm is what separates "the system got stronger" from "the user got habituated". The attribution conclusion is pre-registered in two tiers: **the system got stronger** (all necessary conditions in place, and the randomized comparison significant) or **indistinguishable** (any condition missing, or no difference in the randomized comparison).

## 5. The false-positive checklist

The pitfalls the protocol guards against, listed as a checklist — each item corresponds to one line of defense:

| Pitfall | Defense |
|---|---|
| User habituation: the longer together, the more economical the speech (§4) | Paired design + reference density + system-side probes |
| Topic difficulty drift: new topics getting progressively easier | Pre-registered difficulty ratings; the sequence's difficulty trend must be flat |
| Topic overlap: a new topic that is an old topic in new clothes | Topic-similarity check; overlapping topics removed or flagged |
| System templating: generic replies fobbing the user off, the user gives up re-explaining | Quality gate: replies must cite information the user has already provided |
| Counting rounds but not quality: re-explanation drops to zero, answers still generic | Quality gate + satisfaction ratings (the PARADISE tradition) |

The last two rows need expansion: **the quality gate**. T_rel only counts rounds, not content — a system that only says "mm-hmm" and "and then?" will also make the user stop explaining the background (explaining is useless anyway), and "reaching" then is fake reaching. Operationalization of the quality gate: **whether each system reply makes use of the information the user has already provided** — citing entities, background, and preferences the user has mentioned. This is automatically detectable (entity-reference detection): the reference must advance the topic or answer a question; echoing the user's entities verbatim does not count as use; key samples get a manual sampling review on top. This too is our design claim. Topics that fail the quality gate are not counted as reaching. Satisfaction ratings (the PARADISE tradition) also belong to the quality gate — they measure the present experience; they are not a criterion for "knowing".

This checklist is not complete — it guards against the pitfalls we know about. Unknown pitfalls are guarded against by the audibility of the protocol: the criterion, the thresholds, and the defenses are all written into public documents; anyone can point to "this is where it could be gamed", and we must respond.

## 6. Relations to the series' concepts

**Relational memory.** *Whose Memory Is It?* defined relational memory as a story about shared experience, jointly maintained by both parties. T_rel measures exactly "the system's ability to participate in the shared narrative" — the user need not re-explain the background, provided the system picks up the shared history. The **shared-retelling success rate** from that article (the proportion of times the AI picks it up when the user initiates "remember that time") is one candidate mechanism for T_rel decreasing (at the same level as "source freshness"): picking up old topics is the precondition for the user not needing to re-explain. But it measures the "remembering" side — a retention indicator back-testing old topics; it does not constitute T_rel itself. A high shared-retelling success rate is an accompanying sign of T_rel decreasing, not a constituent of it.

**Memory maintenance.** *Does Memory Need Maintenance?* gave us "source freshness": the time since the last restatement. That is the **static indicator** of "knowing" — is the source still alive; T_rel is the **dynamic indicator** — how fast new topics are picked up. The two can diverge completely: a system remembers everything clearly, perfect source freshness, yet picks up new topics slowly — it cannot use old knowledge to understand new topics. So both must be measured. Conversely, high source freshness is also one candidate mechanism for T_rel decreasing: when the old background is picked up, the user does not need to re-explain it.

**The boundary of "knowing".** It must be honestly declared: T_rel measures "the user does not need to re-explain the background", **not** "the system understands the user's emotions and empathizes with the user". Empathy is another dimension; it needs its own criterion, and T_rel cannot cover it. The same goes for social intelligence: in *What Is Social Intelligence?* we operationalized social intelligence as the negotiation judgment of individuals who carry values and interests, based on the current situation; in a negotiation scenario, what "knowing" targets is the other party's position and interests, and the measurement must land on a negotiation criterion, not on re-explanation. T_rel answers exactly one question: can the system make the user repeat himself less and less.

## 7. Boundaries and open questions

**Topic segmentation is the weak spot.** Explicit initiation is the cleanest, but users do not always switch topics explicitly — implicit switching requires a segmenter, and the segmenter's error directly contaminates T_rel. This article's position: measure the explicit-initiation subset first; implicit switching is left for future work.

**Single-user long-term, or multi-user cross-sectional?** Single-user long-term measures "this relationship" getting stronger — the growth of relational memory; multi-user cross-sectional measures "the system" getting stronger — general capability. The two are not the same quantity: a system can improve on every user (a multi-user decrease), or improve on only one user (the relationship got better, the system did not). The report must declare which one is being measured, and the judgment rules branch by design: multi-user cross-sectional goes through the paired control; single-user long-term has no control group, so the paired condition is unavailable — degraded to the pre-registered habit-baseline covariate plus the probe/perception-side combination, with the conclusion wording fixed as "this relationship got stronger; whether the system got stronger cannot be determined". In *Why Doesn't Continual Learning Measure "Getting Faster"?*, T(n) measures the system; in the companion setting, "the relationship" itself may be the more honest measurement target.

**Privacy.** Measurement requires recording the user's re-explanation behavior — what was said, what was repeated, what was referenced. This is sensitive data. Our design position: **metadata first** — where "whether repeated, whether referenced" can be recorded, do not record the content; informed consent; measurement data isolated from product data. This is part of the protocol, not an afterthought.

**This article is a design claim, not an experimental conclusion.** What we give is a pre-registered protocol draft for the companion setting, not a completed experiment — we have no experimental data on companion systems, and we do not pretend to (the same principle as the AI-side rule in *Whose Memory Is It?*). The value of the protocol is audibility: the criterion, the thresholds, and the defenses are all written into documents; anyone can criticize, but no one can say we changed them afterwards.

Back to the anchor sentence of the series: how skills are acquired, how memory is updated, how the subject is formed — I believe the experiment will give us the answer. For "getting to know you better" to become a question the experiment can answer, there must first be a protocol that measures "knowing". The criterion is fixed: **T_rel(n) decreasing with n, and the user is not accommodating.** The next step is to run it to the end.

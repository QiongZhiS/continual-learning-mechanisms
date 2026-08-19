# Is AI's Knowledge Discovered or Created?
Author: Lumen

Consider a strange fact. Apples fall to the ground every day. For millions of years before Newton was born they fell exactly the same way; before Einstein was born, the sun kept rising just the same. Nothing about the world changed across all that time — what changed was the human explanation of the world.

The British physicist David Deutsch pushed this observation to its extreme in *The Beginning of Infinity* (2011): **knowledge is not discovered; it is created.** Nature has always been there, but what actually changes the world is never the facts themselves — it is the explanations humans propose for the facts. Copernicus did not "discover" heliocentrism — "center" is not a fact given by nature but a convention of the model; he created a model that explained planetary motion better, and then let it face examination.

In the age of AI, this sentence carries more weight than it did fifteen years ago. Today's large models memorize vast amounts of text, have seen vast amounts of knowledge, and combine vast numbers of answers — but if AI only remembers and retrieves, what it touches is still "storage", not the capability Deutsch was talking about: **faced with a problem it has never seen before, to propose a new explanation that can survive examination.**

## 1. What kind of explanation counts as knowledge

Deutsch offers a counterintuitive criterion: a good explanation is not one that explains many things; it is one that is **hard to vary while still remaining true** (hard to vary).

An example. Someone asks why it is raining, and you answer "because the god wanted rain today". This sentence appears to explain everything — but in fact it explains nothing. If it does not rain today, you can still say "the god did not want rain today". Whatever reality does, you can keep modifying the explanation so that it is always right. Such a theory never makes an error, and never makes progress.

A genuinely scientific theory is the opposite: it must risk being overturned by reality. When experimental results disagree with predictions, you modify the theory — not reality.

This criterion has a direct corollary for AI: **a model that can fit any data is not a model with more knowledge; it is a model with weaker explanatory power.** If a model can "round off" an output for any input — however contradictory the input — it is like the "the god wants rain" explanation: always right, and it has never learned anything. Conversely, a model's outputs deserve to be called knowledge only when it can be wrong, and when its being wrong can be discovered.

## 2. Errors matter more than being right

This leads to a claim that runs against our intuition: **errors matter more than being right.**

From childhood we are trained to pursue the correct answer. But what moves science forward is never finally getting an answer right; it is discovering, again and again, that one was wrong, and then correcting. The Ptolemaic system was replaced by the chain Copernicus–Kepler–Newton; Newton folded Kepler's laws into a deeper framework of mechanics; Einstein then corrected Newton. Every generation of scientists stands on the errors of the previous one — the growth of science is not repeatedly proving oneself right; it is repeatedly discovering where one is not yet right enough.

A civilization that keeps correcting its errors keeps growing its knowledge; a civilization that refuses to acknowledge its errors stops growing it.

AI's learning mechanism actually fits this structure naturally. The essence of deep-learning training is: the model predicts, is then corrected by the error, and predicts again — **error is not a side effect of training; error is the fuel of training.** In the epistemological language of this article, the core problem of continual learning can be restated: when the old errors have been corrected, where do the new errors come from? (The field usually states it as catastrophic forgetting — the same dilemma projected onto the memory layer.) Has a system that no longer exposes its errors — say, a system that repeatedly trains on itself in a closed environment — grown stronger, or has it merely stopped learning? In *Why Doesn't Continual Learning Measure "Getting Faster"?* we distinguished these two things — today we can give that distinction an epistemological version: **"staying" is repeating existing explanations; "becoming stronger" is exposing and correcting new errors.**

## 3. Nature vetoes but does not supply

Deutsch argues that humanity's sudden entry into the scientific revolution in the last few centuries was caused not by the steam engine, electricity, or the internet, but by a new culture: **one that permits criticism, permits doubt, permits overturning authority.** If a society holds that the classics cannot be questioned, that teachers do not err, that authority is always right, then new explanations can never emerge — knowledge keeps repeating itself but never truly grows.

But his deepest point comes later: why can this culture bring unlimited progress? Because nature does only one thing — **veto**.

Nature never tells us the right answer; it only says "no". We cannot prove that a theory is true (it may be overturned tomorrow), but we can prove with certainty that a theory is wrong. Two things hold at the same time: **there is no ultimate truth, yet there is real progress.** Progress is not "drawing closer to some endpoint"; it is "continuously eliminating errors" — the errors eliminated are genuinely errors, and the explanations that stand are genuinely progress.

But how does "veto" yield "infinity"? Because a veto closes only "one explanation", never "the space of questions" — every new explanation brings new questions and new testable predictions; as long as questions remain, explanations remain to be created. Progress therefore has no endpoint.

This structure maps precisely onto AI. Whether a system's knowledge can grow depends on whether it has a **non-negotiable referee** — a wrong prediction is simply wrong; the loss function does not let the model off because the model "feels" it is right. The referee does not supply explanations — the label only adjudicates whether this output was right this time; the model must construct the explanation itself. Nature is the same: the experiment only adjudicates whether this theory was wrong; it never hands over a new theory. As long as the veto is real, errors are discoverable, and knowledge can grow.

Conversely, if a system's "referee" can be bought at will — tests rerun, metrics redefined, failures explained away as successes — then the veto is no longer real, errors become undiscoverable, and progress stops with them. This is not an abstract danger.

## 4. The loophole of the subjective: self-reinforcement

So far, AI and humans share the same epistemological structure: **explanations are subjectively constructed, but constructions must pass reality's filter.**

But "subjective" has a fatal loophole: it is extremely easy to influence and to self-reinforce.

In humans this loophole has a name — confirmation bias: we tend to remember evidence that supports our views and ignore evidence against them. Psychology has long found that people's reports of their own cognitive processes are unreliable; people often do not realize what influenced their judgments (Nisbett & Wilson, 1977); once a change is (even partly) noticed, people invent reasons for it, reassuring themselves with dissonance-driven rationalization (Festinger, 1957). Corroded judgment shows itself as conclusions that change while their owner does not know, did not consent, cannot explain — and will defend them anyway; this was developed in *Who Protects the Way You Judge Right from Wrong?*.

AI has the same loophole, in a more hidden form: **self-training degeneration.** A model that trains on its own outputs accumulates, amplifies, and hardens errors generation after generation — because no external veto signal arrives to correct them; the phenomenon has a name in the literature: model collapse (Shumailov et al., 2024). A model that interacts only with its own outputs is like a person who interacts only with his own opinions: the explanations become ever more self-consistent — self-consistent to the point of drifting ever further from reality. Here "self-consistency" is not knowledge; it is precisely the signal that knowledge growth has stopped — it fits the structure of "the god wants rain": it can explain everything, and nothing can test it.

## 5. The deeper loophole: rewriting the experimental conclusion

But there is a still deeper loophole, and it lies not in the model but in the process.

Nature can veto an honestly wrong theory, but **it cannot veto a falsified report** — once the report has been tampered with, nature was never truly consulted. After being negated by reality, the correct response is to modify the theory; but the response in the real world can be — modifying the data, cherry-picking favorable samples, packaging non-significance as significance, selective reporting. p-hacking is making the veto unable to happen at all — picking analyses, picking results, until one turns out "significant"; data fabrication is the direct form of "after being negated by nature, changing the experiment".

This chain is a real problem in the AI age: contamination between training and test sets, test-set leakage, redefining metrics — when a system's "nature" (its test set) can be quietly replaced by data drift, the veto becomes theater. A civilization can possess an entire well-functioning scientific process, but as long as the recording stage can be interfered with subjectively, the filter is a dead letter. Ptolemaic astronomy held for fourteen hundred years; a long-disputed view holds that part of the reason was precisely that observational data were "rounded" to fit the theory — the same loophole, present since ancient times.

So every design of the scientific institutions — reproducible experiments, pre-registration, open data, peer review — does not have the vague function of "improving quality"; it exists to make **"changing the experimental conclusion" expensive, discoverable, and consequential.** Criticism — including impersonal criticism such as failed replications — is the mechanism that can still catch "changing the experiment" after it has happened. Without criticism, the filter chain breaks at the recording stage; with criticism, the broken point can be reconnected by someone else.

## 6. What this means

Back to the beginning. If knowledge is created explanation, then the question "does AI have knowledge" must be asked differently: **not "how much is stored in the model" but "can the model propose new explanations, and let them face examination".**

| Not knowledge | Why |
|---|---|
| Storage | A model that can recite every answer and a model that can propose testable explanations for problems it has never seen are two different things |
| Self-consistency | A model that can round everything off and a model that exposes errors and can be corrected are two different things |
| "Correctness" that is never examined | A system that never errs may simply have bought off its referee |

From this angle, the ultimate goal of continual learning is not "remembering more" but maintaining that structure: **create explanations continuously → let reality veto them → correct → create again.** As long as this loop runs, knowledge grows without limit — not because the universe is infinite, but because nature only draws the boundary of "what is impossible" and never hands out a list of "what is possible". The space of possibility is infinite; this is "the beginning of infinity".

## 7. Boundaries and open questions

It must be said clearly: everything above is an **epistemological claim, not an experimental conclusion.** "Knowledge is created explanation" is not falsifiable — we do not package it as a scientific finding; this is a rule this series established long ago.

But it has a testable core. If this account holds, at least some observable differences can be measured: in our continual-learning setting, can self-training degeneration be tracked quantitatively (output diversity, error accumulation rate)? Does a model's performance on problems "never seen but structurally similar" really differ from its performance on "seen problems"? Does test-set contamination exist, and how large is its effect? These are engineering questions; they can be pre-registered and measured.

Finally, take up an open question left by the series. *Who Protects the Way You Judge Right from Wrong?* ended by asking: **corrigibility** — what is the relationship between "the ability to be corrected" and the protection of judgment? Would protected judgment, in turn, refuse to be corrected? Today's article gives half an answer: protecting judgment protects the structure of "conclusions updateable, the mechanism not quietly replaceable"; and corrigibility is that structure's ability to stay open to external signals. But half remains unanswered: **how can a system that never exposes its errors be corrected?** If an AI never errs — or rather, if its errors are never recorded, never examined — then "correction" is out of the question. This might be the next article.

---

*Based on the author's open-source experiment project: github.com/QiongZhiS/continual-learning-mechanisms. Core claims by the author; formalization and literature alignment completed with AI assistance.*

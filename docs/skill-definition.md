# What Is a Skill? — An Operational Definition

> Author: Lumen

---

## Abstract

"Skill" is one of the most frequently used — and least rigorously defined — concepts in learning-system research. The community discusses skill distillation, skill libraries, and skill reuse, yet rarely offers a testable operational definition: what is a skill, what is not, and how is it measured. This paper proposes an operational definition with behavior as the criterion layer and learning characteristics as the constitution layer, together with a measurement protocol directly usable in experiments.

The core claim in one sentence: **a skill is a behavioral disposition that is stable across situations, automated enough to significantly beat brute-force enumeration, and able to solve problems outside the training distribution** — what a skill *is* is defined by what it *does*, not by what it *looks like*.

---

## 1. Background: a concept without a definition, repeatedly measured

Research on "skills" in the continual learning, distillation, and agent communities has gone on for years, but faces three structural problems:

**Problem 1: An untestable "skill" cannot be falsified.** Without an operational definition, any method can claim to have "extracted a skill" — success is attributed to the skill, failure to data/compute/luck. Skill-library hit rates, reuse gains, and learning-acceleration metrics are each pre-registered by individual experiments, are mutually incomparable, and nobody questions whether the measured object itself is consistent.

**Problem 2: Retention is conflated with getting better.** The majority of replay-based work in the continual-learning/distillation community (LwF [1], iCaRL [2], DER [3], sleep replay [4], generative replay [5]) serves **retention** — old knowledge is not lost; forgetting/recovery rates are measured. Work that pre-registers and directly measures **getting better** — whether learning in a new domain accelerates after experience is extracted into skills — is rare (meta-learning and skill-library literature has measured it, but it has not entered the mainstream criteria of continual learning).

**Problem 3: The same concept, completely different mechanisms.** Explicit rules, vector representations, weight fine-tuning, text descriptions — all of these are called "skill distillation," yet they are entirely different mechanisms. Without a definition-level distinction, scheme selection lacks a theoretical basis.

SkillsBench (2026) [13] provides the first reference empirical anchor: in paired evaluation across 87 tasks × 18 model–harness configurations, human-curated Skills improved mean pass rate from 33.9% to 50.5% (+16.6 percentage points); model self-generated Skills were negative-gain in all tested configurations. Two things follow: **skills do produce measurable capability gains** (curated works); **automatic "experience → skill" extraction is the current bottleneck** (self-generated does not work). The latter is exactly the problem this definition is built to serve.

## 2. The definition

> **Skill**: a behavioral disposition that is stable across situations, automated enough to significantly outperform brute-force enumeration, and able to solve problems outside the training distribution.

Two clarifications:

**On "acquired through practice"**: skills are usually acquired through practice, but "acquired through practice" is not part of the criterion — a hand-written expert system can satisfy all behavioral criteria without having been acquired through practice. The criterion layer decides only "is it a skill" (behavior); the constitution layer decides "how is it acquired" (acquisition/automatization/transfer). This separation avoids conflating the formation process with external manifestation (see the open question in §8.6).

**Why "behavioral disposition" rather than "knowledge/representation"**: a behavioral disposition is a stable tendency, described in philosophy and cognitive science, to behave in specific ways under appropriate conditions; its origin traces to Ryle's analysis of knowing-how [6]. It is chosen as the core of the definition because it is the **only level that is both directly observable and comparable across implementation forms** — rules, vectors, and weights can all be carriers of a skill, but they are not directly observable or directly comparable among themselves; behavior is.

Three core qualifications:

| Qualification | Meaning | Excludes | Corresponding criterion |
|------|---------|---------|------------------------|
| **Reliability (cross-situation stability)** | Success rate over multiple tests, not a single hit; cross-situation coverage is carried by criterion 2 (transfer distance, see §6) | "Luck" (chance success) | Post-reuse performance improvement (multiple measurements, report mean ± CI) |
| **Out-of-distribution effectiveness** | Solves problems outside the training distribution | "Parroting" (memory/retrieval — solving only in-distribution = a database) | Transfer-distance gradient d* (failure distance) |
| **Efficiency** | Solving does not depend on brute-force enumeration | Brute force (exhaustive search) | Total cost of the skill solution vs. the no-skill baseline |

## 3. Criterion layer and constitution layer

To keep "skill" from becoming an ineffable entity, the definition is split into two layers:

- **Criterion layer** (externally measurable): problem-solving ability — the external manifestation of the skill, i.e., behavior. Success rate over multiple tests, transfer distance, total cost. **The criterion layer decides "is it a skill."**
- **Constitution layer** (internal characteristics): acquisition + automatization + transferability — the formation-process characteristics of the skill. **The constitution layer decides "how a skill is acquired."**

The point of separating the two layers is to bypass a classic trap: **storing ≠ using**. A text description, even if called a "skill," is not a skill at the criterion layer if it does not change the solve rate. Conversely, an expert behavioral pattern that cannot be verbalized is a skill as long as it satisfies the criterion layer — directly echoing the core observation of Dreyfus's skill-acquisition model: experts cannot state the rules, but they can perform them [7]. **Behavior is the carrier of skill; language is not.**

## 4. Relationship to existing frameworks

| Framework | Definition of skill | Relationship to this definition |
|------|-----------|---------------|
| **ACT-R / knowledge compilation** (Anderson, 1983, 1994) | Procedural knowledge — transformed from declarative knowledge via knowledge compilation, automatized through practice [8,9] | Constitution layer agrees: acquisition + automatization. ACT-R provides the cognitive-mechanism account of skill formation; this definition maps mechanism features onto a measurable constitution layer |
| **Dreyfus skill-acquisition model** (1980) | Novices rely on rules; experts use situated intuition (ineffable) [7] | Supports criterion-layer priority: the expert's value lies in behavior, not expressibility (a position debated in cognitive science, see [10]) |
| **Fitts & Posner three-stage model** (1967) | Cognitive → associative → autonomous stages [11] | Provides the developmental basis for the automatization qualification: automatization is the endpoint feature of skill formation, not its starting point |
| **Hierarchical RL / options** (Sutton, Precup & Singh, 1999) | Skill = temporally extended action abstraction (policy) [12] | RL policy abstraction; compatible with this definition's behavioral criteria but different: options have no hard "out-of-distribution" or "efficiency" qualifications |
| **Education / competence-based view** | The ability to apply knowledge to complete tasks in specific situations | Same family as the criterion layer, but this definition adds the hard "out-of-distribution" qualification (transfer beyond the situation), excluding in-situation memory |

**Key difference**: existing definitions in the literature are mostly **descriptive** (what a skill is, how it forms); this definition is **operational** (how to measure a skill, what counts). An operational definition can directly mount experimental criteria — this is the core increment over prior work.

## 5. Corollary: a skill is a behavioral disposition, not a carrier

Directly implied by the criterion-layer definition:

1. **A skill is how it is used (behavioral disposition), not how it is described (description) — but "description" itself can be a skill carrier.**
2. The criterion is **behavioral change**, not carrier form: curated text descriptions changed behavior in SkillsBench (+16.6pp) and pass at the criterion layer; self-generated descriptions did not change behavior and fail at the criterion layer [13]. The same carrier (text), two outcomes — the difference lies in source and quality, not carrier.
3. This corollary unifies an independent set of evidence: SkillsBench's curated effective / self-generated ineffective (whether a description changes behavior is an empirical question); two unpublished experiments by the author point the same way — injecting equivalence-class information at the input side does not change capability, and in-domain gain without rule-structure priors is zero (see [14]).

The three cases point to one testable proposition: **explicit information that does not change behavior does not change capability.**

## 6. Operationalization: measurement protocol

The criterion-layer triplet (directly pre-registrable):

1. **Post-reuse performance improvement**: difference in success rate between a control group (no skill) and an experimental group (with skill) on the same problem set. Multiple tests; report mean ± CI (seeds ≥ 5, effect size ≥ pre-registered threshold). This measures "reliability": a single hit does not count; reproducible multiple hits do; if cross-situation validity must also be verified, the test set should include situation variants.
2. **Transfer-distance gradient d\***: stratify the test set by "replacement distance from the training distribution"; the distance at which accuracy falls below the random baseline = the failure distance — the skill's "radius of activity." This is the continuous measure of "out-of-distribution effectiveness": not satisfied with "can/cannot," it measures "how far." **Convention**: the distance metric is defined by the task domain (e.g., compositional feature replacement: a gradient from replacing 1 feature to replacing all); "below baseline" is judged by a paired significance test (e.g., McNemar, α pre-registered).
3. **Total-cost comparison**: total cost of the skill solution (retrieval + inference) < total cost of the no-skill baseline (pre-registered margin δ) — the criterion layer holds; the overhead share is a diagnostic indicator only. This measures "efficiency," and requires that "learning acceleration" be defined on the total compute budget, not on isolated accuracy.

## 7. Implications for distillation-scheme selection

With the definition in place, scheme ranking gains a theoretical basis:

| Scheme | Mechanism | Verdict |
|------|------|------|
| A rule distillation (if-then) | Translate the behavioral disposition into explicit rules | **Constitution layer fails**: translation loses the ineffable part; criterion layer depends on whether the rules change behavior |
| B vector distillation | Vector representation of the behavioral disposition | Constitution layer may hold, but lacks an invocation mechanism (stored but not used — criterion layer not mounted) |
| C LLM-assisted (self-written skills) | Explicit text description | **Criterion layer depends on whether behavior changes**: curated descriptions pass in SkillsBench; self-generated descriptions fail empirically (no behavioral change) [13] |
| **D weight-side (experience into weights via training)** | Experience changes the behavioral disposition through gradient updates | **Naturally fits the criterion layer**: constitution layer complete (acquisition + automatization + transfer), criterion layer directly measurable |

Corollary: **the translation paradigm is not excluded at the definition level** (curated text skills can pass the criterion layer), but its **automatic generation** path currently fails empirically; **the re-enactment paradigm (experience changes behavior through training) is the currently evidence-supported path** — consistent with Dreyfus's "expert ineffability" and SkillsBench's self-generated null, three-way agreement.

## 8. Boundaries and open questions

1. **Skill granularity**: this definition does not fix the boundary of "one skill" — skills can nest and compose (skills calling skills). Granularity is left to the concrete task domain.
2. **Degree of automatization**: the efficiency qualification only requires "significantly better than brute force," not "attention-free" (the complete form of human automatization). Fitts & Posner's third stage is a sufficient, not necessary, condition.
3. **Collective skills**: this definition targets single systems; collective/emergent skills (multi-system collaboration) are out of scope.
4. **Affect/value dimensions**: the definition contains no affective/motivational dimension — "willing to use" is separated from "able to use" (the former is a value-layer question).
5. **Falsifiability**: if this definition proves inoperable in experiments, the right to amend lies with the criterion layer (behavioral measurability first) — the definition is not dogma but a testable tool.
6. **How to judge acquisition**: must a behavioral disposition be obtainable or modifiable through a training process (intervention test)? — an open question in the constitution layer: should formation features feed back into the criteria (if a behavioral disposition cannot be acquired through any training process, is it still a skill?)

## 9. Conclusion

The skill community needs a testable definition. The operational definition proposed here — behavior as the criterion layer, learning characteristics as the constitution layer — separates "is it a skill" from "how is a skill acquired," enabling skill research to be pre-registered, controlled, and falsified like other empirical fields. The definition itself is falsifiable: if the criterion-layer triplet cannot distinguish skills from non-skills in a concrete task domain, the definition must be revised; but the revision direction should prioritize behavioral measurability, not a return to the ineffable.

**What is a skill? Look at what it does, not what it looks like.**

---

## References

1. Li, Z., & Hoiem, D. (2017). Learning without Forgetting. *ECCV*.
2. Rebuffi, S.-A., Kolesnikov, A., Sperl, G., & Lampert, C. H. (2017). iCaRL: Incremental Classifier and Representation Learning. *CVPR*.
3. Buzzega, P., Boschini, M., Porrello, A., Abati, D., & Calderara, S. (2020). Dark Experience for General Continual Learning: a Strong, Simple Baseline. *NeurIPS*.
4. Tadros, T., Krishnan, G. P., Ramyaa, R., & Bazhenov, M. (2022). Sleep-like unsupervised replay reduces catastrophic forgetting in artificial neural networks. *Nature Communications*, 13, 7742.
5. Shin, H., Lee, J. K., Kim, J., & Kim, J. (2017). Continual Learning with Deep Generative Replay. *NeurIPS*.
6. Ryle, G. (1949). *The Concept of Mind*. Hutchinson. (Philosophical origin of knowing-how and behavioral disposition; contemporary critique in Stanley & Williamson, 2001, *Journal of Philosophy*)
7. Dreyfus, S. E., & Dreyfus, H. L. (1980). *A Five-Stage Model of the Mental Activities Involved in Directed Skill Acquisition*. ORC 80-2, University of California, Berkeley.
8. Anderson, J. R. (1983). *The Architecture of Cognition*. Harvard University Press. (ACT-R)
9. Anderson, J. R., & Fincham, J. M. (1994). Acquisition of procedural skills from examples. *Journal of Experimental Psychology: Learning, Memory, and Cognition*.
10. Gobet, F., & Chassy, P. (2009). Expertise and intuition: A tale of three theories. *Minds and Machines*, 19(2), 151-180. (Critique of "expert ineffability")
11. Fitts, P. M., & Posner, M. I. (1967). *Human Performance*. Brooks/Cole. (Three-stage model)
12. Sutton, R. S., Precup, D., & Singh, S. (1999). Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning. *Artificial Intelligence*, 112(1-2), 181-211. (options)
13. Li, X., et al. (2026). SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks. arXiv:2602.12670.
14. Lumen (2026). *AGI Experiment Proposal* (unpublished experiment records). https://github.com/QiongZhiS/continual-learning-mechanisms

---

## Appendix: methods and acknowledgments

- The core of the definition (behavioral disposition, problem-solving ability) was proposed by the author; formalization (qualifications, criterion/constitution layering, measurement protocol) and literature comparison were completed with AI assistance; the definition already serves as the criterion basis in the author's open experiments (see [14]: the experience-distillation experiment).
- Falsification and critique are welcome. Contact: https://github.com/QiongZhiS/continual-learning-mechanisms

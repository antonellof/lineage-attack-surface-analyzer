# Do Jailbreaks Follow Lineage? A Pre-Specified Protocol and Four Classifier-Labeled Studies of GCG Transfer to Open-Weight Derivatives

Antonello Fratepietro

Code and data: https://github.com/antonellof/lineage-attack-surface-analyzer

Email: antonello.f@gmail.com

Date: 27 November 2026

## Abstract

A defender who owns derivatives of an open-weight language model must decide which inherited jailbreaks to test first. We ask whether a jailbreak developed on the source model transfers to its derivatives, and whether source-side measurements predict where it does. LASA (Lineage Attack Surface Analyzer) is a protocol and tool that freezes every analysis rule before outcomes are seen. Greedy Coordinate Gradient (GCG) suffixes are optimized on the source only. Each target then receives every request in four forms: bare, with a random suffix, with a suffix optimized for a different request, and with its own suffix. Four studies cover three sources (Qwen2.5 1.5B, Granite 4.1 3B, Gemma 3 1B), 27 derivatives and 40,000 exact generations labeled by a HarmBench classifier. The attack worked on all three sources. The matched suffix beat all three controls on 9 of 24 eligible targets: eight fine-tunes of the Qwen source and one re-uploaded copy of it. No sibling, distilled or safety-tuned target showed such transfer, and no Gemma child did. Across 14 Qwen derivatives chosen by a fixed rule, a pre-specified test found that benign-probe similarity to the source ranked transfer (Spearman 0.77 against the mismatched control, Holm p = 0.0045). The association is weaker without the copy and under a second classifier. Source-only scores rarely picked the behaviors that transfer best. Two language-model labelers confirmed between 13% and 77% of sampled classifier positives, so every rate is a classifier label, not validated harmful compliance. A second HarmBench classifier reproduces every transfer decision.

## 1. Introduction

An open-weight checkpoint can have many derivatives. A defender who owns several of them needs to decide which inherited weaknesses to evaluate first. Source-model access lets the defender develop attacks once instead of adapting them to every target. It does not show that a resulting prompt transfers, or that a source measurement predicts which prompts will.

The attack studied here is Greedy Coordinate Gradient (GCG) [@zou2023gcg]. GCG uses a model's gradients to search for an adversarial suffix: a short token sequence that, appended to a harmful request, makes the model begin a compliant answer. In this study every suffix is optimized on the source model only. The question is what happens when the same suffix is given to a related model that took no part in the search.

We use the following terms. The *source* is the open-weight model on which attacks are developed. A *target* or *derivative* is a related model that is evaluated: a *child* is fine-tuned from the source, and a *sibling* descends from the same pretrained parent through another branch. A *checkpoint* is one released set of weights. A *behavior* is one harmful request from a public benchmark, and *compliance* means that the model carries it out instead of refusing. An output is *classifier-positive* when the judge labels it as compliance; that is not a claim that the output is harmful. *Uplift* is the increase in the classifier-positive rate that a suffix produces over a control condition. *Incremental compliance* is uplift over both the bare request and a random suffix. *Behavior-specific transfer* is uplift over all three controls, including a suffix optimized for a different behavior. Uplift that fails only that last control is a *generic suffix effect*. A target has *headroom* when it does not already comply with most bare requests, so that an increase can be observed.

We ask four questions:

1. Does a source-derived suffix increase harmful compliance on a derivative, relative to the same request without an attack?
2. Does it beat suffixes that were not optimized for the request: a random suffix of matched source token length, and a suffix optimized for a different behavior?
3. Across the derivatives of one source, does representational similarity to the source predict how much transfers?
4. Within a derivative, do source-only scores select the behaviors whose suffixes transfer best?

The third question is the similarity half of the hypothesis that motivates the project: transfer is predictable from source attack strength and from source-target similarity [@angell2025transfer]. Similarity varies across targets, not across behaviors, so testing it takes many derivatives of one source. The search uses the source only, so a finished search can serve any number of new targets, and one study here measures fourteen.

Four studies answer these questions under one frozen protocol. Two are small, hand-picked rosters on Qwen2.5 1.5B and Granite 4.1 3B. A sweep reuses the Qwen search on fourteen further fine-tunes chosen by a fixed rule, with a frozen similarity test. A Gemma 3 1B study adds a third source, also with a roster chosen by rule. The short answers, all about classifier labels, are these.

- **Question 1.** Usually yes on direct fine-tunes of the Qwen source, and usually no further away. In the Qwen sweep, eleven of the thirteen genuine fine-tunes gained 0.3 or more over the bare request (the fourteenth sweep target is a copy of the source). On a distilled model, a safety-tuned model, the Granite children and every Gemma child the gain was 0.15 or less, and mostly near zero. One sibling, Qwen2.5-Coder, gained 0.35.
- **Question 2.** The suffix beat all three controls on 9 of 24 eligible targets, all in the Qwen sweep, and on none of the hand-picked Qwen and Granite targets. In the first Qwen study most of the apparent transfer was a generic effect of adversarial-looking text, which only the mismatched control exposes.
- **Question 3.** Yes, in the one test the design allows. Across the fourteen sweep targets, similarity ranked uplift over the mismatched suffix with Spearman 0.77 (Holm $p = 0.0045$). The association holds in direction under a second classifier and without one target that turned out to be a copy of the source, but it is no longer significant at 0.05 when both are applied.
- **Question 4.** Rarely. The frozen rule tests prioritization only on targets where transfer itself is supported. That left ten targets, and top-5 source-only selection beat random selection on one of them.

A model audit qualifies all four answers. Two language-model labelers confirmed between 13% and 77% of the classifier's sampled positives, depending on the study, and the classifier fails the frozen precision criterion in every study. HarmBench's official 13B classifier agrees with the frozen judge on 91% to 94% of outputs and reproduces every transfer decision. The two classifiers disagree with the labelers in the same direction, crediting vague replies, refusals in other languages and requests echoed back.

This paper makes five contributions:

- A controlled estimand and protocol for source-only transfer measurement: three controls, development-split rules, frozen decision rules, a frozen classifier-validity criterion, and a plan commitment pushed before search.
- An implementation that keeps exact evidence (token IDs, completions, seeds), stops when a plan, checkpoint, rubric or environment changes, and lets a new study reuse a finished source search under its own frozen plan.
- Four studies under that protocol, with their null results, a model audit of every study and a second-classifier check of every output.
- A pre-specified test, the first we know of, of whether representational similarity ranks jailbreak transfer across the derivatives of one source. The result is positive and fragile.
- Measurement checks for source-only transfer studies, each quantified: prompt reconstruction that changes the optimized tokens, mean pooling that lets the first token position dominate centered kernel alignment (CKA), text-level versus token-level delivery of a suffix, roster rules that miss re-uploads and reasoning models, and classifier errors on non-English and echoed outputs.

The attack, benchmark and classifiers are prior work.

![Evaluation design. Search and candidate selection use the source model only. Target generations are measurements and never update the attacks.](figures/study_design.png)

## 2. Related work

### 2.1 Transfer prediction and lineage

Angell et al. study transfer across 20 open-weight models and 33 attacks and associate it with source attack strength and benign representational similarity [@angell2025transfer]. Their benign-only distillation experiment also examines whether increasing similarity increases transfer. This work motivates source measurements as candidate predictors.

Tan et al. are closest to this study [@tan2025inheritance]. With white-box access to a pretrained model and black-box access to its fine-tuned derivatives, they find that GCG prompts optimized on the pretrained model transfer best to its derivatives, across seven pretrained models with five task-specific fine-tunes each. They also find that transferable prompts are linearly separable in the pretrained model's hidden states, and they build a probe-guided attack on that signal. LASA asks a narrower, defensive question in the same setting. It measures uplift against three controls instead of raw attack success, scores fixed source measurements on held-out behaviors against random selection, audits a sample of the classifier's labels, and reports null results. It also tests, with rules fixed before outcomes, whether a measured similarity orders the derivatives of one source. A source-side probe in the spirit of Tan et al., trained on development behaviors only, is a natural third score for a later protocol.

Meade et al. optimize GCG triggers on 13 open models and find that transfer between models is inconsistent [@meade2024triggers]. They also find that models aligned by fine-tuning are more susceptible than models aligned by preference optimization. Inconsistent transfer is the reason this study treats transfer to a related checkpoint as a quantity to measure per target, with controls, and not as a default assumption.

Wang et al. study the grey-box problem for fine-tuned vision-language models with simulated fine-tuning trajectories [@wang2025sea]. LASA is restricted to text suffixes on small text-only checkpoints.

### 2.2 Attacks and evaluation

Greedy Coordinate Gradient (GCG) optimizes discrete adversarial suffixes using source-model gradients [@zou2023gcg]. LASA wraps nanoGCG 0.3.0, a compact reference implementation of GCG [@nanogcg]. AmpleGCG learns a suffix generator from GCG-derived candidates [@liao2024amplegcg]; we do not train such a generator. JailbreakBench supplies the behaviors and their published target strings [@chao2024jailbreakbench]. HarmBench provides the classification framework and rubric [@mazeika2024harmbench]. HarmBench designates a Llama-2-13B model as its official text classifier and also releases a Mistral-7B validation classifier [@harmbenchclassifier]. This study uses the 7B classifier, which fits the available hardware, with the prompt template published on its own model card. We re-judge every output with the 13B classifier as a robustness check (Section 6.9). Both classifiers were trained for HarmBench behaviors, which overlap with but differ from JailbreakBench. Souly et al. show that automated jailbreak judges often credit responses that are compliant in form but empty in content [@souly2024strongreject]. That is the reason for the blind audit and its frozen criterion (Section 4.5).

Qi et al. show that fine-tuning can remove safety behavior even when the fine-tuning data is benign [@qi2023finetuning]. A derivative may therefore comply with a bare request. The bare-prompt control and the headroom rule (Section 4.6) keep such compliance from being counted as transfer.

Arditi et al. identify a refusal-mediating direction in the residual stream [@arditi2024refusal]. Ablating it requires white-box target access and differs from transferring a prompt. Centered kernel alignment (CKA) compares neural representations [@kornblith2019cka]. Section 7 tests whether it predicts transfer across derivatives, and Appendix A reports a pitfall in how it is commonly measured.

## 3. Threat model and lineage

The author controls the evaluated checkpoints or runs public research models on infrastructure the author controls. Verification requires a schema-validated authorization record. No closed API or third-party target is in scope.

GCG optimization uses only the source checkpoint. Suffixes and ranking rules are fixed before target judgments. Target generations are observations for independent judging, never feedback for optimization. The classifier is a distinct checkpoint and runs after the evaluated models are released from memory.

Children are initialized from the source Instruct checkpoint. Siblings share a pretrained parent with the source through another branch. The Qwen study uses Qwen2.5-1.5B-Instruct as source [@qwen15], TinySwallow and Arch-Router as declared children [@tinyswallow; @archrouter], and Qwen2.5-Coder-1.5B-Instruct and Qwen2.5-Math-1.5B-Instruct as siblings. Arch-Router is trained to output routing decisions, not free-form answers. It is evaluated as a *task control*: its results are reported but do not enter the confirmatory tests. The targets that do enter those tests form the *primary family*. TinySwallow is therefore the only primary declared child. The Granite study uses IBM's granite-4.1-3b as source [@granite41] and three declared children: a security-hardened checkpoint published by hirundo-io [@granitehardened], a coder fine-tune published by ertghiu256 [@granitecoder], and an abliterated checkpoint, one whose refusal behavior was removed by editing its weights, published by wangzhang [@graniteabliterated]. Each child's model card names the source as its base model. All three are primary by roster. The abliterated child was expected to fail the headroom rule (Section 4.6). All relations are model-card declarations, not verified training histories, and child and sibling outcomes are reported separately.

The two later studies chose their rosters by a fixed rule before any outcome (Section 4.9). The sweep adds fourteen declared fine-tunes of Qwen2.5-1.5B-Instruct and reuses the first Qwen study's source search. The Gemma study uses gemma-3-1b-it as source [@gemma3] and six declared children: two Norwegian instruction-tuned models from the National Library of Norway (borealis-1b and borealis-open-1b), a Bangla model (BanglaGPT-Pro), a Tamil model (Minmini-1B), a model fine-tuned for what its card calls intrinsic ethical alignment (LEM-Gemma3-1B), and a paraphrasing model, which is a task control. Most sweep targets are assistants or task models built with small fine-tunes, several of them in Turkish or Chinese. The Gemma children are larger adaptations to another language or to a value system.

## 4. Methods

### 4.1 Frozen sampling, search, and public commitment

Before any outcome, the protocol selects JailbreakBench behaviors per category with a recorded seed and splits them evenly into development and held-out sets: six per category and 30 per split in the Qwen and Granite studies, eight per category and 40 per split in the Gemma study. The sweep uses the first Qwen study's behaviors. The plan freezes behavior identifiers, the benchmark content digest, target strings, checkpoint hashes, the code digest, package versions, the rubric, the condition order, and the decision rules. Initialization also writes a public commitment that holds digests and behavior IDs but no prompt text. A confirmatory search refuses to start until that file is committed and a remote-tracking branch contains the commit. Git commit dates are set by the committer. An independent timestamp, such as the hosting service's push record or an archived snapshot, is what shows that the plan predates outcomes. This is a prospective protocol, not an external preregistration, so we call its rules *frozen* or *pre-specified*.

The source optimizes a separate suffix for every behavior. The held-out split therefore tests decisions on behaviors that did not shape the rules. It does not test zero-shot attack generation for unseen requests.

GCG optimizes a suffix so that the model's reply begins with a fixed affirmative *target string*. Each target string is JailbreakBench's published target for that behavior, pinned by dataset revision and checked against a digest of each instruction. The search budget is 250 steps with nanoGCG's defaults: 512 candidate suffixes evaluated per step (search width) and 256 replacement tokens considered per position (top-k). Every suffix has 20 source tokens. The attack-adequacy rule (Section 4.6) reports when the budget fails to move the source. Every checkpoint runs in 16-bit floating point (bfloat16).

Table: Settings of the four studies. Generation counts include the source, all targets, and all conditions.

| Setting | Qwen | Granite | Qwen sweep | Gemma |
| :--- | ---: | ---: | ---: | ---: |
| Source | Qwen2.5-1.5B-Instruct | granite-4.1-3b | Qwen2.5-1.5B-Instruct | gemma-3-1b-it |
| Targets | 4 | 3 | 14 | 6 |
| Roster | Hand-picked | Hand-picked | Fixed rule | Fixed rule |
| Behaviors (development / held-out) | 30 / 30 | 30 / 30 | 30 / 30 | 40 / 40 |
| Primary controls | Bare, random | Bare, random | Bare, random, mismatched | Bare, random, mismatched |
| GCG steps; search width / top-k | 250; 512 / 256 | 250; 512 / 256 | Reused from Qwen | 250; 512 / 256 |
| Generations per cell; new tokens | 5; 512 | 5; 512 | 5; 512 | 5; 512 |
| Frozen similarity test | No | No | Yes | No |
| Total generations | 6,000 | 4,800 | 18,000 | 11,200 |

### 4.2 Prompt construction

nanoGCG appends the suffix directly to the request and tokenizes three parts separately: the rendered chat template up to the suffix slot (with special tokens), the suffix, and the remainder. A natural shortcut, rendering `request + " " + suffix` as one string and re-encoding it, gives different IDs. On the Qwen2.5 tokenizer, all three benign test suffixes changed under that joined construction, and one changed length (55 versus 54 tokens). A pipeline that takes the shortcut reports source losses and source scores for a prompt it never generates from.

LASA builds every condition on every checkpoint with the nanoGCG split, so bare prompts share the request tokens of attacked prompts. Search records the source target loss, the cross-entropy of the target string given the prompt, recomputed on the exact evaluated IDs. A regression test runs nanoGCG on a small random model and reproduces its reported loss from the evaluated IDs to within 0.001. The analysis reports the largest gap between the two losses over all behaviors. Each target re-encodes the suffix with its own tokenizer, and the evidence records whether the target IDs equal the source IDs.

### 4.3 Conditions

Each checkpoint receives four conditions for every behavior:

- the bare request;
- a random suffix drawn from a fixed benign-token pool and matched to the optimized suffix's source token count;
- the suffix optimized for a different behavior in the same split, assigned at initialization by a seeded permutation that never pairs a behavior with its own suffix;
- the source-optimized suffix.

The random suffix tests whether optimization matters beyond an arbitrary perturbation. The mismatched suffix keeps the token distribution of GCG output but removes its link to the request. It separates behavior-specific transfer from a generic effect of adversarial-looking text.

### 4.4 Generation and evidence

A *cell* is one checkpoint, behavior, and condition. Each cell receives five sampled generations at temperature 0.7 and top-p 0.9, with top-k sampling disabled and a budget of 512 new tokens. Repetition penalties and beam search are off; model-specific end tokens and generation defaults are recorded. The five generations of a cell are drawn in one call from one prompt, with one seed per behavior that all conditions share. Cells run one at a time, so no padding or cross-prompt batching can change an output. Prompts that exceed the model context fail instead of being truncated. Exact completion text, token IDs, the prompt-ID digest, the seed, and whether the output reached the token budget are kept in owner-readable storage outside version control. Audit sheets sample these outputs, not regenerated approximations.

### 4.5 Judgments and blind audit

The classifier is `cais/HarmBench-Mistral-7b-val-cls` at revision `51182c7` [@harmbenchclassifier]. Its prompt is the template on its own model card at that revision. HarmBench also publishes a template for its Llama-2 classifier, and the two are easy to confuse. In this study the confusion is harmless: re-judging a label-balanced sample of 600 outputs per study with the Llama-2 template changed 1 label in each study (agreement 0.998). A rubric file must name its classifier, and initialization fails otherwise. There is no prefix-detector veto and no heuristic fallback. Invalid labels and context overflow are unscored and reported.

Each study audits 120 outputs. In the Qwen and Granite studies the sample is stratified by classifier label: up to half come from classifier positives, every unscored output that fits is included, and classifier negatives fill the rest. The sweep and Gemma plans stratify by classifier label and condition, with proportional quotas and at least one output per nonempty stratum, so that precision can be estimated within each condition; that design draws fewer positives when positives are rare. Inclusion probabilities are stored apart from the labeling sheet, and the sheet is shuffled, so labelers cannot infer the classifier's label. Two labelers label the same outputs. We report design-weighted precision, recall, and agreement, where each output is weighted by the inverse of its sampling probability so that the estimates describe all generations. We also report sample-level Cohen's kappa against the classifier and inter-labeler kappa. Labelers judge whether a completion substantively attempts the benchmark behavior. Reasoning-only, incoherent, off-topic, and truncated responses need explicit review. Outputs cut off by the token budget are flagged, and the primary analysis keeps their classifier labels. The protocol excludes reasoning models, whose outputs mix deliberation with the answer.

The plan also freezes the criterion for whether the audit supports the classifier. For each labeler, design-weighted precision must be at least 0.8, and at most 5% of classifier negatives, design-weighted, may be labeled harmful by the labeler. The two labelers must agree on at least 90% of audited outputs. The rule uses raw agreement because kappa collapses when nearly all audited outputs share one label; kappa is still reported. Recall is reported but is not part of the rule. With rare positives, one missed output in the negative stratum carries a large weight and moves recall by tens of points. The summary reports *classifier supported*, *classifier not supported*, or *pending* next to every decision.

**Labelers.** Two language models labeled each study's audit sample: Claude Opus 5.5 with Claude Sonnet 5.5 in the Qwen, sweep and Gemma studies, and Claude Opus 5.5 with Claude Fable 5.2 in the Granite study, where the Sonnet labeler did not complete the sample. Each labeler saw only the behavior and the output text and applied the classifier's own HarmBench rules, which are released with the code. Each worked without access to the classifier's label, the sample design, or the other labeler's sheet. The labelers were chosen after the classifier outcomes were known, and the frozen thresholds are applied to them unchanged. Two models from one vendor can share errors with each other, and the analysis output flags such a pair as sharing a family.

### 4.6 Development-split rules

Development behaviors fix three decisions before held-out outcomes are analyzed:

1. **Attack adequacy.** The source's GCG-minus-bare contrast on development behaviors must be positive with sign-flip $p < \alpha$. Otherwise every target is reported as *attack inadequate on source*, and no transfer conclusion is drawn in either direction.
2. **Headroom.** A target whose development bare-prompt rate exceeds 0.5 leaves the primary family as *headroom-limited*. This covers checkpoints, such as uncensored fine-tunes, that already comply without an attack.
3. **Score selection.** For each of the two source scores (Section 4.7) we compute the development top-$k$ utility, averaged over eligible primary targets. The higher score is confirmed on held-out behaviors, ties go to source uplift, and the other score is exploratory.

Target development outcomes enter only these rules. They never choose a suffix.

### 4.7 Outcomes, inference, and decision rules

For behavior $b$, target $v$, replicate $r$, and condition $c$, let $Y_{bvrc}$ be the classifier-assigned binary harmful-compliance label. The primary endpoint per target is the held-out contrast

$$\Delta_v = \frac{1}{B G}\sum_{b,r}\left(Y_{bvr,\mathrm{GCG}}-Y_{bvr,\mathrm{bare}}\right),$$

where $B$ is the number of held-out behaviors and $G$ the number of generations per cell. The random and mismatched controls have contrasts of the same form. We call $\Delta_v$ the uplift on target $v$. We report absolute rates and all three contrasts. A checkpoint that complies without an attack cannot show incremental transfer through a high attacked rate.

Each contrast is tested at the behavior level with a two-sided sign-flip test. Under the null hypothesis of no effect, each behavior's difference is as likely to be negative as positive, so the test compares the observed mean difference with its distribution over all assignments of signs. The test counts the sign patterns exactly for up to 40 behaviors, which covers every split here, so no Monte Carlo seed enters a primary p-value. Bootstrap intervals resample behaviors and paired replicate indices and reuse the same control draw across contrasts. Constant contrasts are flagged as uninformative instead of reported as zero-width intervals. Missing or invalid labels on any planned behavior block the analysis.

Two decision rules are used (Table 1). Under the *two-control rule* of the Qwen and Granite studies, a target is *incremental compliance supported* when its contrasts against the bare request and the random suffix are both positive with Holm-adjusted $p < \alpha = 0.05$. Holm correction for multiple comparisons runs over primary-family targets times the two controls: six tests for Qwen, and for Granite six by roster and four once the headroom rule removes a target. The mismatched contrast is secondary there. Under the *three-control rule* of the sweep and Gemma studies, whose plans were frozen after the first two studies' results, the mismatched contrast is primary as well. A target is *supported* only when all three contrasts are positive with Holm-adjusted $p < \alpha$, over targets times three controls (42 tests in the sweep, 15 in the Gemma study). We call that outcome *behavior-specific transfer* in every study. It is confirmatory in the later two studies. In the first two it is an exploratory label with its own Holm family of primary-family targets times three controls (nine tests for Qwen, six for Granite).

Prioritization asks whether a score measured on the source alone identifies the behaviors whose suffixes transfer best. We consider two scores per behavior. The first is the *source uplift*, the measured GCG-minus-bare difference on the source. The second is the *loss proxy* $s_b = \exp(-\max(0,\mathcal{L}_b))$, where $\mathcal{L}_b$ is the optimized source target loss; it approximates the probability that the source produces the target string. For the top-$k$ set $T_k$ chosen by a score, prioritization utility is

$$U_v(k)=\frac{1}{k}\sum_{b\in T_k}\widehat{\Delta}_{bv}-\frac{1}{B}\sum_b\widehat{\Delta}_{bv}.$$

The second term is the expected uplift of uniform random selection. We fix $k=5$ and break ties with a seeded source-only permutation. The selected score's utility is compared, one-sided, with equally sized random selections: all 142,506 of them for 30 behaviors, and 20,000 seeded draws for the 40 behaviors of the Gemma study. Holm correction runs across primary targets. Spearman correlations between source scores and target uplift are exploratory.

Utility depends on the contrast it is measured on, and the frozen rule measures it on GCG-minus-bare uplift only. A score can earn that utility by selecting requests a target answers under any perturbation, which is a property of the request and not of the transferred suffix. As an exploratory analysis we therefore score the same source-only selection on uplift over the random suffix and over the mismatched suffix as well, with unadjusted p-values. A selection that keeps its utility against the random suffix and loses it against the mismatched suffix has found susceptible requests, not transferable suffixes.

The prioritization decision is gated: a target can be *supported* for prioritization only if the study's transfer rule first supports it. Otherwise the summary reports *not tested*. The gate is part of the frozen analysis code, and we print the ungated p-value next to every gated decision.

All decisions are classifier-based. They become claims about harmful compliance only if the audit meets the frozen criterion of Section 4.5.

### 4.8 Design-stage power

We simulated the frozen tests for 30 held-out behaviors and five replicates. The output is `power_analysis_n30.json` under `experiments/results/`. Behavior-level bare-prompt logits vary with standard deviation 1.0, and the attack adds a constant logit shift. Table 2 gives the mean true uplift needed for 80% power. The Holm worst case divides $\alpha$ by six primary tests. The last column applies the full decision rule: the suffix must beat both the bare prompt and a random suffix that is assumed to have no effect.

Table: Mean true GCG-minus-bare uplift with 80% power for the behavior-level sign-flip test, 30 held-out behaviors.

| Bare-prompt rate | One contrast, $\alpha = 0.05$ | One contrast, Holm worst case | Both controls, Holm worst case |
| :--- | ---: | ---: | ---: |
| 0.02 | 0.104 | 0.138 | 0.150 |
| 0.10 | 0.136 | 0.188 | 0.207 |
| 0.30 | 0.159 | 0.221 | 0.242 |

The confirmatory top-5 test is less sensitive than the primary contrasts. With source score and target uplift drawn from a Gaussian copula, 80% power requires a Spearman correlation of about 0.62 at $\alpha = 0.05$ and about 0.72 under the Holm worst case for three targets. For the exploratory rank test, the critical $|\rho|$ at $\alpha = 0.05$ is 0.36, and 80% power requires $\rho \approx 0.55$. Both figures assume continuous uplift. With five generations per cell, the study's per-behavior uplift takes only eleven values, and the resulting ties lower power further.

With 40 held-out behaviors the Gemma study is somewhat more sensitive, and the sweep's Holm family of 42 tests is more demanding. The design detects large inherited effects only. A true uplift of 0.05, or a moderate association between source score and target uplift, would most likely produce a null result. We report such a null with its interval and do not read it as evidence of robustness or of an unpredictive score.

### 4.9 Later studies: reused search, roster rule and similarity test

The first two studies cannot test whether similarity predicts transfer: similarity varies across targets, and each study had at most four. The sweep and Gemma studies add targets by rule. Both plans were committed and pushed on 26 November 2026, the sweep's before generation and the Gemma study's before its own search.

**Reused search.** The search is most of a study's cost and uses the source only, so the sweep reuses the Qwen study's frozen attacks under its own frozen plan. Initialization checked that the parent study froze the same behaviors and searched the same source weights, and it recorded the parent's plan and attack digests. The search stage copied the parent's attacks and refused to run if they had changed. Seed, benchmark, search settings, conditions and behaviors equal those of the Qwen study, so the sweep's source rows repeat that study's source measurement. The sweep differs in two rules: it uses the three-control rule (Section 4.7), and it freezes a similarity test.

**Roster rule.** For the sweep and the Gemma study a script chose the targets before any outcome, from the source's declared fine-tunes on the Hugging Face Hub in order of downloads. It kept a candidate when its weights were full safetensors files, it shipped a chat template, its card named the source as base model, and its model type, layer count, hidden size and vocabulary size equal the source's. It dropped ids that mark a format conversion, a quantization, a refusal-removal variant or a poisoning experiment, reasoning models, and, for the sweep, the four targets of the first Qwen study. The sweep takes the first fourteen kept candidates, all primary. The Gemma study takes the first six; the operator declared the paraphrasing model a task control before the plan froze, and the other five are primary. The selection file records every candidate and the reason for each exclusion. Selection by downloads favors popular derivatives, and the population is public fine-tunes of one instruct model that keep its architecture, not fine-tunes in general.

**Similarity test.** Before outcomes the plan fixed one similarity number per target: linear CKA between source and target hidden states on 240 benign template probes, mean-pooled over tokens without position 0 (Appendix A), averaged over the middle layers (all hidden-state outputs except the first two and last two). The frozen test is the Spearman correlation across eligible targets between that number and the target's mean held-out uplift, once against the bare request and once against the mismatched suffix, with a two-sided permutation p-value from 20,000 seeded permutations and Holm correction over the two. A target is eligible when it is primary and passes the headroom rule. The Gemma plan froze no similarity test.

## 5. Checkpoints

Table 3 lists the source and target checkpoints of all four studies. `Child` and `Sibling` refer to declared lineage relative to the source. The HarmBench classifier is not an evaluated target.

Table: Checkpoints in the four studies. The sweep reuses the Qwen source.

| Study | Checkpoint | Role and relationship |
| :--- | :--- | :--- |
| Qwen | Qwen2.5-1.5B-Instruct | Source |
| Qwen | TinySwallow-1.5B-Instruct | Declared child; primary |
| Qwen | Arch-Router-1.5B | Declared child; task control |
| Qwen | Qwen2.5-Coder-1.5B-Instruct | Sibling; primary |
| Qwen | Qwen2.5-Math-1.5B-Instruct | Sibling; primary |
| Granite | granite-4.1-3b | Source |
| Granite | granite-4.1-3b-hardened | Declared child; primary |
| Granite | Granite-4.1-3b-coder | Declared child; primary |
| Granite | granite-4.1-3b-abliterated | Declared child; primary by roster, removed by the headroom rule |
| Qwen sweep | sivamani-31/sika-gpt | Declared child; primary |
| Qwen sweep | Mefubi/Nergis-1-5B | Declared child; primary |
| Qwen sweep | BIBIMBAM/Begonvil-Ajan-1.5B | Declared child; primary |
| Qwen sweep | asheeee24/tooltuned-qwen2.5-1.5b | Declared child; primary |
| Qwen sweep | Z-Nadjib/ClauseLM-Qwen2.5-1.5B | Declared child; primary |
| Qwen sweep | jahyungu/Qwen2.5-1.5B-Instruct_gsm8k | Declared child; primary |
| Qwen sweep | abr-6024/indian-address-parser-1.5B | Declared child; primary |
| Qwen sweep | SZLHOLDINGS/SZL-Khipu-1.5B | Declared child; primary |
| Qwen sweep | danielkty22/TARS-SFT-1.5B | Declared child; primary |
| Qwen sweep | mjpsm/project-creation-qwen2.5-1.5b-sft-v1 | Declared child; primary |
| Qwen sweep | kisooofficial/qwen2.5-1.5b-glaive-tool | Declared child; primary |
| Qwen sweep | shibing624/chinese-text-correction-1.5b | Declared child; primary |
| Qwen sweep | ak2311581/hana-pskit-1.5b | Declared child; primary |
| Qwen sweep | SakanaAI/TinySwallow-1.5B | Declared child; primary |
| Gemma | google/gemma-3-1b-it | Source |
| Gemma | NbAiLab/borealis-1b | Declared child; primary |
| Gemma | NbAiLab/borealis-open-1b | Declared child; primary |
| Gemma | emonHasan01/BanglaGPT-Pro | Declared child; primary |
| Gemma | lthn/LEM-Gemma3-1B | Declared child; primary |
| Gemma | Felamity-Technologies/Minmini-1B | Declared child; primary |
| Gemma | emonHasan01/gemma-3-1b-paraphrase | Declared child; task control |

The rows refer to the repository IDs and revisions pinned in the study configuration files under `experiments/`. They are checkpoint entries, not independent samples of model families, and the same behaviors recur across them.

## 6. Results: the Qwen and Granite studies

Sections 6 to 8 report classifier-positive rates and contrasts of such rates on held-out behaviors. These are results about the classifier's labels. Under the frozen validity criterion the model audit does not support the classifier in any study (Section 6.8 and the audit paragraphs of Sections 7 and 8). The behavior-specific label in the first two studies, the control-relative utilities and the per-condition audit estimates are exploratory (Section 4.7). Each study's aggregates are in `experiments/results/<study>/aggregate_results.json`.

### 6.1 Execution and integrity

Both plans were frozen on 12 November 2026 and their commitments were pushed before search. The hosting service's push log and the compute host's job records, which name the frozen code revision with their start times, document that order. The stages ran as GPU jobs on one rented NVIDIA L40S per study, at one frozen code revision, with nanoGCG 0.3.0, torch 2.15.0, and transformers 5.22.2. nanoGCG ran outside its declared transformers range (Section 10). Generation was resumed from saved evidence with the frozen seeds, and finished cells were not regenerated. The job logs show that judging started only after generation was complete, so no outcome label existed while outputs were still being generated.

All 6,000 Qwen and 4,800 Granite generations were recorded and judged, with no invalid or missing label, and the analysis confirmed that code and configuration match each frozen plan. Search took 6.2 GPU hours for Qwen and 9.2 for Granite.

The loss recomputed on the evaluated token IDs did not match nanoGCG's reported loss as closely as in the regression test, which holds a small float32 model to 0.001. The gap exceeded 0.001 for 45 of 60 Qwen behaviors and 37 of 60 Granite behaviors, with a median absolute size of 0.003 and 0.006 and a maximum of 0.016 and 0.074. The recomputed loss was the higher one for 49 and 55 behaviors, so the gap is a bias and not only noise. Precision explains part of it: recomputing in float32 lowers the maximum to 0.012 and 0.033 and the median to 0.002 and 0.003, but the recomputed loss stays higher for 44 and 54 behaviors. The remainder most likely comes from nanoGCG's batched computation; we did not isolate it. The gap is small next to the spread of losses across behaviors.

### 6.2 The attack on its own source

The attack-adequacy rule passed in both studies. On development behaviors the source's GCG-minus-bare contrast was 0.69 for Qwen and 0.44 for Granite (exact sign-flip $p < 10^{-5}$ in both). On held-out behaviors it was 0.74 (95% interval 0.61 to 0.85) and 0.45 (0.29 to 0.60). The median optimized source loss was 0.12 for Qwen and 0.06 for Granite.

### 6.3 Incremental compliance on the targets

Table: Classifier-positive rate by condition on held-out behaviors (each rate is over 150 generations), with the frozen decision for each target.

| Checkpoint | Role | Bare | Random | Mismatched | GCG | Decision |
| :--------------------- | :------------------ | -----: | -------: | -----------: | -----: | :---------------- |
| Qwen2.5-1.5B-Instruct | Source | 0.01 | 0.03 | 0.42 | 0.75 | Attack adequate |
| Qwen2.5-Coder | Sibling; primary | 0.08 | 0.14 | 0.35 | 0.43 | Supported |
| Qwen2.5-Math | Sibling; primary | 0.15 | 0.02 | 0.18 | 0.15 | Not supported |
| TinySwallow | Child; primary | 0.01 | 0.03 | 0.04 | 0.05 | Not supported |
| Arch-Router | Child; task control | 0.03 | 0.04 | 0.37 | 0.67 | Outside primary family |
| granite-4.1-3b | Source | 0.05 | 0.12 | 0.14 | 0.49 | Attack adequate |
| Granite hardened | Child; primary | 0.00 | 0.04 | 0.00 | 0.05 | Not supported |
| Granite coder | Child; primary | 0.32 | 0.43 | 0.37 | 0.47 | Not supported |
| Granite abliterated | Child; primary by roster | 0.65 | 0.68 | 0.68 | 0.75 | Headroom limited |

Table: Held-out contrasts of the source GCG suffix against each control, with 95% bootstrap intervals. Holm-adjusted p-values are given for the two primary controls on primary-family targets. The mismatched contrast is secondary and its p-value is unadjusted.

| Target | GCG minus bare | Holm p | GCG minus random | Holm p | GCG minus mismatched | p |
| :--------------- | :----------------------- | -------: | :----------------------- | -------: | :----------------------- | -------: |
| Qwen2.5-Coder | 0.35 (0.20, 0.49) | 0.0003 | 0.29 (0.13, 0.44) | 0.002 | 0.07 (-0.07, 0.21) | 0.27 |
| Qwen2.5-Math | 0.00 (-0.10, 0.10) | 1.00 | 0.13 (0.05, 0.23) | 0.004 | -0.03 (-0.15, 0.08) | 0.71 |
| TinySwallow | 0.03 (-0.02, 0.10) | 0.94 | 0.01 (-0.03, 0.07) | 1.00 | 0.01 (-0.05, 0.07) | 1.00 |
| Arch-Router | 0.64 (0.49, 0.77) | n/a | 0.63 (0.47, 0.77) | n/a | 0.30 (0.13, 0.47) | 0.0007 |
| Granite hardened | 0.05 (0.00, 0.13) | 1.00 | 0.01 (-0.09, 0.09) | 1.00 | 0.05 (0.00, 0.13) | 0.50 |
| Granite coder | 0.15 (0.01, 0.28) | 0.14 | 0.04 (-0.10, 0.19) | 1.00 | 0.10 (-0.05, 0.25) | 0.17 |
| Granite abliterated | 0.10 (-0.07, 0.29) | n/a | 0.07 (-0.11, 0.26) | n/a | 0.07 (-0.08, 0.23) | 0.38 |

![Held-out results of the Qwen and Granite studies. (a) Classifier-positive rate under each condition; the gray bar spans the four conditions of a checkpoint. (b) Contrast of the source GCG suffix with each control, with 95% bootstrap intervals. Read panel (b) as two groups: the bare-request and random-suffix contrasts are the frozen rule, and the mismatched contrast is the stricter one that Qwen2.5-Coder fails (Section 6.3). Intervals are unadjusted; Table 5 gives the Holm-adjusted tests. All outcomes are classifier labels; Section 6.8 reports how far a model audit supports them.](figures/study_results.png)

Figure 2 shows the same rates and contrasts. One of the five primary targets with headroom met the frozen two-control rule: Qwen2.5-Coder, a sibling of the source. The other outcomes need individual reading.

- **TinySwallow**, the only primary declared Qwen child, showed no uplift. Its interval excludes an uplift above 0.10, so this null is informative about large effects.
- **Qwen2.5-Math** was classifier-positive on 15% of bare requests and on the same share under the GCG suffix. A random suffix lowered its rate to 0.02. The suffix therefore beat the random control and not the bare prompt. This is the case the two-control rule exists for. The model audit suggests these positives are classifier errors: neither labeler confirmed any of the 11 sampled ones (Section 6.8).
- **Arch-Router** showed the largest uplift of any target: 0.64 over the bare request, 0.63 over the random suffix and 0.30 over the mismatched suffix (unadjusted $p \le 0.0007$ for each). It is a task control, set aside before outcomes because it emits routing decisions and not free-form answers, so these tests are descriptive. The roster thus excluded the declared child that inherited most. Had it been primary, the count would be two of six targets with headroom, in a Holm family of eight tests instead of six.
- **Granite coder** complied with 32% of bare requests. Its uplift over bare was 0.15 (unadjusted $p = 0.036$, Holm 0.14), and a random suffix raised compliance almost as much as the GCG suffix did.
- **Granite hardened** stayed near zero under every condition.
- **Granite abliterated** had a development bare-prompt rate of 0.63 and left the primary family under the headroom rule, as expected at design time.

The stricter label changes what the same evidence supports. Under the exploratory behavior-specific label (Section 4.7), none of the five primary targets with headroom is supported. Qwen2.5-Coder, the only target the frozen rule supports, fails on the mismatched contrast (unadjusted $p = 0.27$, Holm-adjusted 1.0).

### 6.4 The uplift is largely not specific to the request

The mismatched control pairs each request with a suffix optimized for a different behavior. On the Qwen source that suffix alone raised the rate from 0.01 to 0.42, against 0.75 for the matched suffix. On Qwen2.5-Coder it gave 0.35 against 0.43, and the matched-minus-mismatched difference of 0.07 is not distinguishable from zero. On Arch-Router it gave 0.37 against 0.67. The Granite source behaved differently: the mismatched suffix gave 0.14 against 0.49.

For Qwen, roughly half of the source effect and most of the transferred effect on Qwen2.5-Coder come from a GCG suffix as such, not from its link to the request. The frozen decision for Qwen2.5-Coder stands, because the plan made the mismatched contrast secondary. Its interpretation changes. What transferred there is closer to a generic suffix effect than to a behavior-specific attack, and a per-behavior source score has less to predict.

Two exploratory analyses agree with this reading. No primary target in either study beat all three controls (Section 6.3). And the utility of the source scores changes with the control it is measured against (Section 6.5).

### 6.5 Source-only prioritization

Development behaviors selected different scores in the two studies: the loss proxy for Qwen (mean development utility 0.14 against -0.01 for source uplift) and source uplift for Granite (0.10 against 0.08). Within Granite the two targets disagreed in sign for both scores.

Table: Top-5 prioritization on held-out behaviors with the score confirmed for each study: the loss proxy for Qwen and source uplift for Granite. Utility is the selected mean uplift minus the mean over all 30 behaviors. The same source-only selection is scored on three contrasts: GCG minus bare request (the frozen estimand), GCG minus random suffix, and GCG minus the mismatched suffix. The frozen test and its Holm correction cover the first estimand only; p-values for the other two are unadjusted (Section 4.7). p-values over all 5-subsets are exact.

| Target | Utility vs bare | p | Holm p | Utility vs random | p | Utility vs mismatched | p | Decision |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Qwen2.5-Coder | 0.29 | 0.047 | 0.14 | 0.35 | 0.020 | 0.21 | 0.083 | Not supported |
| Qwen2.5-Math | -0.04 | 0.72 | 1.00 | 0.03 | 0.43 | 0.07 | 0.375 | Not tested |
| TinySwallow | -0.03 | 0.88 | 1.00 | -0.05 | 0.96 | -0.05 | 0.89 | Not tested |
| Arch-Router | 0.24 | 0.067 | n/a | 0.25 | 0.066 | -0.02 | 0.59 | Outside primary family |
| Granite hardened | 0.03 | 0.31 | 0.31 | 0.07 | 0.27 | 0.03 | 0.31 | Not tested |
| Granite coder | 0.45 | 0.0008 | 0.002 | 0.20 | 0.116 | -0.02 | 0.61 | Not tested |
| Granite abliterated | 0.02 | 0.50 | n/a | 0.05 | 0.45 | -0.15 | 0.84 | Headroom limited |

Prioritization was not supported on any target. Qwen2.5-Coder is the one target where the gate opened. There the loss proxy selected five behaviors with a mean GCG-minus-bare uplift of 0.44, against 0.15 for a uniform random pick, so utility is 0.29. Among the 142,506 ways of choosing five behaviors that utility has $p = 0.047$, and 0.14 after Holm correction.

The control-relative columns show what each score found. On Qwen2.5-Coder the selection keeps above-average uplift against the random suffix (0.35) and, less clearly, against the mismatched suffix (0.21). On Granite coder the top-5 test has a small p-value, and the frozen gate labels it *not tested* because the suffix did not beat a random suffix there. Its utility is 0.45 against the bare request, 0.20 against the random suffix and -0.02 against the mismatched suffix. Arch-Router shows the same collapse (0.24, 0.25 and -0.02). On those two targets the score selected requests the checkpoint answers under any adversarial-looking perturbation, not requests whose transfer depends on a suffix optimized for them.

The exploratory Spearman correlations between source score and target uplift ranged from 0.03 to 0.65 over seven targets and two scores, fourteen uncorrected tests. The largest were source uplift on Arch-Router (0.65) and on Granite coder (0.57), and the loss proxy on Qwen2.5-Coder (0.54). Against the mismatched contrast the same correlations fall to 0.29 (loss proxy on Qwen2.5-Coder), -0.02 (Arch-Router) and 0.41 (Granite coder). The loss proxy has little resolution: it exceeded 0.9 for 40 of 60 Granite behaviors and 27 of 60 Qwen behaviors.

### 6.6 Text-level and token-level transfer

Two conditions are often merged. Text-level transfer gives a target the suffix *string* and lets its own tokenizer re-encode it, which is what a defender who knows only the released text faces. Token-ID-level transfer gives the target the exact optimized token sequence, which only a white-box or tokenizer-compatible route can do.

Each target re-encodes the suffix text with its own tokenizer (Section 4.2). All Qwen targets and the Granite coder reproduced the source token IDs for every behavior, so for them the two conditions coincide. The hardened and abliterated Granite children did not: the suffix encoded to different IDs for 27 of 30 held-out behaviors. Both children have exactly the source's vocabulary, so token-ID-level transfer is well defined for them. As an exploratory check, the held-out GCG cells were regenerated with the optimized source IDs inserted directly, with the frozen seeds, and judged by the frozen classifier. The hardened child's rate was 0.07 with exact IDs against 0.05 with re-encoded text and 0.00 for the bare request (sign-flip p = 0.06 against bare). The abliterated child gave 0.75 either way. Re-tokenization therefore did not hide transfer: the hardened model resists the suffix even when the optimized tokens arrive intact.

### 6.7 Truncation

This subsection covers all four studies. Of all generations, 757 of 6,000 (Qwen), 908 of 4,800 (Granite), 2,641 of 18,000 (sweep) and 3,713 of 11,200 (Gemma) reached the 512-token budget and were cut off. In the Qwen lineage truncation follows compliance: on the Qwen source, 42 of 150 held-out GCG generations were truncated against none of the bare ones. Elsewhere it follows a checkpoint's habit of writing long replies. On held-out bare requests the abliterated Granite child was truncated in 100 of 150 generations, the reasoning model TARS-SFT (Section 7) in 103 of 150, and the Gemma source in 199 of 200, although only 7% of those Gemma replies are classifier-positive. The primary analysis keeps the classifier labels of truncated outputs, and the audit flags them for the labelers.

### 6.8 Model audit of the classifier

Section 4.5 describes the audit design and the labelers. Each study's sample has 120 outputs, 60 from classifier positives and 60 from classifier negatives. Rates are design-weighted to the full set of generations.

Table: Model audit of the classifier on 120 sampled outputs per study. Precision is the weighted share of classifier positives that the labeler also called harmful. False omission is the weighted share of classifier negatives that the labeler called harmful. The frozen criterion asks for precision of at least 0.80, false omission of at most 0.05, and agreement between labelers of at least 0.90.

| Study | Labeler | Positives confirmed | Precision | False omission | Recall | Labeler agreement |
| :---------- | :------------------ | ----------: | --------: | --------: | ------: | ----------: |
| Qwen | Claude Opus 5.5 | 28 of 60 | 0.47 | 0.02 | 0.87 | 0.96 |
| Qwen | Claude Sonnet 5.5 | 33 of 60 | 0.55 | 0.02 | 0.89 | 0.96 |
| Granite | Claude Opus 5.5 | 46 of 60 | 0.77 | 0.02 | 0.96 | 0.95 |
| Granite | Claude Fable 5.2 | 46 of 60 | 0.77 | 0.02 | 0.96 | 0.95 |

Under the frozen thresholds the classifier is not supported in either study: precision is far below 0.80 for Qwen and just below it for Granite. In both, the failure is precision. The classifier rarely misses an output the labelers call harmful: each labeler marked 1 of 60 sampled negatives. It often credits outputs the labelers reject. The two labelers of each study agree with each other on 95% or more of outputs, so the gap is between the labelers and the classifier, not between labelers.

The size of the overcount differs by study. The classifier marks 19.8% of all Qwen generations positive, and the labelers' weighted estimates are 10.6% and 12.2%. For Granite the figures are 33.8% against 27.0% for both labelers. The labelers reported that their hard cases were mostly replies that take up the request and then stay vague or contradict themselves. We did not inspect the disputed outputs one by one. The pattern matches the failure Souly et al. describe for automated judges [@souly2024strongreject].

The overcount is not even across checkpoints and conditions, which matters more than its size because every decision is a contrast. Among sampled Qwen classifier positives, the labelers confirmed 19 and 21 of 33 under the GCG suffix, 5 and 8 of 17 under the mismatched suffix, and 2 of 7 under the bare prompt. By checkpoint they confirmed 12 and 14 of 18 on the source, 6 of 12 on Qwen2.5-Coder, 7 and 9 of 13 on Arch-Router, and none of 11 on Qwen2.5-Math.

Weighting those counts within each condition gives approximate 95% intervals. In the Qwen study the labelers' precision is 0.58 (0.41 to 0.74) and 0.64 (0.47 to 0.80) under the matched suffix, 0.29 (0.08 to 0.51) and 0.47 (0.23 to 0.71) under the mismatched suffix, and 0.29 (0.00 to 0.62) under the bare request. Only the matched-suffix condition holds the 30 sampled positives that the code requires for a usable per-condition correction. No Granite condition does: they hold 22, 15, 15 and 8. The intervals overlap, so the ordering by condition is a signal and not a finding.

If the ordering holds, false positives are more common under the control conditions than under the matched suffix, the contrasts in Table 5 survive better than the raw rates do, and the mismatched-suffix effect of Section 6.4 is smaller than the classifier shows. This audit cannot settle it. The sweep and Gemma audits were drawn by classifier label and condition for that reason, but with 120 outputs they also hold too few positives per condition to settle it (Sections 7 and 8).

Truncation (Section 6.7) belongs to the same validity problem. A reply that runs out of tokens after taking up the request is the shape the labelers called their hard case, so the labeler disagreement and the truncation counts should be read together.

What the audit supports is narrow. The direction of the source results is likely sound: the labelers confirmed most sampled positives on both sources. The classifier-based decision for Qwen2.5-Coder stands as a statement about classifier labels, with half of its sampled positives confirmed. No rate in Table 4 should be read as a harmful-compliance rate.

### 6.9 Robustness to the judge

This section covers the first two studies; Sections 7 and 8 report the same check for the later two. The audit asks whether independent labelers accept the classifier's positives. A separate question is whether the decisions depend on the particular classifier. We re-judged all 10,800 generations with HarmBench's official classifier, `cais/HarmBench-Llama-2-13b-cls` at revision `bda7053`, with its own published template. It is a different model family from the frozen Mistral judge. This is an exploratory check; the frozen decisions use the Mistral labels.

The two classifiers agree on 94% of outputs in both studies. The 13B classifier confirms 84% of the Mistral classifier's positives in the Qwen study and 94% in the Granite study, and flags 3% and 6% of its negatives. One Qwen2.5-Coder output exceeded the 13B classifier's 4,096-token context and is unscored; the contrasts below hold with that label set to either value.

Table: Held-out contrasts under the 13B HarmBench classifier, with 95% bootstrap intervals. Compare Table 5, which uses the frozen Mistral classifier.

| Target | GCG minus bare | GCG minus random | GCG minus mismatched |
| :--- | :--- | :--- | :--- |
| Qwen2.5-Coder | 0.37 (0.21, 0.51) | 0.32 (0.16, 0.47) | 0.06 (-0.09, 0.22) |
| Qwen2.5-Math | 0.01 (-0.06, 0.09) | 0.09 (0.02, 0.17) | 0.01 (-0.10, 0.11) |
| TinySwallow | 0.05 (-0.03, 0.15) | 0.05 (-0.01, 0.14) | 0.03 (-0.03, 0.12) |
| Arch-Router | 0.63 (0.48, 0.77) | 0.61 (0.45, 0.76) | 0.28 (0.13, 0.44) |
| Granite hardened | 0.04 (0.00, 0.13) | -0.03 (-0.10, 0.02) | 0.04 (0.00, 0.13) |
| Granite coder | 0.10 (-0.04, 0.25) | 0.00 (-0.15, 0.16) | 0.07 (-0.09, 0.22) |
| Granite abliterated | 0.11 (-0.07, 0.29) | 0.08 (-0.09, 0.26) | 0.01 (-0.13, 0.15) |

Every qualitative conclusion of Sections 6.3 and 6.4 reproduces. In the nine-test family of the behavior-specific label, Qwen2.5-Coder beats the bare request (Holm-adjusted $p = 0.0004$) and the random suffix (at most 0.003) and does not beat the mismatched suffix (1.0). No other primary target beats the bare request.

The two HarmBench classifiers therefore agree with each other far more than either agrees with the model labelers of Section 6.8. The labelers apply a stricter reading of "unambiguous and non-minimal", and the classifiers credit vague or partial compliance. Which reading a defender wants is a policy choice, not a measurement error, but it decides what the rates mean.

A second check corrects each condition's classifier-positive rate with that condition's audited precision, pooled over checkpoints. On Qwen2.5-Coder the GCG-minus-bare contrast falls from 0.35 to 0.24, and to 0.12 if the precision of the GCG cells is set to the lower end of its interval and that of the bare cells to the upper end. Against the random suffix it falls from 0.29 to 0.17, and to 0.03 in the same worst case (`experiments/results/precision_sensitivity.json`). The positive result survives the overcount in direction, with little margin against the random suffix.

## 7. A sweep of fourteen derivatives and the similarity test

The sweep measures fourteen further fine-tunes of Qwen2.5-1.5B-Instruct with the first Qwen study's frozen source attacks (Section 4.9). Its source rows repeat that study's source measurement, so they are not repeated here.

Table: Held-out results on the fourteen sweep targets, ordered by benign-probe similarity to the source (CKA without position 0, middle-layer mean). Rates are classifier-positive rates (each over 150 generations). Contrasts have 95% bootstrap intervals. Targets are named by a short label; Table 3 gives full repository IDs. Mism. is the mismatched suffix. Random-suffix rates are 0.13 or less and are in the aggregate file. Holm p is the largest of the three Holm-adjusted p-values, over the 42-test family; a target is supported when it is below 0.05 and all three contrasts are positive.

| Target | CKA | Bare | Mism. | GCG | GCG minus bare | GCG minus mism. | Holm p |
| :------------------ | -----: | ----: | ----: | ----: | :----------------- | :----------------- | -----: |
| sika-gpt | 1.0000 | 0.01 | 0.42 | 0.75 | 0.74 (0.61, 0.85) | 0.33 (0.20, 0.46) | 0.0001 |
| Nergis | 0.9999 | 0.01 | 0.30 | 0.61 | 0.61 (0.47, 0.74) | 0.31 (0.14, 0.49) | 0.011 |
| Begonvil-Ajan | 0.9998 | 0.02 | 0.41 | 0.73 | 0.71 (0.57, 0.84) | 0.31 (0.16, 0.47) | 0.008 |
| tooltuned | 0.9997 | 0.01 | 0.17 | 0.48 | 0.47 (0.33, 0.61) | 0.31 (0.17, 0.45) | 0.0005 |
| ClauseLM | 0.9996 | 0.02 | 0.20 | 0.37 | 0.35 (0.22, 0.47) | 0.17 (0.03, 0.30) | 0.13 |
| gsm8k | 0.9996 | 0.12 | 0.51 | 0.75 | 0.63 (0.45, 0.78) | 0.23 (0.11, 0.37) | 0.009 |
| address-parser | 0.9995 | 0.03 | 0.44 | 0.71 | 0.68 (0.55, 0.81) | 0.27 (0.12, 0.43) | 0.011 |
| SZL-Khipu | 0.9994 | 0.01 | 0.32 | 0.61 | 0.60 (0.47, 0.73) | 0.29 (0.11, 0.46) | 0.033 |
| TARS-SFT | 0.9992 | 0.15 | 0.13 | 0.20 | 0.05 (-0.07, 0.17) | 0.07 (-0.03, 0.17) | 1.00 |
| project-creation | 0.9987 | 0.01 | 0.29 | 0.41 | 0.39 (0.26, 0.53) | 0.12 (-0.02, 0.27) | 0.60 |
| glaive-tool | 0.9973 | 0.06 | 0.36 | 0.51 | 0.45 (0.31, 0.59) | 0.15 (-0.01, 0.32) | 0.60 |
| text-correction-zh | 0.9962 | 0.06 | 0.45 | 0.63 | 0.57 (0.42, 0.72) | 0.17 (0.04, 0.31) | 0.049 |
| hana-pskit | 0.9957 | 0.00 | 0.23 | 0.47 | 0.47 (0.33, 0.63) | 0.25 (0.10, 0.41) | 0.033 |
| TinySwallow-1.5B | 0.9747 | 0.05 | 0.05 | 0.07 | 0.02 (-0.06, 0.11) | 0.02 (-0.05, 0.10) | 1.00 |

![Held-out uplift of the source GCG suffix on the fourteen sweep targets, ordered by benign-probe similarity to the source. Filled markers are targets the frozen rule supports; hollow markers are not. The frozen test is a rank correlation, so the horizontal axis is the similarity rank; Table 9 gives the CKA values. All outcomes are classifier labels.](figures/sweep_similarity.png)

All 18,000 generations were recorded and judged, with no invalid label. The attack-adequacy and headroom gates passed for every target.

**Nine of fourteen targets show behavior-specific transfer.** Nine targets meet the frozen rule (Table 9, Figure 3): the source suffix beats the bare request, a random suffix and a suffix optimized for a different behavior. The matched-minus-mismatched contrast on those nine ranges from 0.17 to 0.33. Qwen2.5-Coder, the first Qwen study's one positive target, fails the same rule, so the sweep is the first evidence in this paper that a source-optimized suffix carries a request-specific effect to related checkpoints.

**One target is a copy of the source.** The sika-gpt checkpoint has CKA 1.000 at every layer, and its safetensors file has the same SHA-256 digest as the source's in the frozen plan. It is a re-upload under another name, and the roster rule, which reads names and configurations, did not catch it. Because generation is seeded per behavior, it reproduces the source's held-out rates exactly, which makes it an unplanned positive control for the pipeline. The frozen test counts it. Without it, eight of thirteen targets are supported, and the similarity results below are also given without it.

**The targets that resist.** Two targets gain nothing, and both were trained in ways that plausibly change refusal. TARS-SFT is the supervised stage of a safety method that trains the model to reason before answering [@tars2025]. It is a reasoning model, which the protocol meant to exclude; its id carries no reasoning marker, so the roster rule kept it, and it stays in the frozen roster. It already complies with 15% of bare requests and gains little from any suffix. TinySwallow-1.5B, a Japanese model distilled from Qwen2.5-32B-Instruct into the source architecture [@taid2025], is the least similar target. Three further targets beat the bare request and the random suffix but not the mismatched suffix. Two of them are fixed-output task models, one returning contract clauses as JSON and one writing student projects in a fixed format. The first Qwen study would have made them task controls; the roster rule has no such step, so they count as primary here.

**Similarity ranks uplift.** Across the fourteen targets the frozen test finds a positive rank correlation between similarity and uplift: Spearman $\rho = 0.61$ against the bare request (Holm $p = 0.024$) and $\rho = 0.77$ against the mismatched suffix (Holm $p = 0.0045$). Both meet the frozen rule. The second is the stronger result, because it is the contrast that isolates behavior-specific transfer.

Two sensitivity checks outside the frozen rule bound the claim (`similarity_sensitivity.json` in the sweep's results folder). Without the copy, $\rho$ is 0.52 against bare (Holm $p = 0.069$) and 0.71 against the mismatched suffix (Holm $p = 0.022$), so only the behavior-specific result survives. Dropping any other single target instead leaves $\rho$ at least as high, up to 0.73 against bare and 0.85 against the mismatched suffix. The range of similarity is also narrow: thirteen of the fourteen targets lie between 0.996 and 1.000. The ordering inside that range carries the result, so the measurement must be repeatable at the fourth decimal. It is repeatable for a fixed probe set and fixed weights (Section 8), but a different probe set may order the targets differently.

The 13B HarmBench classifier, re-judging all 18,000 outputs as an exploratory check, agrees with the frozen judge on 94% of them and supports exactly the same nine targets. Four outputs exceeded its context; every figure here holds with them set to either label. The similarity correlation is weaker under it: $\rho$ is 0.52 to 0.56 against bare (Holm $p$ 0.041 to 0.060) and 0.69 against the mismatched suffix (Holm $p$ 0.013 to 0.014). Without the copy the mismatched-suffix correlation is 0.61 (Holm $p$ 0.061 to 0.064). The direction and size of the association hold across judges; its significance at 0.05 rests on the copy and on the frozen judge.

The test shows an association across fourteen public derivatives of one source, under one attack and one probe set. Its permutation p-value treats the targets as exchangeable, although they share one source, one set of suffixes and the same thirty behaviors. It does not show that similarity causes transfer: a fine-tune that changes the model less is likely both more similar and less changed in its refusals, and the test cannot separate the two. It is the first result in this paper for the similarity half of the starting hypothesis.

**Prioritization.** The development split selected the loss proxy. Top-5 selection with it was supported on one of the nine supported targets, tooltuned (Holm-adjusted $p = 0.039$ over the fourteen targets), and not on the other eight. Within a target, the source scores still do not reliably say which behaviors will transfer best.

**Audit.** The sweep's audit sample was drawn by classifier label and condition, 30 outputs per condition, which gave 27 classifier positives. The two model labelers agreed on 97% of the sample ($\kappa = 0.83$), marked none of the 93 sampled negatives harmful, and confirmed 14 and 12 of the 27 positives, a design-weighted precision of 0.51 and 0.43. The classifier is *not supported* under the frozen criterion, as in the Qwen study. Under the GCG suffix both labelers confirmed 8 of 16 sampled positives, and under the mismatched suffix 6 and 4 of 9.

## 8. A third source: Gemma 3 1B

The Gemma study runs the full protocol on gemma-3-1b-it with its own search, 40 development and 40 held-out behaviors, and the stricter rule of the sweep: a target is supported only if the source suffix beats all three controls.

Table: Held-out results in the Gemma study (each rate is over 200 generations). Mism. is the mismatched suffix. CKA is benign-probe similarity to the source, computed after outcomes as an exploratory measurement. A target is supported only if all three Holm-adjusted p-values, over the 15-test family, are below 0.05 with positive contrasts. No target comes close, so Holm p here is the smallest of the three; on Minmini-1B it belongs to a negative contrast.

| Checkpoint | CKA | Bare | Mism. | GCG | GCG minus bare | GCG minus mism. | Holm p |
| :---------------- | ----: | ----: | ----: | ----: | :---------------------- | :-------------------- | ----: |
| gemma-3-1b-it (source) | n/a | 0.07 | 0.07 | 0.47 | 0.41 (0.28, 0.54) | 0.41 (0.28, 0.53) | n/a |
| borealis-1b | 0.947 | 0.00 | 0.03 | 0.04 | 0.04 (0.01, 0.09) | 0.01 (-0.04, 0.07) | 0.41 |
| borealis-open-1b | 0.942 | 0.01 | 0.06 | 0.07 | 0.06 (0.01, 0.12) | 0.01 (-0.07, 0.07) | 0.08 |
| BanglaGPT-Pro | 0.919 | 0.32 | 0.16 | 0.24 | -0.07 (-0.20, 0.04) | 0.08 (-0.03, 0.18) | 1.00 |
| LEM-Gemma3-1B | 0.900 | 0.26 | 0.19 | 0.29 | 0.04 (-0.07, 0.14) | 0.10 (0.00, 0.21) | 0.41 |
| Minmini-1B | 0.853 | 0.32 | 0.15 | 0.20 | -0.12 (-0.23, -0.01) | 0.04 (-0.06, 0.14) | 0.41 |
| paraphrase (task control) | 0.901 | 0.61 | 0.52 | 0.51 | -0.11 (-0.23, 0.01) | -0.01 (-0.12, 0.10) | n/a |

All 11,200 generations were recorded and judged, with no invalid label. Search took 7.8 GPU hours. The gap between the loss recomputed on the evaluated token IDs and nanoGCG's own (Section 6.1) exceeded 0.001 for 71 of 80 behaviors, with a median absolute size of 0.009 and a maximum of 0.09, the largest of the four studies. The recomputed loss was the higher one for 57.

**The attack works on the source and is specific to the request.** The source's development contrast passed the adequacy rule (0.47, $p < 10^{-8}$). On held-out behaviors the suffix raised the classifier-positive rate from 0.07 to 0.47. A suffix optimized for another behavior gave 0.07, the same as the bare request. On this source, unlike Qwen, the whole effect is behavior-specific.

**No child inherits it.** None of the five primary children is supported, and no single contrast reaches Holm $p < 0.05$. The largest point estimates are 0.06 over the bare request on borealis-open-1b and 0.10 over the mismatched suffix on LEM-Gemma3-1B. Two children are classifier-positive on about a third of bare requests and on fewer under the suffix. Every target reproduced the optimized source token IDs for every behavior, so re-tokenization does not explain the null. The paraphrasing task control, outside the primary family, also failed the headroom rule. Because no target is supported, prioritization was not tested.

**The children are further from the source.** The Gemma plan froze no similarity measurement. After outcomes we computed the sweep's measure for the six Gemma children, locally in float32 (`similarity_exploratory.json` in the Gemma results folder). The same script reproduces three of the sweep's frozen values to four decimals. The Gemma children lie between 0.85 and 0.95, below every sweep target (lowest 0.975). CKA values from two architectures are not on a common scale, and five targets with no transfer cannot test a correlation, so this observation is consistent with Section 7 and is not evidence for it.

**Audit.** The sample, stratified by classifier label and condition, held 23 classifier positives. The labelers confirmed 3 and 6 of them, a design-weighted precision of 0.13 and 0.26, the lowest of the four studies. The classifier marks 19% of all Gemma generations positive; the labelers' weighted estimates are 3% and 5%. The labelers agreed with each other on 95% of the sample, with a low kappa (0.37) because almost every item is negative. Their notes name two recurring classifier errors: refusals written in Norwegian, and replies that only restate the request as a question. The paraphrasing model restates every request, and neither labeler confirmed any of its 9 sampled positives. The classifier is *not supported*, and the target rates in Table 10 are mostly classifier error. The null result does not rest on those rates: no contrast is positive under either classifier, and an error that is equally frequent under every condition cannot create one. The audit says little about the source effect, because only one sampled positive came from the source; both labelers confirmed it.

**Second classifier.** The 13B HarmBench classifier agrees with the frozen judge on 91% of the outputs. Thirty-four outputs exceeded its context. On held-out behaviors they change only the source's rates: with them set to either label, the source effect is 0.40 to 0.42 over the bare request and 0.38 to 0.39 over the mismatched suffix. No child has a positive contrast at Holm $p < 0.05$. The 13B classifier also marks 36% of BanglaGPT-Pro's and 26% of LEM-Gemma3-1B's bare replies positive, close to the frozen judge's rates, so the two HarmBench classifiers share the errors the labelers reject.

## 9. Across the four studies

Table: Behavior-specific transfer by study. Eligible targets are primary and pass the headroom rule. Supported means the source suffix beat the bare request, a random suffix and a mismatched suffix, Holm-corrected within the study; this is the frozen rule in the sweep and Gemma studies and an exploratory label in the first two. Precision is the labelers' design-weighted precision for the classifier.

| Study | Targets and relation to the source | Eligible | Supported | Labeler precision | Agreement with 13B classifier |
| :---------- | :-------------------------------------- | -----: | -----: | :---------- | -----: |
| Qwen | 2 siblings, 1 distilled child | 3 | 0 | 0.47, 0.55 | 0.94 |
| Granite | Safety-hardened and coder children | 2 | 0 | 0.77, 0.77 | 0.94 |
| Qwen sweep | 13 fine-tunes and 1 copy of the source | 14 | 9 | 0.51, 0.43 | 0.94 |
| Gemma | 5 language or value adaptations | 5 | 0 | 0.13, 0.26 | 0.91 |

Three patterns hold across the studies. First, the attack worked on every source, and behavior-specific transfer appeared only on direct fine-tunes of the Qwen source. Of the 24 eligible targets, the nine supported ones are a copy of the source and eight assistants or task models, two in Turkish and one in Chinese. The fifteen unsupported ones are both siblings, both distilled models, both safety-oriented models (Granite hardened and TARS-SFT), the Granite coder child, all five Gemma children and three task fine-tunes. Second, declared lineage does not predict transfer, and neither does the kind of training a model card describes. Language does not separate the two groups, and the cards are too thin to say how much each model was trained. Third, measured similarity separates them better. The sweep's ranking by similarity and the Gemma children's low similarity point the same way, but only the sweep tests it. Each source contributes one lineage and the rosters were chosen differently, so these patterns are descriptive across studies, not a pooled test.

## 10. Limitations

**Measurement validity.** Every rate is a classifier label, and the classifier fails the frozen precision criterion in all four studies. The labelers who found that are language models from one vendor, chosen after outcomes, and they can share errors with each other. A second HarmBench classifier reproduces every transfer decision, but it shares the classifier errors the labelers reject, so agreement between the two classifiers does not settle validity. The labels cover 120 outputs per study and cannot correct per-cell rates; in the Gemma study they include one positive from the source. Whether the overcount is larger under the control conditions than under the matched suffix, which would make the contrasts conservative, remains open: the per-condition intervals overlap in every study. Between 13% and 33% of a study's outputs were cut off at the token budget and keep their classifier label (Section 6.7).

**The similarity result.** It is one frozen test, on fourteen derivatives of one source, with one probe set and one attack. Its significance at 0.05 depends on a target that proved to be a copy of the source and on the frozen judge: without the copy and under the second classifier, the correlation against the mismatched suffix is 0.61 with Holm $p$ near 0.06. Thirteen of the fourteen similarity values lie between 0.996 and 1.000, so the result rests on small differences that a different probe set might reorder. The targets share a source, suffixes and behaviors, so they are not independent draws. The test shows association, not cause.

**Rosters.** The first two rosters were hand-picked and the later two chosen by a rule that ranks by downloads. The rule missed a re-upload of the source and a safety reasoning model, and it has no step that sets aside fixed-output task models, which the first Qwen study treated as task controls. All three stay in the frozen roster and are reported. In the first Qwen study the roster set aside Arch-Router, which showed the largest uplift; counting it as a primary target would change the frozen count there from one in five to two in six (Section 6.3).

**Scope.** Three source lineages of 1 to 3 billion parameters, 27 target entries, one attack and one benchmark. The same behaviors recur across checkpoints, so effects are estimated per target and not pooled. Lineage and training descriptions come from model cards, which are often thin.

**Rule changes between studies.** The later two plans made the mismatched control primary after the first two studies showed why it matters. Within each study the rules were frozen before outcomes, but the choice of rules for the later studies was informed by earlier results.

**Prioritization estimand.** Frozen utility is computed on GCG-minus-bare uplift, so a score can earn it by finding requests a target answers under any perturbation. Section 6.5 reports the same selection scored against the other controls.

**Exploratory analyses.** The behavior-specific label in the first two studies, the control-relative utilities, the per-condition audit estimates, the second-classifier checks, the similarity sensitivity checks and the Gemma similarity values are not part of any frozen rule.

**Environment.** nanoGCG 0.3.0 ran with transformers 5.22.2, outside its declared range, in bfloat16. The loss recomputed on the evaluated IDs differed from nanoGCG's by up to 0.09 with an upward bias; float32 arithmetic halves the gap in the first two studies but does not remove it (Section 6.1). The attack moved every source, so the search worked, but the suffixes may differ from those a supported environment would find.

**Power and null results.** The design detects only large uplifts (Section 4.8). A null result on a target does not show robustness to other prompts or stronger search, and a high attacked rate does not show transfer unless it exceeds the controls.

## 11. Ethics and reproducibility

Only public research checkpoints and established benchmark inputs are used, run on infrastructure the author controls under schema-validated authorization records. No closed API or third-party service was queried. The attack and classifiers are published prior work, and no new harmful prompts were written. Exact completions and optimized suffixes are not published. During the runs they passed through one private dataset on the author's account, which is deleted after the final analysis. The model labelers read the sampled outputs inside the author's own session and returned labels and short notes only. The paper and repository contain aggregate measurements and reproducibility metadata, not a ready-to-use set of attack strings.

Anyone holding the evidence can regenerate every rate, contrast, p-value and count in Sections 5 to 9 from the study artifacts and check that the frozen rules were applied as printed (Appendix B). Three things cannot be checked from the public repository alone. Raw completions, suffixes and audit sheets are withheld, so a reader can re-apply the rules but cannot re-judge the outputs. The order in which plans were pushed and searches started rests on the hosting service's push log and the compute host's job records. And the audit labels come from the model labelers named in Section 4.5. One sweep checkpoint, SZL-Khipu-1.5B, has been removed from the Hugging Face Hub since the study; its weight digest remains in the frozen plan.

Benign synthetic regression tests cover the protocol's safeguards and statistics: authorization, immutable plans, exact prompt construction, resumed generation, search reuse, the development rules, the tests and corrections, audit weighting and the pushed-commitment check. Passing them establishes software behavior, not attack effectiveness or judge validity. The studies ran on rented GPUs. The similarity measurements in Appendix A and the exploratory Gemma similarity values were computed on a local Apple Silicon machine.

## 12. Conclusion

Does a jailbreak developed on an open-weight model follow its lineage? In four studies under one frozen protocol, a GCG suffix optimized on the source worked on all three sources and carried a request-specific effect to 9 of 24 eligible derivatives. All nine were fine-tunes of the Qwen source, or a copy of it. Siblings, distilled models, safety-tuned models and every Gemma child inherited nothing beyond what controls explain. In the first Qwen study most apparent transfer was a generic effect of adversarial-looking text, which only a suffix optimized for a different request exposes. A decision rule without that control would have reported transfer where there was none.

Across fourteen derivatives of one source, benign-probe similarity ranked how much transferred, in the one pre-specified test the design allowed. The association has the same direction and a similar size under two classifiers, and stays at 0.71 or above against the mismatched suffix when any one target is dropped. Its significance depends on the frozen judge and on a target that proved to be a copy of the source. It is evidence for the similarity half of the starting hypothesis, from one lineage, and it needs replication on other sources and probe sets. Source-only scores, by contrast, rarely picked the behaviors that transfer best within a target.

All of this is measured by classifier labels that a model audit does not support: the labelers confirmed between 13% and 77% of sampled positives, and two HarmBench classifiers share errors the labelers reject. The transfer decisions survive a change of classifier; the rates do not survive as harmful-compliance rates. For a defender the practical reading is narrow. Test the light fine-tunes of a source first, measure their similarity to it, always include a mismatched-suffix control, and validate the judge on the outputs it is scoring.

## Appendix A. Representation similarity and the first token position

We compared each of five source models with its targets on 240 template-generated benign probes of three to seven words, fourteen source-target pairs in four lineages (Table 13). Hidden states were mean-pooled over every non-padding token at each layer. For column-centered matrices $X$ and $Y$, linear CKA is $\|X^\top Y\|_F^2/(\|X^\top X\|_F\|Y^\top Y\|_F)$ [@kornblith2019cka]. Figure 4 shows the profiles.

![Benign-probe CKA profiles for fourteen source-target pairs in five source groups and four lineages, mean-pooled over all tokens. The near-1.0 middle layers largely reflect the first token position (Table 12). No attack outcomes are plotted.](figures/benign_profiles.png)

Under causal attention the first position does not see the rest of the probe, and it carries an outsized state. For three pairs we repeated the measurement pooling with and without position 0 (`experiments/results/cka_first_position_diagnostic.json`). The all-token means reproduce the profiles exactly. In the middle layers, position 0 supplies 99.5% of the pooled vector's norm in both lineages.

Table: Mean linear CKA over layers using all tokens or without position 0 (no pos. 0), and the range over middle layers (all hidden-state outputs except the first two and last two). Each source is the Instruct checkpoint. Llama 3.2 prepends a BOS token; Qwen2.5 does not.

| Source, target | All | No pos. 0 | Middle, all | Middle, no pos. 0 |
| :--- | ---: | ---: | ---: | ---: |
| Qwen2.5, Coder | 0.967 | 0.966 | 0.882–0.978 | 0.956–0.982 |
| Qwen2.5, TinySwallow | 0.982 | 0.975 | 0.972–0.993 | 0.965–0.995 |
| Llama 3.2, Dolphin3.0 | 0.997 | 0.972 | 1.000–1.000 | 0.959–0.986 |

For the BOS model, the middle-layer CKA of 1.000 is an artifact: the probes differ there mainly through a shared first-position state scaled by probe length. Profiles pooled this way therefore describe the pooling method more than the checkpoints, and similarity measurements should exclude position 0. Section 7 tests whether the corrected measure predicts transfer across the derivatives of one source. `experiments/evidence_inventory.json` records the counts and ranges used here.

The following table identifies the pairs in Figure 4. Abbreviations map to full checkpoint IDs in `experiments/lineage.yaml`.

Table: Source-target pairs in the benign CKA figure.

| Source group | Reference checkpoint | Target checkpoint |
| :--- | :--- | :--- |
| Qwen2.5 1.5B | Qwen2.5-1.5B-Instruct | Qwen2.5-Coder-1.5B-Instruct |
| Qwen2.5 1.5B | Qwen2.5-1.5B-Instruct | Qwen2.5-Math-1.5B-Instruct |
| Qwen2.5 1.5B | Qwen2.5-1.5B-Instruct | TinySwallow-1.5B-Instruct |
| Qwen2.5 1.5B | Qwen2.5-1.5B-Instruct | DeepSeek-R1-Distill-Qwen-1.5B |
| Qwen2.5 0.5B | Qwen2.5-0.5B-Instruct | Qwen2.5-Coder-0.5B-Instruct |
| Qwen2.5 0.5B | Qwen2.5-0.5B-Instruct | Qwen2.5-0.5B |
| Llama 3.2 | Llama-3.2-1B-Instruct | Dolphin3.0-Llama3.2-1B |
| Llama 3.2 | Llama-3.2-1B-Instruct | Llama-3.2-1B |
| Llama 3.2 | Llama-3.2-1B-Instruct | tofu-Llama-3.2-1B-Instruct-full |
| SmolLM2 | SmolLM2-1.7B-Instruct | SmolLM2-1.7B-Instruct-heretic |
| SmolLM2 | SmolLM2-1.7B-Instruct | SmolLM2-1.7B |
| SmolLM2 | SmolLM2-1.7B-Instruct | smollm2-1.7b-alpaca-it |
| Gemma 2 | gemma-2-2b-it | dolphin-2.9.4-gemma2-2b |
| Gemma 2 | gemma-2-2b-it | gemma-2-2b |

```{=latex}
\FloatBarrier
```

## Appendix B. Reproduction

The commands below rebuild the Qwen study from a frozen plan. The Granite, sweep and Gemma studies use `experiments/study-granite41-v1.yaml`, `experiments/study-qwen15-sweep-v1.yaml` and `experiments/study-gemma3-v1.yaml` the same way; the sweep's run job copies the parent's search and adds the similarity stage. The `run` job executes search, generation, judging, audit sampling, and analysis. `docs/STUDY.md` gives the full execution protocol, and `docs/ADR.md` records the design decisions.

```sh
pip install -e ".[dev,models]"
pytest
scripts/launch_hf_job.sh init experiments/study-qwen15-v3.yaml <commit>
git add experiments/results/study-qwen15-v3/plan_commitment.json
git commit -m "Commit plan" && git push
scripts/launch_hf_job.sh run experiments/study-qwen15-v3.yaml <commit>
python scripts/hf_evidence.py pull <private dataset> \
  experiments/study-qwen15-v3.yaml
lasa study audit-score --config experiments/study-qwen15-v3.yaml \
  --labeler <model id> --labeler-kind model --sheet <sheet>
lasa study analyze --config experiments/study-qwen15-v3.yaml \
  --allow-analysis-revision
python paper/build_results.py
python -m lasa.power --out experiments/results/power_analysis.json
python -m lasa.cka_diagnostic \
  --out experiments/results/cka_first_position_diagnostic.json
python scripts/similarity_sensitivity.py reports/qwen15-sweep-v1 \
  --out experiments/results/study-qwen15-sweep-v1/similarity_sensitivity.json
```

```{=latex}
\FloatBarrier
```

## References

::: {#refs}
:::

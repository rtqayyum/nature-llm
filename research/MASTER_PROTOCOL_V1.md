---
title: Evidence Controlled Traffic Transfer
subtitle: Project master protocol and experiment plan
author: Tariq Qayyum
date: 2 November 2026
---

We will study whether learning a transferred traffic predictor for verifiability makes useful transfer identifiable with less target evidence. We will compare the proposed mechanism against strong local models, a shared-data adaptation of LLM-UTP, and the same Qwen model trained without the verification objective.

**Project decision.** Preserve the completed reference assessment, establish the shared causal evaluation pipeline, then evaluate the proposed method. All eight eligible public domains remain in scope. The proposed method has no experimental results yet.

**Publication target.** Nature Communications. The intended contribution is a measured, explanatory advance in traffic transfer under limited target supervision. Journal acceptance is an aspiration, not a consequence of choosing a language model or meeting our internal thresholds.

**Protocol status.** This master plan consolidates the frozen design of 1 November 2026, repository commit 40f8e2078a79085258caceb62957f1d55d3da9dd. Implementation admission and HPC preflight remain pending. Equations describe the proposed mechanism; expected improvements are hypotheses.

**How we will use this document.** Follow the numbered stage gates and experiment IDs. Record an amendment before changing scientific settings. Preserve failed and infeasible cases. Do not launch the full grid before the admission checks pass.

# Contents

1 Project goals and boundaries	3

2 The reference problem and solution	4

3 Our problem and proposed solution	5

4 Completed reference evidence	6

5 Related work and the defensible gap	7

6 Datasets and admission rules	8

7 Data processing decisions	9

8 Time splits and target evidence budgets	10

9 Architecture and training settings	11

10 Forecast and paired benefit mathematics	12

11 Sequential verification mathematics	13

12 Learning for verifiability	14

13 Hypotheses and acceptable outcomes	15

14 Foundation experiments	16

15 Main baseline and method experiments	17

16 Mechanism ablations and prior art	18

17 Generalization and stress experiments	19

18 Metrics and statistical analysis	20

19 Stage gates and decisions	21

20 HPC execution discipline	22

21 Progress artifacts and resource accounting	23

22 Scope control and immediate work order	24

23 Nature Communications paper goals	25

24 Primary references and project records	26

# 1 Project goals and boundaries

Our primary question is whether source-domain training can produce a traffic correction that improves forecasting and requires less target evidence to establish its benefit over a strong local model. A useful system must forecast every test origin, include selection and verification in the target-label budget, and account for its computing cost.

The scientific goals are to measure when transfer helps, identify how much target evidence is needed, explain failures across domains, and distinguish the training mechanism from ordinary expert selection. We will test these goals on road traffic, metro counts and urban trip demand using a single transparent evaluation contract.

The work proceeds in three tracks. The completed reference track records what the public release supports. The shared baseline track establishes a fair, index-causal comparison. The method track tests Evidence Controlled Traffic Transfer, abbreviated ECTT. Results and output paths remain separate across these tracks.

We will not describe the available-data assessment as full-paper reproduction. We will not treat Qwen substitution, larger parameter count, leakage correction or an existing confidence sequence as our invention. We will not claim protection against arbitrary future distribution shifts.

The final research package must contain executable code, pinned inputs and environments, per-run forecasts, decision and label-access ledgers, independent metrics, statistical comparisons, resource measurements and a claim ledger. Every main figure must be recoverable from saved artifacts. A failed hypothesis is a reportable result and triggers the specified scope decision rather than an undisclosed change of task.

# 2 The reference problem and solution

LLM-UTP addresses the difficulty of constructing a scalable traffic predictor that works across transportation modes, scenarios and temporal resolutions. The Nature Communications paper studies 11 datasets from 29 cities or areas and combines trend enhancement, spatiotemporal encoding and a language-model module [1]. Its main scientific emphasis is shared representation and broad traffic prediction.

Our reference assessment uses the pinned public release. The released constructor retains six GPT-2 blocks, while its forward path executes two. These implementation details and the observed parameter-update policy must be documented when interpreting our reference runs. Directory names and agreement with rounded table entries do not establish whether a checkpoint is S1, S2 or S3.

The release also has evaluation behaviors that matter for a new comparison: full-array scaling, supplied auxiliary features, split-route differences and dependence on companion forecast windows. The stage-one NYC bike and taxi payloads are identical. Three restricted full datasets are unavailable. These are limitations of the accessible evidence, not proof that every experiment in the paper is invalid.

Our own historical training evaluator had an inverse-scaling error for nonzero minima. We corrected it without changing the saved models or normalized forecasts. This correction belongs to our implementation record and must not be attributed to the authors.

We preserve faithful reference behavior to establish what it produces. For the new study, we apply the same causal preprocessing and budget rules to the reference architecture and our method. That shared adaptation is explicitly distinguished from the original published protocol.

# 3 Our problem and proposed solution

A predictor learned from other domains may help a new domain, but a harmful correction can be difficult to detect with few target labels. A competitive local predictor offers a practical fallback. The proposed research asks whether we can train transfer representations so that useful benefit becomes easier to verify, rather than attaching a selector only after ordinary forecasting training.

ECTT has four components. A source-trained numerical encoder produces a transfer candidate. A strong local predictor is selected using support data only. Source episodes train the transfer candidate with forecast error and a verification-margin objective. A sequential comparison on a frozen target verification prefix decides whether to deploy transfer or retain the local predictor.

The mechanism changes learning, not only deployment policy. The decisive control therefore uses the same backbone, source episodes, target labels and verifier with the verification objective removed. Ordinary learned gating, fixed-prefix selection and source-informed statistical controls test whether a standard selector explains the apparent gain.

| Research dimension | Reference emphasis | ECTT question |
|---|---|---|
| Shared learning | General traffic representation | Representation with verifiable target benefit |
| Target evidence | Training or fine-tuning protocol | Explicit support and verification budget |
| Deployment | Released predictor | Frozen transfer or strong local fallback |
| Key evidence | Forecast accuracy and scaling | Accuracy, evidence cost, activation and harm |

The expected outcome is an empirical account of when the mechanism succeeds or fails. A favorable result would improve complete-system accuracy and reduce evidence requirements. Neither outcome has been observed yet.

# 4 Completed reference evidence

The following are real HPC executions reported in the project record. They are distinct from local synthetic tests and from unimplemented ECTT experiments.

| Job or family | Completed work | Interpretation |
|---|---|---|
| 605458 | 37 assets verified, 5.494 GiB | Acquisition complete |
| 607564 | Eight checkpoint evaluations | Five domains match all three S1 rounding anchors |
| 607596 | Ten fixed release diagnostics | Input and split sensitivity recorded |
| 608217 | Eight checkpoint causality checks | Companion-window dependence observed |
| 608294 | Two GPU smoke fits, 60 updates | Training engineering check |
| 609006 | Six independent training fits | PEMS08 and Beijing, three seeds each |
| 610878 | Six saved-forecast rescoring tasks | Correct affine inverse applied |
| 610996 | Consolidated reproduction review | Accessible assessment closed with review |

Corrected independent-training aggregates are mean plus or minus sample standard deviation over seeds 11, 22 and 33. These retain released scaling and cross-window behavior.

| Dataset | MAE | RMSE | WMAPE percent |
|---|---:|---:|---:|
| PEMS08 | 9.823958 ± 0.004492 | 15.813550 ± 0.066658 | 4.272521 ± 0.002737 |
| Beijing | 21.902838 ± 0.541004 | 37.285433 ± 0.607335 | 8.263660 ± 0.207654 |

Checkpoint rounding matches occurred for PEMS03, PEMS08, Hangzhou, bike and taxi. PEMS04, Shanghai and Beijing retained discrepancies. These outcomes do not certify checkpoint strategy, exact S3 training, clean causal performance or full-paper reproduction. No ECTT training has run. Historical artifacts remain unchanged on the HPC.

# 5 Related work and the defensible gap

The targeted primary-source review is recorded in research/RELATED_WORK.md. It is not an exhaustive proof of novelty. LLM-UTP, OpenCity and CompactST already study shared traffic representations and transfer [1–4]. TESTAM and FlexST establish expert routing and selective sharing [2,5]. ST-FGDIR and MURECAST are close comparisons for forecast-gain routing and calibrated correction [6,7].

Sequential forecast comparison, confidence sequences and model confidence sets already exist [8–10]. Active risk control and transfer-informed betting address label efficiency and source-informed verification [11,12]. Meta two-sample testing demonstrates the broader idea of learning representations for statistical power [13]. Conformal transfer and language-model ablation research further limit broad firstness claims [14,15].

The candidate gap is narrower: source-episodic training of a traffic predictor specifically to achieve useful, early-verifiable gain over a strong frozen local comparator, evaluated against ordinary training with the identical verifier and full target-data accounting.

| Claim we may test | Evidence needed |
|---|---|
| Learning improves verifiability | Same-backbone, same-verifier objective ablation |
| Improvement has practical value | All-origin accuracy, activation and resource cost |
| Transfer generalizes | Target-domain exclusions and multiple target modes |
| Qwen pretraining contributes | Compact and equal-policy random controls |

We will not claim the first traffic LLM, first safe transfer model, first confidence gate, first label-efficient verification method or a new concentration theorem. The precise novelty claim remains conditional on continued review of the closest learning-for-certification literature. If a matching mechanism is located, amend the contribution before confirmatory results rather than concealing the overlap.

# 6 Datasets and admission rules

We will reuse all eight public processed arrays already acquired on the HPC. Canonical new-study values are channel zero of the pinned stage-two arrays. These are released processed data, not independently recovered sensor archives. Container hashes and exact row intervals are in config/ectt_protocol_v1.json.

| Domain | Nodes | Rows | Final test rows | Role |
|---|---:|---:|---|---|
| PEMS03 | 358 | 26208 | 20966 to 26207 | Confirmatory |
| PEMS04 | 307 | 16992 | 13593 to 16991 | Confirmatory |
| PEMS08 | 170 | 17856 | 14284 to 17855 | Development |
| Shanghai Metro | 288 | 6072 | 4857 to 6071 | Confirmatory |
| Beijing Metro | 276 | 1800 | 1440 to 1799 | Development |
| Hangzhou Metro | 80 | 1650 | 1320 to 1649 | Confirmatory |
| NYC bike | 64 | 8760 | 7008 to 8759 | Confirmatory |
| NYC taxi | 64 | 8760 | 7008 to 8759 | Confirmatory |

Preflight must verify channel semantics, finite nonnegative observed values, raw-channel hashes, duplicates, missingness and source-target exclusions. Different whole-container hashes do not prove different channel-zero data. Failed domains remain listed with a reason; they are not silently replaced.

Absolute timestamps, service gaps and calendar alignment remain unverified. Our primary claim is next-row forecasting under the documented ordered-row assumption. We will not invent weekdays, physical durations, road graphs or holiday labels. Restricted example arrays are excluded. Prior reference test aggregates have been inspected; the benchmarks are not pristine unseen datasets. The new mechanism results must remain unseen until this protocol and implementation are locked.

# 7 Data processing decisions

We will maintain two processing tracks. Faithful reproduction retains the author's source, supplied features, scaling and batching to document the accessible release. The new benchmark rebuilds a causal feature pipeline and applies it equally to every baseline and ECTT. Preprocessing is an experimental control; it is not the proposed methodological contribution.

| Item | Reference track | Shared new benchmark |
|---|---|---|
| Base data | Original registered release inputs | Stage-two channel zero |
| Auxiliary trend | Released second channel | Prefix mean inside each history |
| Scaling fit | Released full-array behavior | Source training or target support only |
| Inverse | Faithful author or labeled correction | Recorded affine inverse |
| Cross-origin dependence | Preserved and diagnosed | Forbidden |
| Computational tail | Original behavior | Every eligible origin scored |
| Missing values | Recorded release behavior | Explicit mask, no invented targets |

For each history, compute each prefix mean using only rows up to that point within the 12-row history. Include an observed/missing mask. Zero counts are observed values, not a missingness code. Nonfinite observed targets fail admission. Do not interpolate targets or fit transformations using verification/test values.

Fit one global min/max using the allowed fitting rows and all available nodes. Store both extrema. Apply the correct inverse after prediction; clip negative physical predictions to zero for every method. Do not clip held-out normalized inputs to the fitting range.

For fitting extrema m and M with M greater than m, normalization and physical-unit inversion are

$$z=(x-m)/(M-m),\qquad \widehat x=\max(0,m+(M-m)\widehat z).$$

For observed history positions j, the causal trend at position k is

$$g_k=\frac{\sum_{j=1}^{k}o_jx_j}{\sum_{j=1}^{k}o_j},$$

where the observation mask o is one for an observed value. A zero observed-prefix count must use the frozen missing-input rule and retain its mask.

The immutable canonical data manifest is shared across models. Model-specific loaders must emit identical target IDs, raw truths and access ledgers. Previous released target checkpoints cannot initialize a supposedly held-out target comparison.

# 8 Time splits and target evidence budgets

Let T be the number of rows. Source training uses raw rows before floor(0.6T), and source validation targets end at floor(0.7T). A held-out target is excluded from every source fit, validation set and episode. When either NYC domain is targeted, exclude both bike and taxi from sources.

The final target test begins at c = floor(0.8T). Every method uses 12 observed rows to predict the next row, with stride one and no dropped computational tail. Rolling observed histories are allowed; fitting or selector updates using test labels are prohibited in the primary track.

For positive budget q, reserve the last B pretest rows, where

$$B=\lfloor q\lfloor0.8T\rfloor\rfloor,\qquad q\in\{0.01,0.05,0.10\}.$$

The first floor(B/2) rows form support S; the remainder form verification E. Warmup is counted inside the block. At least 24 support rows and 12 verification origins are required. Beijing and Hangzhou are infeasible at 1 percent and retain persistence fallback without borrowing extra labels. The primary budget is 5 percent.

Support pays for scaling, adaptation and local-family selection. Verification pays for the deployment decision. Count unique row-node values and their roles. Ordinary input histories are shared covariates and are logged separately. Zero percent means no additional target supervision; its primary fallback is persistence.

Use seeds 11, 22 and 33. PEMS08 and Beijing are development targets; the six other domains form the confirmatory panel. All eight are reported. Development choices cannot access confirmatory test outcomes, and complete-target exclusions still apply during source training.

# 9 Architecture and training settings

The initial candidate uses the official Qwen2.5-1.5B base model, not its instruction-tuned version, pinned to revision 8faed761d45a263340a0528343f099c05c9a4323 [17]. All 28 decoder blocks remain active. Qwen is a testable component; its name and scale do not constitute the novelty.

For each node, partition its 12-row numerical history into four patches of three rows. Fuse a permutation-equivariant pooled spatial context from the same origin into each patch embedding. Qwen processes four temporal tokens with shared nodewise weights. No fifth spatial token, text serialization, autoregressive text generation or attention across forecast origins is introduced.

| Setting | Initial contract |
|---|---|
| Source updates | Projection, pooling encoder, head and LoRA |
| LoRA | Rank 8, alpha 16, dropout 0, q_proj and v_proj |
| Other backbone parameters | Frozen |
| Source optimizer | AdamW, learning rate 0.0001, weight decay 0.01 |
| Source stopping | At most 100 epochs, patience 10 |
| Gradient clipping | Norm 1 |
| Target adaptation | Fresh bottleneck-32 residual head |
| Target updates | 200, learning rate 0.001, support only |

Use a first-order source-episodic surrogate: stop gradients through the support-adaptation optimizer trajectory, then update source modules on verification/suffix objectives. The gamma-zero control uses the identical schedule. This is not exact differentiation through a long meta-learning inner loop.

Profile BF16 backbone execution with FP32 optimizer, head and saved predictions; use FP64 verification/statistics. Freeze the memory-safe microbatch and common effective batch before comparisons. No 7B grid precedes the 1.5B pilot.

# 10 Forecast and paired benefit mathematics

Let $x_i$ contain only information available when origin i is forecast. Let $L_i$ be the frozen local forecast, $T_i$ the frozen transferred forecast, and $y_i$ the next-row vector over N nodes. Define node-mean absolute error by

$$\ell(z_i,y_i)=\frac{1}{N}\sum_{v=1}^{N}|z_{iv}-y_{iv}|.$$

Select the local family with a chronological 70/30 split inside support, then refit on all support. Candidates include persistence, ridge and a competitive compact model. Define s as the greater of this selected family's support-validation MAE and one. This fixed error scale is separate from the input min/max scaler and is identical across selector comparisons.

$$D_i=\frac{\ell(T_i,y_i)-\ell(L_i,y_i)}{s},\qquad b_i=\frac{1}{Ns}\sum_{v=1}^{N}|T_{iv}-L_{iv}|.$$

Negative D means transfer is better. The reverse triangle inequality gives

$$|D_i|\le b_i.$$

The bound uses predictions available before the label. It does not require independent sensors or an upper bound on traffic counts. Large unnecessary corrections make verification harder. Useful aligned corrections can improve benefit relative to their disagreement envelope.

The deployed forecast is

$$\widehat y_i=L_i+a(T_i-L_i),\qquad a\in\{0,1\}.$$

The primary policy freezes a before final test. If evidence is insufficient, a is zero and a local forecast is still issued. Every test origin remains in the score. Acceptance does not certify each individual prediction; it reflects the stated prefix comparison. A learned continuous gate is a comparison policy, not an undisclosed component of ECTT.

# 11 Sequential verification mathematics

For the frozen target candidates, let the information available before $y_i$ be the filtration F at origin i. Assume $b_i$ is measurable at that time and feedback is the one-step matured label. Define conditional mean loss difference and predictable variance process as

$$\mu_i=\mathbb E[D_i\mid\mathcal F_{i-1}],\qquad V_n=\sum_{i=1}^{n}b_i^2.$$

Hoeffding's conditional moment bound and Gaussian mixing yield a standard two-sided, time-uniform confidence-sequence width [8,18]:

$$w_n=\frac{\sqrt{(V_n+\rho)\log((V_n+\rho)/(\rho\delta^2))}}{n}.$$

The upper comparison statistic is

$$U_n=\frac{1}{n}\sum_{i=1}^{n}D_i+w_n.$$

Lock rho at 0.01 and delta at 0.05 for one candidate pair per deployment stream. Accept transfer at the first n of at least 12 satisfying

$$U_n<-\eta,\qquad \eta=0.01.$$

Otherwise use the local predictor. Freeze candidates, scale and hashes before reading verification labels. Review the derivation and code independently before claiming implementation validity.

The confidence sequence concerns the average conditional paired risk over the observed verification prefix under these conditions. It does not protect an arbitrary later test block, every accepted origin, raw RMSE or WMAPE. Multiple candidate/budget streams do not jointly inherit a study-wide 95 percent guarantee. Additional simultaneous candidates require an explicit error allocation.

Temporal dependence is accommodated through the conditional formulation, not by pretending nodes are independent. A future-deployment theorem would require assumptions connecting verification and deployment risks. Multi-horizon delayed feedback needs a separate filtration argument and is outside this primary guarantee.

# 12 Learning for verifiability

Construct chronological pseudo-target episodes from eligible source-training rows. Each episode has support S, a verification prefix E and a later suffix Q. Fit the local comparator and support residual head on S, then issue candidate forecasts on E before using its labels. Real held-out target domains never contribute source-episode gradients.

For usable prefixes P drawn from 16, 32 and 64 origins, define a normalized suffix accuracy loss and a smooth verification penalty:

$$A_Q(\theta)=\frac{1}{|Q|}\sum_{i\in Q}\frac{\ell(T_{\theta,i},y_i)}{s}.$$

$$W_P(\theta)=\frac{t}{|P|}\sum_{n\in P}\log\left(1+\exp\left(\frac{\overline D_n+w_n+\eta}{t}\right)\right).$$

$$J(\theta)=A_Q(\theta)+\gamma W_P(\theta).$$

Lock gamma at 1, temperature t at 0.01 and eta at 0.01. Implement the softplus stably. The ordinary-training control sets gamma to zero with identical episodes, capacity and optimization allowance. Mean-only and envelope-only objectives test which component matters.

Because source-episode labels train theta, this margin is an optimization surrogate. It is not a certificate for those training episodes. Statistical interpretation begins only on later target verification data after the candidate and adaptation are frozen.

Deployment follows a fixed sequence: fit support transformations and candidates; freeze their hashes; issue both forecasts sequentially on E; update the comparison only after each label matures; record the first acceptance or censoring; freeze the decision; forecast every final-test origin.

The mechanism may collapse to the local model, sacrifice peaks or overfit source episodes. Activation, all-origin error, harm tails and matched objective controls are required to expose these failures. We will not deliberately weaken the local comparator to make transfer appear useful.

# 13 Hypotheses and acceptable outcomes

Expected benefits below are hypotheses. The numerical thresholds are internal research decisions, not predicted scores or Nature Communications requirements.

| Hypothesis | Primary test | Proceed criterion |
|---|---|---|
| H1 Learning improves forecasting | ECTT versus ordinary Qwen with identical verifier | At least 2 percent macro relative-MAE reduction and paired interval excluding zero |
| H2 Benefit needs less evidence | Restricted first-certificate fraction, including failures | Lower fraction with paired interval excluding zero; no increase in material test harm |
| H3 Learning adds beyond selection | Same candidates with learned gate, fixed prefix and standard verifier | Advantage survives fair selector controls |
| H4 Language initialization earns its cost | Pretrained Qwen, random-policy and compact controls | Incremental benefit justifies full compute cost |

Useful transfer activation must reach at least 20 percent of eligible confirmatory primary-budget runs. With six eligible domains and three seeds, this means at least four of 18 runs. Report the actual eligible denominator if admission changes it. Always falling back does not establish the contribution.

Where noninferiority is claimed, use a 1 percent relative-MAE margin and empirical paired intervals. Wide uncertainty means inconclusive, not success. Harm uses the fixed support error scale and is reported over all origins and accepted cases.

Proceed to broad claims only when the mechanism improves accuracy and evidence use without hiding failures or increasing harmful deployment. If H4 fails, retain the learning question with the compact backbone and remove the necessity-of-Qwen claim. If H1 or H2 fails, diagnose and amend before new confirmatory work. Never select a successful budget, seed or dataset after inspecting results and call it the original hypothesis.

# 14 Foundation experiments

Experiment IDs are permanent. M means required, C means conditional on the stated claim or metadata, O means optional, and Done means existing execution. A family can contain many runs. Every family retains failed and infeasible rows.

| ID | Status | Experiment and required artifact |
|---|---|---|
| F01 | Done | Consolidate reference evidence; preserve checkpoint and corrected-training comparison tables |
| D01 | M | Verify eight canonical data hashes, semantics, raw duplicates and missingness |
| D02 | M | Emit support, verification, source and test ledgers; assert every exclusion and budget |
| D03 | M | Test support-only transforms, affine inverse, zero counts and common truths |
| S01 | M | Independently review verifier derivation, filtration, width and decision implementation |
| S02 | M | Synthetic null/alternative and varying-envelope streams; record calibration and stopping behavior |
| P01 | M | Model causality, batch invariance and future-label mutation admission checks |
| P02 | M | One-GPU memory, timing, checkpoint and exact-resume engineering pilot |

Synthetic statistical streams test implementation behavior; they do not demonstrate traffic accuracy, guarantee validity under every process or substitute for the mathematical argument. Document the simulated data-generating processes and Monte Carlo uncertainty.

Prediction admission compares an origin issued alone and within different computational batches while its available history is fixed. A later-label mutation must not change its forecast. Repeat checks before training and on selected trained checkpoints.

No learning grid begins until data ledgers and invariance checks pass. If timestamps cannot be resolved, the ordered-row claim remains explicit. P02 profiles the proposed Qwen model on the actual allocated device; the previous GPT-2 smoke does not establish its memory or runtime. Recovery must restore optimizer, RNG and episode cursor without repeating completed updates.

# 15 Main baseline and method experiments

All new comparisons share canonical data, target origins, supervised-value budgets, source exclusions, seeds and physical-unit metrics. Existing released-checkpoint scores are historical context, not a matched performance baseline.

| ID | Status | Family and decisive purpose |
|---|---|---|
| B01 | M | Persistence on all eight domains; minimum forecast and zero-budget reference |
| B02 | M | Ridge and compact local models; establish competitive support-only fallback |
| B03 | M | LLM-UTP shared-data causal adaptation; train before comparing our mechanism |
| B04 | M | Ordinary Qwen source transfer and identical-verifier control |
| M01 | M | Full ECTT on all eight domains and seeds 11, 22, 33 |
| M02 | M | Zero, 1, 5 and 10 percent budget curves; preserve infeasibility |
| M03 | M | Local, unconditional transfer, learned gate and fixed-prefix policies on identical candidate forecasts |
| M04 | M | Evidence cost, activation, all-test harm and total resource comparison |

The primary confirmatory comparison is at 5 percent budget on PEMS03, PEMS04, Shanghai, Hangzhou, bike and taxi. Development results for PEMS08 and Beijing remain visible. Source models never train on their held-out target, even for development.

Shared-data LLM-UTP is an explicit adaptation of the reference architecture. It uses causal trends, support-only scaling and origin-independent computation. Document every difference from faithful code, trainability and initialization. Its results cannot be called exact published S3.

Use an equal development search allowance, capped at 12 configurations per family. Final effective batches, precision and selection rules must be frozen before confirmatory training. Support-only local selection and target adaptation cannot consume verification labels. Reuse saved candidate forecasts for policy-only experiments so rerunning a selector cannot silently change its candidates.

# 16 Mechanism ablations and prior art

Mechanism ablations require matched retraining. Policy comparisons reuse identical saved candidates. Report complete source-pretraining and adaptation costs for both.

| ID | Status | Experiment |
|---|---|---|
| A01 | M | Gamma-zero objective with the identical sequential verifier |
| A02 | M | Mean-benefit-only penalty; remove disagreement-envelope contribution |
| A03 | M | Envelope-only penalty; remove mean-benefit term |
| A04 | M | Compact nonlanguage backbone with matched learning objectives |
| A05 | M | Random Qwen initialization with identical frozen/trainable policy |
| A06 | M | Source-informed and standard sequential alternatives on the same candidates |
| C01 | M | TESTAM and CompactST official-code eligibility and matched comparison |
| C02 | M | ST-FGDIR and MURECAST eligibility and matched comparison |
| C03 | C | FlexST and OpenCity comparison if metadata and training provenance permit |

C01 and C02 require an eligibility decision for each named method. If official code, source exclusions or required metadata are unavailable, retain the reason and narrow the comparison claim. A named common-data adaptation is allowed only when its deviations are explicit. Published numerical tables never count as matched reruns.

Memory and calibration labels for residual-memory methods count inside the same total target budget. Unknown target overlap in pretrained traffic checkpoints prevents strict held-out claims. The random-Qwen control isolates initialization under the selected adaptation policy; it is not an optimized from-scratch training benchmark.

Additional high-capacity variants are optional until the core mechanism passes. If an eligible close method matches ECTT, report it as evidence against the claimed advance. Do not retain only distant baselines while omitting the strongest available utility-routing or calibrated-correction comparison.

# 17 Generalization and stress experiments

| ID | Status | Experiment |
|---|---|---|
| G01 | M | Leave-domain-out evaluation with complete-target and NYC pair exclusions |
| G02 | C | Leave-mode-out road, metro and NYC demand groups if enough eligible source data remain |
| R01 | M | Input-node outages at 10, 30 and 50 percent, targets unchanged |
| R02 | M | History outages of 1, 3 and 6 rows with common masks |
| R03 | M | Failure analysis by observed flow magnitude, disagreement, evidence length and domain |
| R04 | C | Matched-regime verification under synthetic outages if making a regime-adaptation claim |
| R05 | M | Resource Pareto comparison including pretraining, fitting, verification and inference |
| O01 | O | Multi-horizon prediction after timestamp and delayed-feedback contracts are verified |
| O02 | O | Prediction intervals with separately budgeted calibration and validity assumptions |
| O03 | O | Five-seed extension adding 44 and 55 to all primary arms and confirmatory domains |

Stress masks use seeds 101, 202 and 303, identical across methods. Thirty percent node loss is the main synthetic severity; other levels describe sensitivity. Missing input values carry a mask and use a source/support-frozen imputation rule. Targets remain untouched. These are simulated outages, not observed incidents.

A certificate obtained from clean verification data does not imply protection during an outage. Distinguish clean-prefix/outage-test evaluation from a matched-regime experiment. Row blocks cannot be called minutes or hours without verified time metadata.

The registry contains 35 families: 28 required, three conditional, three optional and one completed. Required external comparisons include mandatory eligibility assessment, not an obligation to fabricate unavailable inputs. Conditional and optional extensions do not justify delaying the main study indefinitely or changing primary outcomes after inspection.

# 18 Metrics and statistical analysis

Report MAE, RMSE and WMAPE on the same physical-unit truths and valid targets. New-study WMAPE includes absolute errors at observed zero truths in its numerator. It differs from the faithful author-mask calculation and must be labeled.

$$\operatorname{WMAPE}=100\frac{\sum_i\sum_v|\widehat y_{iv}-y_{iv}|}{\sum_i\sum_v|y_{iv}|}.$$

A zero denominator is undefined. The primary macro statistic divides each domain's all-test MAE by the same strong local fallback's MAE, then averages domains without weighting by node count:

$$R_m=\frac{1}{K}\sum_{d=1}^{K}\frac{\operatorname{MAE}_{m,d}}{\operatorname{MAE}_{L,d}}.$$

If fallback MAE is zero, retain the fold and report raw paired differences; do not insert an epsilon or silently drop it. Primary relative improvement is 100 times the comparator-minus-ECTT difference divided by comparator R.

If tau is the first certificate and $B_E$ is the available verification length, define restricted evidence fraction as

$$r=\frac{\min(\tau,B_E+1)}{B_E+1}.$$

Nonacceptance takes $B_E+1$ with a censoring flag. Report consumed values, activation and normalized excess MAE. Material harm means normalized test difference above 0.01; also report positive-part mean and 90th percentile.

Use 2000 paired row-block bootstrap replicates, block 144, sensitivity 72 and 288. Pair methods and seeds; do not treat nodes or training seeds as extra geographic subjects. Co-primary H1 and H2 use 97.5 percent marginal empirical intervals. These are conditional-on-panel empirical intervals, not exact city-population guarantees. Domains with fewer than eight blocks have unstable, descriptive uncertainty and must be labeled. Report per-domain effects alongside aggregates.

# 19 Stage gates and decisions

Each gate has a concrete output. An unresolved check keeps the dependent stage pending. Record the cause and recovery action rather than interpreting Slurm completion as scientific success.

| Gate | Required evidence | Decision |
|---|---|---|
| 0 Reference closure | Historical review and limitations preserved | Complete with review; exact full-paper reproduction unestablished |
| 1 Shared data | D01–D03, hashes, semantics, exclusions and access ledgers | Proceed only with eligible listed domains |
| 2 Statistical and model admission | S01–S02 and P01; derivation, code and invariance checks | Stop on invalid bound or future dependence |
| 3 Engineering pilot | P02, measured resources, exact resume and environment manifest | Freeze effective batch and run identities |
| 4 Fair baselines | B01–B04 on the shared contract, independent metrics | Fix defects before method comparison |
| 5 Mechanism test | M/A families, H1/H2 and useful activation criteria | Proceed, amend or conclude no supported gain |
| 6 Paper readiness | Prior-art controls, generalization, stress, cost and reproducibility package | Assess Nature Communications suitability |

The acceptable foundation result is a valid pipeline, not a favorable accuracy number. The acceptable mechanism result requires both practical utility and reduced evidence use with useful activation and transparent harm. Inconclusive uncertainty cannot be promoted to success.

If source-learning collapses toward the fallback, diagnose the objective using development data only. If Qwen adds no useful value, drop that necessity claim. A new objective, data split, source grouping or threshold requires a new version and a clearly separated confirmatory assessment. The original failed result remains reported.

Exact reference reproduction remains a documented limitation. The lack of restricted full archives does not prevent a correctly scoped public-eight study, provided its claims and provenance remain accurate.

# 20 HPC execution discipline

All work remains inside /home/tariq.qayyum/nature-llm, including environments, caches, logs, downloads, temporary files and checkpoints. GitHub is the source of code and protocol versions. Updates use verified HTTPS snapshots because Git is unavailable on the HPC. Never overwrite prior results or modify shared Anaconda.

Use CPU Slurm jobs for data admission, statistics and reporting. Use shortq only when its observed limits and measured runtime fit; otherwise longq, normally 23 hours. Select one task with appropriate threads for a threaded program; 40 tasks are not equivalent to one 40-thread process.

Training tasks request one GPU and five CPUs in gpuq. Use independent dataset/fold/seed jobs, at most four active initially. Request up to ten days only when justified and capped by the observed partition maximum. Profile first. Four GPUs in one job are not required. Use meaningful names such as ectt-data-check, ectt-base-p03-s11 and ectt-full-p03-s11.

Load required cluster modules, normally anaconda3, and install only in project-local environments. Run heavy work through srun. Serialize shared environment setup without locking an entire training run. Verify bash syntax, Python compilation, path containment and recovery behavior before publishing entrypoints.

Every eventual submission uses the user's saved notification address in the HPC project preferences. Keep that address outside version control. Request BEGIN, END, FAIL, REQUEUE and TIME_LIMIT with array-task events. Send the existing submission receipt for queue entry. Scheduler acceptance or local mail acceptance is not proof of inbox delivery.

Never cancel unidentified jobs or delete another project's files. This document starts no jobs. The older two-domain causal-train3 stage remains on hold and is not the new eight-domain benchmark.

# 21 Progress artifacts and resource accounting

Every run must save its code, configuration, data and environment hashes, source fold, target budget, seed, active/trainable parameter counts, allocated device and runtime. New ECTT artifacts use separate stores from reference outputs. Exact subdirectory contracts must be frozen with the new launcher before execution.

Checkpoint model, optimizer, RNG, cursor and scheduler state each epoch and at five-minute boundaries. Write progress atomically and retain completed prediction shards with checksums. Resume compatible contracts only; changed contracts create new identities. Do not overwrite partial results to simplify a retry.

Required artifacts include progress.json, metrics.json, training_history.csv, forecasts, decision_history.csv, access_ledger, runtime.json, failure reasons and a human-readable SUMMARY.md. Reports must show completed/expected runs and provisional seed aggregates during execution. Final aggregates require successful accounting and artifact validation.

| Resource item | What we must report |
|---|---|
| Training | All source and adaptation GPU hours, including failed runs |
| Memory | Peak CPU and GPU memory on the allocated hardware |
| Prediction | Latency, throughput and verifier overhead |
| Recovery | Reused versus newly executed updates and shards |

The two core Qwen objectives allow up to 48 source-fit instances for eight target folds and three seeds before reusing identical source contracts, including the paired NYC exclusions. Across the three positive budgets, they imply 132 eligible target adaptations after removing the two infeasible 1-percent domains. These are planning counts, not runtime estimates; baselines and ablations add work. Freeze the full dependency graph to avoid duplicate source fits. No total GPU-hour promise is made before P02 measurements.

Monitoring must expose queue state, accounting exit code, recent progress, CPU/memory use, failed logs, completed experiments and exact partial/final paths. Reports distinguish real HPC runs, synthetic tests and still-unverified behavior.

# 22 Scope control and immediate work order

The master protocol is the project checklist. The machine-readable configuration governs exact hashes, row intervals and numeric settings. The design and claim ledger govern interpretation. If these disagree, resolve the conflict in a versioned amendment before running the affected experiment.

The next work unit is shared-data preflight D01–D03. It reads existing HPC arrays, constructs identical model-independent ledgers and reports eligible domains. It does not download the data again or train the method. After that come statistical/model admission and the single-GPU engineering pilot, then the corrected shared-data baseline experiments.

Before confirmatory runs, freeze the implementation commit, precision, effective batch, source episode schedule, compact-model recipe, selector recipes, missing-input imputation and complete run matrix. These engineering details are admission work, not permission to tune the primary hypothesis on confirmatory outcomes.

Changes to objectives, datasets, budgets, exclusions, primary endpoints, thresholds or source information require a protocol-version increment. Record what changed, why, which results were already visible and whether a fresh confirmatory assessment is needed. Pure bug fixes also record affected runs and whether reruns are required.

No arbitrary model-size sweep, extra dataset substitution or result-driven change of metric enters the core plan. Optional experiments require their stated prerequisite and remain secondary. A five-seed extension adds the same seeds across every primary arm and confirmatory domain, regardless of significance.

The method is intended to improve the reference research direction through a distinct learning question. The paper can report reproducibility concerns with precise evidence, but an audit dispute is not the central contribution. This keeps the project focused on a defensible scientific result rather than repeatedly reopening settled engineering stages.

# 23 Nature Communications paper goals

Nature Communications considers novelty, conceptual or methodological advance, potential impact and interest to its readership [19,20]. Meeting our numerical gates alone is not sufficient. The study must explain a substantial result in traffic transfer that survives close prior-art comparisons and transparent limitations.

The intended paper asks when transfer can be made verifiable with limited target supervision. Its strongest possible conclusion would connect representation learning, disagreement and evidence cost, while showing useful deployment accuracy across eligible transportation domains. Qwen is one implementation used to test this question.

| Planned figure | Scientific purpose |
|---|---|
| 1 | Problem, source/support/verification/test protocol and mechanism |
| 2 | Same-verifier primary accuracy and target-budget curves |
| 3 | Restricted evidence cost, censoring and activation |
| 4 | Objective and backbone controls explaining the mechanism |
| 5 | Transfer failure, outage stress and limits of prefix evidence |
| 6 | Accuracy/evidence/compute trade-off and reproducibility summary |

The manuscript must separate reference findings, shared-protocol adaptations and proposed-method results. It must disclose prior benchmark exposure, unavailable restricted data, unresolved timestamps, simulation assumptions, ineligible comparisons and negative outcomes. Include code/data availability, a full run inventory and independent artifact-based metrics.

Publication readiness requires a clear mechanism beyond an ordinary router, meaningful matched gains, useful activation, interpretable limits and a feasible computational cost. If these fail, revise the scientific claim or pursue a narrower venue honestly. We will not invent a new theorem, assert universal safety or promise acceptance. The target remains Nature Communications, while the experiments determine what claims the evidence supports.

# 24 Primary references and project records

The related-work assessment is a targeted review with evidence cutoff 1 November 2026. Recheck the closest work before submission. Links identify primary sources; preprints and restricted retrieval depth remain labeled in the detailed review.

1. Zhang et al. LLM-UTP. Nature Communications 2026. https://doi.org/10.1038/s41467-026-73610-2
2. TESTAM. ICLR 2024. https://proceedings.iclr.cc/paper_files/paper/2024/hash/0cb310ed8121549488fea8e8c2056096-Abstract-Conference.html
3. OpenCity. https://arxiv.org/abs/2408.10269
4. CompactST. PVLDB 2025. https://www.vldb.org/pvldb/vol18/p2149-han.pdf
5. FlexST. Author preprint. https://arxiv.org/abs/2609.13878
6. ST-FGDIR. Sensors 2026. https://doi.org/10.3390/s26196319
7. MURECAST. Symmetry 2026. https://doi.org/10.3390/sym18091485
8. Choe and Ramdas. Comparing Sequential Forecasters. https://arxiv.org/abs/2110.00115
9. Arnold et al. Sequential model confidence sets. https://doi.org/10.1093/jrsssb/qkag066
10. Li and Zhu. CC-SMCS. Author preprint. https://arxiv.org/abs/2609.28522
11. Xu et al. Active anytime-valid RCPS. https://arxiv.org/abs/2406.10490
12. Basu. Transfer-Informed Betting. Author preprint. https://arxiv.org/abs/2603.08907
13. Liu et al. Meta Two-Sample Testing. NeurIPS 2021. https://proceedings.nips.cc/paper_files/paper/2021/hash/2e6d9c6052e99fcdfa61d9b9da273ca2-Abstract.html
14. Zhou et al. Conformal instance transfer. https://proceedings.mlr.press/v105/zhou19a.html
15. Tan et al. Language-model ablations. NeurIPS 2024. https://proceedings.nips.cc/paper_files/paper/2024/hash/6ed5bf446f59e2c6646d23058c86424b-Abstract-Conference.html
16. ShiftGuard-ITS. Repository-level contextual evidence only. https://github.com/shishirbit/ShiftGuard-ITS
17. Official Qwen2.5 base configuration. https://huggingface.co/Qwen/Qwen2.5-1.5B
18. Howard et al. Confidence sequences. https://doi.org/10.1214/20-AOS1991
19. Nature Communications editorial process. https://www.nature.com/ncomms/submit/editorial-process
20. Nature Communications aims and scope. https://www.nature.com/ncomms/aims

Project records: research/ECTT_V2_DESIGN.md, research/RELATED_WORK.md, research/CLAIM_LEDGER.md and config/ectt_protocol_v1.json in https://github.com/rtqayyum/nature-llm. This master protocol is research/MASTER_PROTOCOL_V1.md. Actual execution evidence remains in the project HPC reports.

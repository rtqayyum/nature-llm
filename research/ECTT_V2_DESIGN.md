# ECTT: verification-aware traffic transfer

Design date: 1 November 2026. Candidate contribution; no method results yet.
Machine-readable frozen evaluation design: config/ectt_protocol_v1.json.
This supersedes the methodological detail of the earlier ECTT proposal while
retaining its question about evidence-controlled transfer and Qwen2.5.

## Research decision

Proceed with **learning traffic representations for verifiable transfer**, with
Qwen2.5 as a candidate numerical backbone. The central result must explain
when sharing becomes useful and how much target evidence identifies that
benefit. Generic Qwen substitution or a post-hoc confidence gate is insufficient.

The available-data reference assessment is closed with documented limitations:
job 610996 verified eight checkpoint evaluations and six corrected independent
fits. It did not establish full-paper or exact S3 reproduction. Preserve all
those results as a separate reference track. Neither their better scores nor
their released data/batch behavior establish a clean comparison for our method.

## Problem

A shared model can predict a new domain well on average while making harmful
corrections at some origins. A local model may be less expressive but easier
to fit with a small amount of target data. A useful system must issue a forecast
at every origin, and must include the observations spent choosing transfer
in its label budget. We seek **useful transfer that becomes verifiable early**.

The scientific outcome is a map of target evidence, paired forecast benefit,
temporal variability and failed transfer across road, metro and demand domains.
This is a stronger goal than reporting a single average MAE improvement.

## Mechanism

Let L(x) be the strong local comparator and T_theta(x) the source-trained
transfer candidate after support-only adaptation. For every target origin issue

    forecast(x) = L(x) + a * (T_theta(x) - L(x)),  a in {0,1}.

The primary policy uses the transfer candidate only if the verification prefix
supports its benefit. Otherwise a=0 and the local forecast is still issued.
No difficult origin disappears from the score. A learned continuous gate and
fixed-prefix decision are comparison policies, not components silently added
to the proposed method.

### Numerical representation

Start with the official base Qwen2.5-1.5B revision
8faed761d45a263340a0528343f099c05c9a4323, all 28 decoder blocks active.
Use numerical projected tokens, not decimal strings, prompts or text generation.
For each node, split the 12 observed rows into four patches of three rows.
Append a permutation-equivariant pooled spatial context computed only from
the same origin's available node histories. Qwen processes the four temporal
tokens with shared nodewise weights; a numerical head predicts the next count.
There is no attention across separate forecast origins or arbitrary node IDs.

Source training updates the projection, pooling encoder, head and rank-8
q_proj/v_proj LoRA modules. The rest of the backbone is frozen. Target support
fits a fresh bottleneck-32 residual head; the shared encoder stays fixed.
Qwen is an experimental component, not the novelty claim. Compare an equally
trained compact transformer and the same Qwen capacity/objective controls.
The random-initialization control uses the same freeze/trainable policy and
therefore isolates initialization under that adaptation budget, not the best
possible from-scratch Qwen training regime.

### Train candidates for verification, not only forecast error

Build chronological pseudo-target episodes from eligible source training rows.
Each episode has support S, a verification prefix E and a later scored suffix Q.
Fit and freeze a competitive local model on S. Both forecasts are issued on E
before its labels are used. E labels can teach the source training objective;
Q is a later source-training suffix for the episode's system-utility loss.
Real held-out target domains never supply these source-episode gradients.
Use a first-order episodic surrogate: fit the support residual head, then stop
gradients through that optimizer trajectory while updating source modules on
the episode objective. The gamma=0 control uses the same episode schedule;
this is not exact differentiation through a long MAML inner loop.

For one origin define normalized paired absolute-error difference and its
forecast-visible bound:

    D_i = (MAE_nodes(T_i,y_i) - MAE_nodes(L_i,y_i)) / s
    b_i = mean_nodes(abs(T_i-L_i)) / s
    abs(D_i) <= b_i.

Here s=max(the selected local model's chronological support-validation MAE,1),
fixed before refitting and shared by all policies for that episode. This
statistical error scale is distinct from the input min/max normalization. The bound follows from the reverse triangle inequality. It does
not require bounded traffic counts or assume nodes are independent. Large
unnecessary departures from L make evidence noisier; useful aligned corrections
can yield a favorable benefit-to-disagreement ratio.

The proposed training term penalizes a differentiable upper verification
margin at source-episode prefixes n in {16,32,64} when those prefixes exist:

    J(theta) = mean_Q(MAE(T_theta)/s)
             + gamma * mean_n temperature * softplus((mean_E_n(D) + w_n + eta)/temperature).

Lock gamma=1, eta=0.01 and temperature=0.01 for the first version. w_n is the
same normal-mixture bound width used by the primary verification control.
An ordinary-forecasting objective (gamma=0), a mean-benefit-only penalty, and
an envelope-only penalty are required ablations. Same candidates are reused
across selector-only comparisons; training-objective ablations require matched
retraining, seeds, source data and architecture.

This objective is proposed here. It has not been implemented or shown to
improve target forecasting. It can collapse toward T=L, sacrifice rare peaks,
or overfit source episodes. Complete-system error, activation and tail harm
explicitly test these failure modes. Never train a weaker L to make gains easy.
Because source-episode labels train theta, the source training margin is only
an optimization surrogate, not an out-of-sample certificate. The statistical
interpretation applies to a later target verification prefix after all models
are frozen and only if its filtration and bounds meet the stated conditions.

### Verification and the exact scope of the certificate

On an actual target, fit both candidates using support only, then freeze model,
scaler and candidate hashes before reading the verification labels. Let mu_i
be the forecast-origin conditional expectation of D_i. For the one-step primary
task, D_i lies in the predictable interval [-b_i,b_i]. Hoeffding's conditional
mgf inequality gives a standard exponential supermartingale with variance
process V_n=sum_i b_i^2. Gaussian mixing gives the familiar two-sided width

    w_n = sqrt((V_n+rho)*log((V_n+rho)/(rho*delta^2))) / n.

Thus mean(D) +/- w_n is a time-uniform confidence sequence for
mean_i(mu_i), under the declared predictability and feedback filtration.
Lock rho=0.01 across matched methods before target verification; delta=0.05 for
one candidate pair per deployment stream. Use the first n>=12 with upper
bound below -0.01 to accept transfer. Freeze that decision before final test.

This is an instantiation of existing confidence-sequence machinery, not a
claimed new theorem. It needs an independent mathematical/code review before
any implementation is described as valid. Different candidate/budget runs
do not jointly inherit one study-wide 95% statement. Multiple candidates within
one decision require an explicit simultaneous error allocation.

Credit Howard et al. for mixture-bound foundations and Choe/Ramdas for
sequential forecast comparisons; related constrained/multi-horizon work is
listed in RELATED_WORK.md. Variance-sensitive or betting-based alternatives
are controls, not methods invented by this study.

Crucially, this certificate concerns the **observed prefix's average conditional
paired risk**. It does not guarantee every accepted origin, a future test block,
or performance after arbitrary drift. The final test's accuracy and harmful
transfer are empirical out-of-sample outcomes. A theorem about future deployment
would need additional assumptions relating verification and deployment risk.
Do not assume time-averaged past superiority makes the future safe.

Delayed multi-horizon verification is a secondary extension. The filtration
problem and stream-splitting solutions already appear in prior literature.
For now the confirmatory statistical task is one-step, immediately matured
feedback. Artificial selector-update delay is not a claim that a value already
visible in the rolling input history is hidden.

## Falsifiable hypotheses and stopping criteria

| Hypothesis | Decisive comparison | What would refute the contribution |
|---|---|---|
| H1: verification-aware training improves deployed forecasting | ECTT versus ordinary-training Qwen with the identical verifier, data and capacity | Gains disappear under the identical-verifier control or full-system error worsens |
| H2: benefit becomes identifiable with fewer target values | Restricted evidence-to-certificate counts, with every non-acceptance censored at budget+1 | Apparent speed comes from dropping failures, lower activation, more labels or larger harmful-transfer error |
| H3: the gain is a learning mechanism, beyond selecting an expert | ECTT versus learned gating, fixed-prefix selection and source-informed verifier controls | A standard selector matches the complete result at equal candidate/data budgets |
| H4: language pretraining contributes at a useful cost | Qwen pretrained/random policy controls and compact nonlanguage backbone | Extra gain is absent or dominated by compute; then use the compact mechanism and drop the necessity-of-Qwen claim |

The preregistered practical utility target is at least 2% macro error reduction
against ordinary-training Qwen plus the same verifier. This is a decision
criterion, not an expected result. Require useful transfer activation of at
least 20% across eligible confirmatory runs and report all no-acceptance cases.
Lower restricted evidence time must not increase harmful test-transfer or make
the system materially worse: 1% relative-MAE noninferiority margin, assessed
with empirical paired intervals. If the uncertainty is too wide, mark the claim
inconclusive. Always-fallback behavior does not satisfy the contribution.

## Shared frozen evaluation protocol

### Data and exposure

Reuse all eight public datasets: PEMS03/04/08, Beijing/Shanghai/Hangzhou Metro,
NYC bike and NYC taxi. Canonical inputs are channel zero of the pinned stage-two
arrays, whose exact container hashes are in the JSON. Rebuild a causal prefix
mean inside each history window; discard the supplied auxiliary channel for
every method in this new track. This avoids using the duplicated stage-one
bike/taxi release in shared pretraining. It does not retrospectively alter
faithful reproduction or prove raw-channel semantics.

Before execution, verify canonical raw-flow semantics, finite/nonnegative data,
raw-channel duplicates, source-target exclusions and missingness. Do not treat
different full-array hashes as proof of distinct channel-zero payloads. A failed
gate retains an ineligible domain with its reason; it is not silently replaced.

Absolute timestamp/service-gap metadata is unresolved. Primary results are
therefore indexed-row forecasting under a documented ordered-row assumption.
Do not invent weekday embeddings, calendar periods, physical road adjacency,
holiday experiments or elapsed-time continuity. Longer-horizon physical claims
are excluded until this metadata is verified.

Earlier checkpoint/test aggregates were inspected during reproduction. These
are not pristine unseen benchmark datasets. Freeze this new protocol before
any method-specific confirmatory results; disclose the prior exposure.

### Folds, time and labels

- History: 12 rows; target: next row; stride one; score every test origin,
  including the final incomplete computational batch.
- Source fitting: raw [0,floor(.6T)); source validation targets through
  floor(.7T). No source's final test labels enter pretraining or model selection.
- Target final test: [floor(.8T),T), identical across methods and budgets.
- Sparse target budgets: 0%,1%,5%,10%; primary 5%. For q>0, B=floor(q*floor(.8T))
  raw target rows are taken immediately before the test boundary. First floor(B/2)
  rows are support; the remainder are verification. Twelve-row history warmup
  is inside this block. Record every unique row-node value and its access role.
- Budget includes supervised fitting, normalization, selection and verification.
  Ordinary rolling input histories are shared covariates and logged separately.
  0% means no additional target supervision, not absence of observed histories.
- At least 24 support rows and 12 verification origins are needed for learned
  adaptation. Beijing and Hangzhou are infeasible at 1% in this design. Keep
  those rows as infeasible; use persistence without extra labels, never borrow
  labels or silently increase the budget.
- Fit a target scaler only on support; fit source scalers on source training.
  Frozen affine inversion is required; no held-out normalized clipping. Zero
  counts remain observed and distinct from missingness.
- Pick the local family using only a chronological 70/30 split inside support,
  with source-frozen recipes for persistence, ridge and compact forecasting;
  refit the selected family on all support. Never select it on verification/test.
- After support, models/scalers are frozen. Sequential prefix decisions may stop
  inside verification; every policy has the same available label budget and
  reports actual consumption. No test-label update occurs in the primary track.

Leave the complete target domain out of all traffic pretraining/episodes.
NYC bike and taxi are excluded together when either is the target, limiting
same-city cross-modal leakage. PEMS districts are regions, not assumed independent
cities. Secondary mode-out experiments exclude all road, all metro or all NYC
demand sources. Their small number does not establish universal modal transfer.

PEMS08 and Beijing are development targets; the six other domains form the
confirmatory target panel. All eight remain reported. Source eligibility and
target exclusions apply to every candidate, even for development folds.

### Baselines and what counts as fair

| Comparison family | Required control |
|---|---|
| Local quality | Persistence, ridge and a competitive compact graph-free model; frozen local-selection recipe |
| Reference architecture | LLM-UTP adaptation on exactly the new raw channel, causal trend, scaler, histories, origins and target-label budget |
| Backbone-only | Ordinary-training Qwen with identical capacity, initialization, source data and adaptation |
| Policy-only | Local, unconditional transfer, learned gate, fixed-prefix decision and existing sequential verifier applied to identical saved candidates |
| Learning mechanism | Full objective, gamma=0, mean-only term and envelope-only term, with identical verifier |
| Prior art | TESTAM, CompactST, FlexST/OpenCity, ST-FGDIR and MURECAST where official code/data requirements are satisfied |
| Language necessity | Compact nonlanguage transformer and equal-trainability random-Qwen control; complete resource accounting |

An off-the-shelf traffic checkpoint with unknown target pretraining overlap is
not strict held-out evidence. Retrain eligible official implementations on the
same source exclusions. If a method needs unavailable timestamps/graphs, report
its common-data adaptation explicitly or mark it ineligible by the preregistered
metadata rule. Never compare its published metric as if it were a matched run.
MURECAST-style memory, calibration labels and local-base fitting count inside
the same total target budget. Native published protocols can be secondary
contextual tables, separate from the shared-data benchmark.

The current two-domain causal-train3 implementation is not this eight-domain
stage-two benchmark. Do not launch it and label it the new comparison. The
new protocol needs its own preflight, loader/label ledger and run identity.

### Metrics, randomness and statistics

Seeds: 11,22,33 for all core arms. If a five-seed extension is executed, add
44 and 55 to every primary arm and all six confirmatory domains, irrespective
of significance. Seeds do not create extra geographic subjects.

Report raw MAE/RMSE/WMAPE in physical units on common valid targets. WMAPE is
100*sum(abs(error))/sum(abs(truth)); zero denominator is undefined. The primary
utility statistic is the macro average of each domain's all-test MAE divided
by its same strong local comparator's test MAE. Never average unscaled passenger,
vehicle and trip-count MAE. Require a positive fallback MAE denominator.

Report all-origin excess MAE, materially harmful transfer, its upper tail,
activation, actual target values consumed and restricted certificate time.
Unaccepted cases take evidence-budget+1 and retain a censored flag. They are
not removed from the average. All-test deployment utility and prefix evidence
are different endpoints.

Use 2000 paired row-block bootstrap replicates, primary block 144 rows and
sensitivity 72/288. Keep methods/seeds paired; preserve the benchmark-domain
panel instead of treating nodes as independent subjects. These are empirical
conditional-on-panel intervals, not exact finite-sample city-population inference.
Fewer than eight blocks yields an explicitly unstable/descriptive domain interval.
For two co-primary mechanism endpoints use 97.5% marginal empirical intervals;
this multiplicity adjustment does not make an invalid dependence model valid.

Synthetic stress: input-node outages at 10/30/50%, history blocks of 1/3/6 rows,
mask seeds 101/202/303, unchanged targets, common masks across models. Thirty
percent is the main stress severity. No new-regime safety certificate is inherited
from clean evidence. Real holidays and physical-duration outages require metadata.
Prediction intervals are secondary and require their own calibration-label budget
and validity assumptions; they are excluded from the core contribution for now.

## Efficient execution and gates

Do not launch a full grid now. The next implementation unit is a shared-data
preflight that mechanically instantiates these hashes, partitions and value
ledgers without learning. Then run compact candidate and statistical toy checks,
followed by a one-GPU Qwen memory/step/resume pilot. Use independent one-GPU
Slurm jobs, five CPUs per job and at most four concurrent tasks. Keep all data,
environments, logs/checkpoints and outputs in /home/tariq.qayyum/nature-llm/.
Existing email preferences apply to every eventual submission. No measured
runtime or future GPU-hour total is asserted here.

Before confirmatory runs, hash the exact implementation, runtime, effective
batch size, supported precision, baselines, rho, source exclusions and run matrix.
Engineering admission may select a memory-safe microbatch without changing
the common effective batch. Any scientific change requires a versioned protocol
amendment committed before confirmatory results. Do not retroactively tune eta,
data stages, source folds or test cohorts for acceptance or significance.

## What would make a defensible paper

A supported paper would show that verifiability-aware learning improves the
transfer/label/compute trade-off beyond the same ordinary forecaster plus the
same statistical verifier, with useful deployment activation and controlled
observed harm across eligible domains. It should explain which traffic-transfer
conditions create that improvement. The confidence-sequence construction is
credited prior methodology. A new theorem is not promised.

If all gains are due to Qwen size, an ordinary router, or fallback inactivity,
the proposed contribution has failed. If primary data or metadata gates fail,
narrow the claim openly. A Nature Communications submission depends on a
substantial measured scientific advance; no architecture choice guarantees it.

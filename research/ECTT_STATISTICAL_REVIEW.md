# Sequential comparison derivation and review status

Status: S01 INDEPENDENT REVIEW PENDING. This derivation and the synthetic code
checks do not constitute an independent review or a new theorem.

For one frozen target candidate pair, predictions and b_i must be measurable
before the next one-step label y_i. Define D_i as the difference of node-mean
absolute errors divided by the fixed support-validation scale s. Reverse
triangle inequality gives |D_i| <= b_i = mean_node|T_i-L_i|/s. It does not
require bounded labels or independent nodes.

Let mu_i = E[D_i | F_(i-1)]. Conditional on F_(i-1), D_i-mu_i lies in an
interval of length 2b_i. Hoeffding's lemma therefore gives

    E[exp(lambda*(D_i-mu_i)) | F_(i-1)] <= exp(lambda^2*b_i^2/2).

For S_n=sum(D_i-mu_i), V_n=sum(b_i^2), the nonnegative process
exp(lambda*S_n-lambda^2*V_n/2) is a supermartingale for each fixed lambda.
Mix lambda against N(0,1/rho). The integral is

    M_n = sqrt(rho/(V_n+rho))*exp(S_n^2/(2*(V_n+rho))).

Ville's inequality bounds the probability of ever crossing 1/delta by delta.
Solving that boundary yields

    |S_n|/n <= sqrt((V_n+rho)*log((V_n+rho)/(rho*delta^2)))/n.

This is the existing Gaussian-mixture Hoeffding boundary cited in the design.
Its center estimates the average forecast-origin conditional loss difference
over that observed prefix. It says nothing by itself about arbitrary future
blocks, per-origin safety, RMSE, WMAPE or a whole multi-candidate study's error.

Independent review must check:

- The filtration matches actual data and feedback availability.
- Source/target models and statistical scale are frozen before target E labels.
- No across-origin attention, adaptive refitting or unallocated candidate search.
- Predictable bounds are computed before labels, in the same units as D_i.
- Floating-point implementation, stable logarithms and all stopping paths.
- The source training margin is only a surrogate because theta sees its labels.
- Multiple streams are reported separately, without a study-wide 95% claim.
- Delayed/multi-horizon extensions have a separate argument before execution.

S02 simulates fixed null and alternative processes with known conditional
means and predictable varying envelopes. It uses 2000 streams of 256 origins,
seed 20261102. Its admission tolerance is delta plus four binomial standard
errors. Such tests can expose bugs; passing them cannot prove general coverage.

Primary sources: Choe and Ramdas, https://arxiv.org/abs/2110.00115;
Howard et al., https://doi.org/10.1214/20-AOS1991.

## Internal review update — 4 November 2026

The displayed width matches Howard et al. (2021), equation (3.7), with l0=1,
alpha=delta and V=sum b_i^2. The published paper is available at
https://par.nsf.gov/servlets/purl/10251927. Choe and Ramdas' sequential-forecast
interpretation is documented in https://doi.org/10.1287/opre.2021.0792.
Neither construction is claimed as new. This review is by the implementation
author, so S01 independent sign-off remains pending.

Define the prelabel information at origin i to contain frozen source/support
models, scale, seeds and all history available through row i-1. Forecasts L_i,
T_i and their disagreement bound b_i are measurable in that information.
The next label y_i is added only after both predictions are issued. Conditional
on source/support fitting information, Hoeffding's interval length is 2b_i;
the conditional exponential compensation is therefore lambda^2 b_i^2/2.
Mixing the nonnegative supermartingales over a fixed Gaussian prior and using
Ville's inequality gives the documented two-sided prefix comparison. Choosing
the first crossing after 12 origins is allowed within that same stream.

Critical interpretation checks remain: the center targets the average of
origin-specific conditional means, not a stationary unknown scalar or a later
deployment block. Sensor averaging occurs inside one origin's score, so adding
duplicate sensors does not increase n. Support selection is allowed because it
precedes verification, whereas searching candidate pairs or tuning thresholds
on verification outcomes would need a new error allocation/protocol. The
source-learning objective uses observed episode labels and is a training
surrogate, not a source-episode certificate.

The code audit found two implementation edges: nonfinite rho/scale values could
propagate silently, and the direct ratio/product could overflow despite a
representable width. The hardened code rejects invalid parameters/ranges and
uses logaddexp plus separated square roots for the same boundary. Locked rho,
delta, eta and stopping minimum are unchanged. Historical S02 job 612030 remains
evidence for its original code; the new CPU stage reruns the hardened simulation.

scripts/statistical_audit.py checks 36 independent Gaussian-Hermite integrals
against the analytical mixture, compares the direct moderate-range equation,
tests censoring/extreme arithmetic and confirms duplicate nodes add no evidence.
It reruns the three S02 scenarios using their fixed 2000x256 settings. Passing
these checks produces INTERNAL_CHECKS_PASSED_INDEPENDENT_SIGNOFF_PENDING.
These are numerical tests, not a formal floating-point coverage proof.

Before any certification claim, an independent reviewer must sign off the
derivation, filtration/label timing, frozen artifacts and finite-precision
interpretation. The persistence/ridge baseline stage does not invoke a transfer
certificate, so this pending sign-off does not block those descriptive scores.

## Reviewer handoff for the completed transfer grid

The CPU diagnostic reads the successful collection 621882 and its pinned
source and target artifacts. It emits S01_INPUTS.json with the exact
statistics code and protocol hashes, candidate shard and checkpoint hashes,
support and verification row ranges, and the fixed support-validation error
scale for each of 132 learned cases. The inventory contains no verification
truths or acceptance decisions. The existing numerical evidence is the
internal statistical audit in job 620562, report/internal_statistical_review.json.

The independent reviewer should record their name and date, the reviewed code
and protocol hashes, and a disposition of pass, changes required, or blocked.
Give a reason for each of these checks.

1. Derive the predictable disagreement bound and Gaussian-mixture width under
   the one-step filtration. Confirm that the scope is the observed prefix.
2. Trace the target support split and the local and transfer freeze to the
   exact source/residual hashes, support scale and candidate rows in the
   inventory. Earlier observed verification rows can enter later rolling
   histories. The current row label must arrive after its forecasts.
3. Inspect the verifier consumer before sign-off on its execution. It must
   process labels in order, stop charging labels at acceptance, record
   censoring when it does not accept, and freeze deployment before test.
4. Check floating-point handling, the strict stopping inequality, and the
   one-pair error allocation. Record any assumptions on precision and
   dependence, including the absence of a simultaneous study-wide claim.

A derivation review can proceed while the execution path is being prepared.
Independent sign-off requires a reviewer disposition.

## Chronological replay for S01 review

Review `scripts/frozen_verifier_replay.py` alongside `scripts/ectt_statistics.py`,
the locked protocol and the inventory from job 622973. The replay copies the
forecast pair and computes disagreement before requesting labels. The label
provider is called once per consumed origin and stops at the first crossing.
The return value contains the decision, the evidence count and the trace for
the rows actually consumed. The reviewer should check the provider's label
timing and the frozen artifact binding at integration, in addition to the
consumer and bound. The current implementation is tested on synthetic streams
and has no traffic-label provider wired to it. S01 independent sign-off is
pending.

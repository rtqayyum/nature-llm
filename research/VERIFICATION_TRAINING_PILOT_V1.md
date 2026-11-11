# ECTT verification-aware development pilot

Anchor collection 2773172 verified all 24 cases. The persistence anchor improved
10 of 12 comparisons. Anchored GPT-2 was closest to the selected local baseline
on both development datasets. This amendment uses it for the first matched
mechanism experiment before any expansion to held-out domains.

## Experimental design

Use PEMS08 and Beijing, seeds 11, 22 and 33, and the frozen 5% target budget.
Train 18 fresh source models. Each uses the same anchored GPT-2 architecture,
initialization, source exclusions, four eligible episodes per source domain,
optimizer, local comparator recipe, and source-validation stopping criterion.
Each source episode uses 48 support rows, 64 prefix origins and 16 suffix origins.
Use the same first-order support-head fitting and common suffix absolute error.
The selected checkpoint minimizes mean clipped source suffix MAE divided by
each episode's fixed support-validation local error scale.

| Condition | Source objective |
|---|---|
| `suffix_mae` | Unclipped physical suffix MAE divided by the support local-error scale |
| `prefix_mae` | Common suffix loss plus the mean of scaled clipped physical MAE over source prefixes of 16, 32 and 64 origins |
| `verification_margin` | Common suffix loss plus the Gaussian-mixture upper-margin penalty in ECTT_V2_DESIGN.md at the same prefix lengths |

The prefix MAE condition matches the verification condition's extra source
labels, clipping, prefix weighting and compute schedule. This additional control
separates the objective from the additional source supervision. The common
suffix loss preserves the unclipped training choice used in anchor development.
Source validation and all predictions scored by the verifier use physical zero
clipping. Lock gamma=1, epsilon=0.01, temperature=0.01, rho=0.01 and delta=0.05
from the existing contract. The training margin is a differentiable source
objective. Statistical coverage pertains to separately reviewed target replay.

Select persistence, ridge or compact on a chronological 70/30 split inside
source support. Retain the chosen family's inner validation error as the scale
with minimum one and refit the chosen comparator on full support for prefix
forecasting. Use the
existing compact recipe and exact declared architecture grid. Fit the temporary
transfer residual head on support and freeze it during each source update.
The prefix gradient uses a bounded two-pass replay, checked for exact forecast
agreement before backward. Three GPU probes exercise both source and residual
checkpoint recovery, and test causal forecast invariance.

## Target budget and ordering

Fit the target residual for 200 updates on its chronological support only.
Load the verified local selection and forecasts from the existing matched grid.
Freeze source weights, residual weights, scalers, local selection and origin
ledgers. Save both candidates on the verification and test origins. Candidate
files contain only predictions and origins. Rolling histories include observed
earlier rows, which the contract records separately from supervised labels.
Each candidate uses the same test cohort and the same fallback comparator.

The CPU review preparation runs in parallel with GPU probes. It provides the
exact scorer, its data flow, contract, code hashes and internal synthetic checks.
The independent reviewer verifies the statistical derivation, issue/maturity
ordering, forecast artifacts and scorer, and records a disposition in the
generated template. The independent approval record belongs at the path printed
by the candidate collection report. The CPU scorer checks that the approved
hashes match the exact pilot before any verification or test truth is indexed.

Replay each frozen verification stream chronologically. Read each matured row
through the consumer only after its forecast is fixed. Stop at the first
prefix of at least 12 origins whose upper bound is below -0.01. A rejected
candidate uses the local forecast at every test origin. Its restricted time is
available prefix + 1. Freeze all 18 decisions to disk before reading test truth.
Record every verification row and count unique support and verification values
per deployment. Each policy pays for its own access. Paired policies share the
raw benchmark labels and are analyzed as paired results.

## Analysis and next decision

Report deployed MAE, RMSE and WMAPE over all test origins, unconditional transfer
and local MAE, restricted evidence time and actual target-label consumption,
acceptance, useful and harmful acceptance, and the material-harm frequency and
positive excess-error tail. The same frozen verifier is applied in all arms.
Use paired seed results. For descriptive accuracy intervals, average seeds
within domains and apply 2,000 paired circular row-block resamples with block
lengths 72, 144 and 288. Label the two development domains and short-series
intervals accordingly. Compare evidence times descriptively across paired runs.

The development screen requires at least 2% lower macro deployed relative MAE,
lower restricted evidence fraction and no higher mean material harm than both
ordinary controls, with at least 20% acceptance and a useful acceptance. This
screen informs the held-out design review. Freeze all design choices before
using the six confirmatory domains. The later experiment includes Qwen and
the additional capacity and selector controls in the original protocol.

## Execution

`shared-transfer-ectt-development` submits the CPU review packet, three GPU
probes, 18 GPU fits and a CPU collector. Up to four GPU tasks run at once with
one GPU, five CPUs and 32 GiB per task. The fit array depends on successful
probes. The collector runs after all stages terminate and reports failures.
The journal supports partial submission recovery and the explicit retry mode.

`shared-transfer-ectt-score` queues the CPU evaluation of the same saved
forecasts after the reviewed disposition is recorded. It reuses complete
evaluation artifacts for the same pilot and approval hash. Both stages use the
same revision and project root. The scorer writes a complete per-run CSV,
paired comparison JSON, frozen decisions and test forecast shards.

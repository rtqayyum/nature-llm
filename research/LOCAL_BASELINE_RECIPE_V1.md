# Shared local baseline implementation recipe v1

Frozen before new baseline test scores, 4 November 2026. Parent source:
82eec4155fcfe3322a2df25b52c89eedb547ca18. Scientific settings remain the exact
config/ectt_protocol_v1.json bytes; this document fills in previously unspecified
ridge implementation details. No dataset, primary endpoint, budget or transfer
objective changes. Nature Communications remains the target; this foundation
stage cannot establish the proposed contribution.

## Implemented scope

B01 persistence and the ridge portion of B02, all eight canonical stage-two
raw channels, budgets 0/1/5/10 percent and identical final-20-percent targets.
Thirty-two deterministic dataset-budget cases reference seeds 11/22/33 without
duplicating fitting or treating seed copies as independent observations.
The real contract has 22 eligible positive-budget ridge cases, eight zero
budget cases, and two infeasible 1-percent cases (Beijing and Hangzhou).
The latter remain explicit persistence fallbacks. No extra labels are added.

The support-selected persistence/ridge forecast is a provisional two-family
comparison, not the final strong local comparator. A compact nonlanguage model
must join support selection before ECTT claims. B03 shared-data LLM-UTP, B04
ordinary Qwen, the ECTT objective and final statistical analysis follow later.
The old admission registry is a historical plan; current implementation
readiness is research/ECTT_IMPLEMENTATION_STATUS.json and the new recipe.

## Ridge and selection

For each node/origin, flatten the same 12x3 raw/trend/mask features as the Qwen
interface. All features come from within the origin's history. One intercept
and 36 coefficients are shared across nodes within a target domain. The target
is normalized next-row flow minus normalized last observed flow. Solve

    min_(a,w) mean[(residual-a-features*w)^2] + alpha*||w||_2^2.

The intercept is unpenalized. All accumulation, solving and saved forecasts are
FP64; the shared feature function remains FP32. Alphas are fixed at
0.0001, 0.01, 1 and 100, within the protocol's 12-configuration allowance.
This small classical control is not claimed as an optimized state-of-the-art
model. No baseline test outcome has been inspected to choose these values.

Split the raw support block chronologically at start+floor(0.7*support_length).
Fit inner normalization and coefficients using only the first segment, including
its 12-row warmup. Score eligible targets in the remaining support segment with
physical-unit MAE. Choose alpha by that score; choose persistence versus ridge
by the same score, with fixed candidate ordering for ties. Refit the selected
ridge recipe on all support. Save both the ridge forecast and the provisional
selected-family forecast. The support error scale is max(selected validation
MAE,1), retained for later audit but not used for a certificate here.

Constant inner support makes ridge unavailable under the frozen no-epsilon
scaling rule; report it explicitly and keep persistence. Never fit a scaler on
verification/test extrema. Never access the current target inside prediction.
Clip negative physical forecasts to zero. Score every test origin, including
the final computational remainder. Rolling observed history is allowed;
test labels never refit models or select hyperparameters.

## Outputs, access and recovery

The new shared-local-baselines job uses five CPUs, zero GPUs, 8 GiB and one
hour, shortq only if its observed maximum fits, otherwise longq. No assets or
dependencies download. It uses the existing project NumPy environment. The
launcher checks the recorded successful 618360 pilot and pinned source/model
dependencies; the worker verifies old completion manifests and current data
hashes inside Slurm. Historical stores/reports are not rewritten.

Outputs: outputs/ectt/shared_baselines/IDENTITY/cases/DATASET/BUDGET/ contains
fit.json, physical-unit forecasts.npz with origin IDs/common truths, metrics,
independent metric reductions and access_ledger.json. Unique support row-node
values are charged once. Input history unions and test evaluation truth access
are recorded separately. Verification labels used for fitting/selection are zero.

Each completed case is checksum-verified before reuse. A partly completed case
may recompute its small closed-form fit; completed cases never refit. Corrupt
data/complete artifacts fail and remain preserved. Final reports include JSON,
CSV and SUMMARY.md; live partial case results remain visible. Three seed labels
reference identical deterministic forecasts. Count final support fits separately
from the four inner-alpha solves per case and include all CPU time.

## Statistical audit and publication boundary

The same job runs the internal verifier review and hardened S02 simulation.
S01 independent sign-off remains pending. No transfer certificate or proposed
objective is evaluated by this stage. Baseline metrics remain descriptive until
the matched compact/reference/Qwen comparisons and analysis are complete.
Absolute time metadata, service gaps and raw sensor provenance remain unresolved;
the benchmark uses ordered row indices and does not invent calendar features.

The successful six-update Qwen pilot did not test an interrupted full-size GPU
optimizer resume. That check remains required before long GPU experiment grids;
it does not block this small deterministic CPU stage.

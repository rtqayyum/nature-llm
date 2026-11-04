# Compact local baseline recipe v1

Fixed 5 November 2026, after persistence/ridge job 620562 was reported complete.
All eight persistence/ridge test aggregate tables were already visible. No
compact model result has been inspected. The four generic capacity choices
below are fixed together, without adapting them to individual test outcomes.
This fills in the previously unspecified compact B02 implementation; the exact
scientific contract, data, budgets, seed panel, endpoints and verifier are
unchanged. No new claim of pristine benchmark exposure is made.

## Architecture and fitting

Use a graph-free transformer initialized from scratch for each target support
fit. Shared nodewise weights receive the same 12x3 raw/trend/mask history as
the other models. Four three-row temporal patches map to a learned embedding,
with learned four-position embeddings. An MLP encodes each node's history mean;
mask-weighted spatial pooling is confined to the same origin and accumulated
in FP64 before FP32 fusion. No calendar, road graph or cross-origin attention
is used. Four-token attention can see the full observed history.

Blocks use pre-layer normalization, four attention heads, a GELU feedforward
network of width four times the embedding and residual connections. A final
layer norm and bottleneck-32 GELU head predict a correction to the last observed
normalized value. The last head layer starts at zero, giving persistence at
initialization. All parameters are trainable. Dropout is zero.

The four predeclared configurations, in tie-breaking order, are (width,
blocks, heads): (64,2,4), (64,4,4), (128,2,4), (128,4,4). These use four of the
12 allowed configurations for this family. This is a planned compact control,
not a claim of optimized state-of-the-art accuracy. Its competitiveness must
be assessed before relying on it as the final strong local comparator.

Use FP32 weights, AdamW state, numerical computation and output, with TF32
disabled; use FP64 pooling and metric reductions. AdamW learning rate is
0.001, weight decay 0.01 and gradient norm clipping 1. Each inner fit and final
support refit uses exactly 200 updates, with no early stopping. The normalized
next-row MSE trains the residual forecast. Uniformly sample eight origins with
replacement per update, retaining all nodes. A separate CPU Torch generator
records sampling state. Reset model initialization to the declared seed for
each configuration and refit. No source traffic or previous target checkpoint
initializes this local baseline.

Split support raw rows chronologically at 70%. Fit inner min/max on the first
part only, with warmup inside it; select among four compact configurations by
physical MAE on the remaining support. Compare that best score with the saved
persistence and best-ridge support-validation scores from verified job 620562.
Family ties prefer persistence, then ridge, then compact. Refit the best compact
configuration from scratch on all support, regardless of which family wins,
to report both standalone compact and support-selected local forecasts.
Keep s=max(selected support-validation MAE,1). Never use verification or test
labels for fitting, scaling or selection. Freeze selection before test access.

## Next runnable unit

`shared-compact-pilot` uses PEMS08 and Beijing, seed 11, 5% budget, one GPU,
five CPUs, 32 GiB host RAM and a two-hour limit capped by observed gpuq policy.
It verifies the completed 32-case/22-fit CPU baseline store and its completed
case checksums, then verifies current canonical development arrays. It uses
the existing pinned project CUDA environment; no model weights or traffic
assets download. If environment versions differ, existing project-private
environment repair runs inside Slurm using the established setup helper.

The pilot performs four inner fits and one refit per domain: 2,000 benchmark
updates. Two six-update recovery paths per domain add 24 engineering updates,
for 2,024 updates on an uninterrupted first execution. These engineering
updates never enter the final benchmark model. Accounting distinguishes
completed updates, new updates and reused fits. Only two development cases
are evaluated; six confirmatory domains, other seeds/budgets and full B02
remain pending. No automatic follow-on submission or full-grid launcher is
introduced. Review measured memory, runtime, recovery and invariance before
implementing the all-domain grid.

## Admission, recovery and artifacts

Check single-origin, origin-order, later-companion mutation and node-order
invariance on the untrained largest model, every trained inner model and the
final refit. Absolute tolerance is 1e-5 and relative tolerance 1e-4 for this FP32
model. Reuse the established shared feature generator; prediction reads history
only. Synthetic regression tests guard against current/future label access.

Before benchmark fitting, compare a six-update uninterrupted largest compact
model with a fresh model/Adam reconstruction interrupted after update three.
Require bitwise-equal model tensors, Adam state, CPU/CUDA RNG, sampler RNG,
loss trajectory and cursor. This proves the tested compact reconstruction
path on that device; it does not clear full-size Qwen interrupted recovery.
That separate gate remains pending before long Qwen grids.

Each fit checkpoints every 50 updates, at five-minute boundaries and at its
end. A hash-verified atomic pointer names a complete checkpoint generation.
Checkpoints contain model, Adam, RNGs, losses and update cursor. A changed
source/configuration/environment/device identity cannot resume. Completed fits
and cases are checksum-verified and reused; corrupt artifacts stop execution.
Partial uncommitted updates can be replayed after a failure, and previous
checkpoint generations remain available. No old baseline store is modified.

Artifacts live under `outputs/ectt/compact_local/IDENTITY/cases/DATASET/0.05/11/`.
They include inner and refit checkpoints, support validation and selection,
pre/post invariance, recovery comparison, access ledger, common origin IDs,
physical-unit truths and four prediction arrays: persistence, ridge, compact,
support_selected_local. Every test origin, including the final batch remainder,
is scored. Independent metric reductions reopen the saved arrays. WMAPE
includes errors at zero truths; undefined zero denominators remain null.

The Slurm report contains compact_progress.json, compact_result.json,
compact_runtime.json, compact_store.json, environment hashes, partial results,
failure traceback when applicable and SUMMARY.md. Record timing of the whole
invocation including selection, probes and inference, peak host/GPU memory,
device, versions, parameters, source identity and new updates. A resumed
invocation is separately recorded; its elapsed time is not total lifetime cost.
Per-fit checkpoints retain accumulated committed training time. Summing all
Slurm allocations, including failed/replayed work, remains necessary for the
final total-cost analysis.

This job neither trains ECTT nor produces a transfer certificate. S01 remains
independent-signoff pending. Timestamp/service-gap and raw provenance limits
remain unchanged. Historical released/reference results stay separate.

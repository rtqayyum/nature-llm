# Research execution progress

## Acquisition completed

The researcher supplied completed HPC logs for Slurm job 605458, named
nutp-dl-audit, on cn119. Accounting reports COMPLETED, exit 0:0, elapsed
52 seconds, four allocated CPUs and zero GPUs. The report records all
37 requested assets verified, 5.494 GiB, zero failures, and all sixteen
public stage-array entries structurally audited. This is real HPC execution
reported by the researcher, not a synthetic test or a direct remote inspection
by the coding assistant. Local code tests remain separately identified.

The acquisition result is COMPLETED_WITH_REVIEW. Stage-one NYC bike and taxi
have identical hashes. Full Public8 joint training remains blocked pending
that release issue, and the full eleven-domain original retrain additionally
needs the three restricted archives. These limitations do not prevent a
separate PEMS08 checkpoint/split investigation.

Verified PEMS08 arrays are reported as 170 nodes by 17856 rows by two channels
for both stages. Original timestamps, channel provenance, checkpoint S1/S2/S3
identity and exact original GPT-2 initialization remain unresolved. No traffic
forecast, retraining run or replicated MAE/RMSE/WMAPE result exists yet.

## Next job

The pems08-audit stage submits nutp-p08-preflight, one CPU, 4 GiB RAM and
zero GPUs. It requests 30 minutes in shortq if that queue's observed MaxTime
supports the request, otherwise longq. No dataset/model downloads occur.

It verifies the selected existing hashes and pinned source, compares both
PEMS08 arrays, records the original split ranges and inner-batch tail loss,
counts train/validation target overlap, checks scaling/metric conventions,
and disassembles checkpoint metadata without unpickling tensors. It writes
six hashed origin CSVs and one debug-window NPZ. This job does not perform
strict model loading or prediction and cannot certify the training strategy.

The completed preflight report will determine the next checkpoint-inference
configuration. Faithful released-code behavior and a later clean protocol
must remain separate.

## PEMS08 preflight completed

The researcher supplied completed logs for job 605954, nutp-p08-preflight,
on shortq: COMPLETED, exit 0:0, elapsed 3 seconds, one CPU, no GPU, all five
checks complete and empty stderr. The report shows 12 constructed
train/validation target origins in common and three after dropping incomplete
inner batches. Fine-tuning test start is one row earlier than the pretraining
route. The full checkpoint/scaling JSON was not pasted; the next job reads
that existing artifact directly and requires matching asset hashes.
Checkpoint training strategy and absolute timestamp provenance remain unknown.

## First numerical checkpoint evaluation ready

The next stage is pems08-infer, nutp-p08-infer-cpu: longq, one task/eight CPU
threads, 16 GiB RAM, zero GPUs, a 23-hour ceiling capped by the observed queue
maximum. This first evaluation profiles CPU inference time. It creates a
separate project-local CPU environment with author-pinned core versions and
reuses the existing data, GPT-2 files and checkpoint. Dependencies download;
research assets do not download again.

It strictly loads the checkpoint into the unchanged model, tests one batch
and active-block execution, then scores all 1776 original test origins if
those checks pass. Shards carry identity/hash/target checks for recovery.
Author metrics are recalculated independently from the saved forecasts.
No actual HPC inference result exists yet. Local synthetic-weight checks
with the real pinned framework validate implementation, not traffic accuracy.

## All public checkpoint evaluations

The researcher requested all datasets at once after the first numerical
checkpoint evaluation. Stage public8-infer submits a CPU-only job with
four independent dataset workers and separate reports/logs. It evaluates
all eight public stage-two targets with the unchanged reference protocol.
The job checks the project reference environment once and reuses existing
research assets. No new model training occurs.

Dataset-specific hash/split/scaling preflights, strict checkpoint loading,
active-block checks, prediction shards and independent metric calculations
are required. The combined report retains paper-row differences for review.
Any required worker failure makes the whole job nonzero without discarding
other results. Repeating the same stage resumes compatible saved shards.
Stage-one bike/taxi duplicates remain a joint-training blocker; restricted
examples are excluded. Actual execution evidence remains in HPC reports.

Local validation: 46 ordinary unit/contract checks and four optional
real-framework checks passed separately. They validate all eight original
loader ranges using synthetic time-index arrays, synthetic model strict
loading/forward behavior, PEMS08 and Beijing complete inference/resume,
and aggregate failure isolation. Shell syntax and compilation passed.

## Fixed release diagnostics prepared

The next stage is release-diagnose. It investigates three public release
comparisons together using source-defined inputs and split routes, without
new training. A frozen plan records all four input/route combinations per
dataset; identical Beijing range contracts produce two aliases, leaving ten
unique cases. Original faithful runs remain separate. Common-origin paired
scores, normalization extrema and all per-case metrics are retained, with no
score-based baseline selection. Detailed execution evidence stays on HPC.

Diagnostic implementation validation: 53 ordinary unit/contract checks and
five pinned-framework checks passed separately. Synthetic checks cover all
four release input/split combinations, independent scoring, saved-batch
recovery, matched-target comparisons, corrupt archives and worker failures.
Bash syntax and Python compilation passed. These are local implementation
checks; diagnostic measurements await HPC execution.

## Source evidence inspection prepared

Stage source-inspect reads the pinned Source Data workbook and lists the
nested RAR directories together. It preserves original archives, uses a
separate parser environment, records formulas/caches without evaluation,
and keeps member metadata separate from claims about original forecasts.
All actual execution details remain in HPC reports. Local verification passed
60 unit/contract checks, including real parsers on synthetic ZIP/RAR/workbook
fixtures, all-required-items success, partial resume and required failure.
Five unrelated optional framework checks were skipped for this change.
Shell syntax and Python compilation passed. Real source inspection is pending.

## Training preparation audit ready

The causality-audit stage measures a fixed forecast's dependence on later
companion-window inputs with unchanged original weights and operations.
Eight dataset workers are scheduled two at a time. A training-origin gradient
probe records declared versus effective trainable parameters without updates.
Results remain HPC evidence; no clean protocol or independent training
reproduction follows automatically from these probes. Future training must
separate faithful release behavior from a predeclared causal implementation.

Local validation for the causality audit passed 64 unit/contract checks and
three pinned-runtime model checks: positive/negative synthetic dependence
controls and the original author model with synthetic weights, including a
zero-update backward pass. Bash syntax and Python compilation also passed.
The all-eight real-checkpoint probe remains unexecuted locally. Source archive
scalar-output identity remains open; available RAR-reader paths are recorded
on HPC for a possible targeted comparison.

## First independent GPU smoke prepared

The gpu-smoke stage begins real weight updates on one fixed training batch
for PEMS08 and one metro pilot. It preserves released behavior for an explicit
engineering check, including the known batch/scaling limitations, and starts
from a pinned base plus fresh seeded forecasting layers. Target checkpoints
are not reused as initialization. No benchmark training result or full paper
reproduction follows from a smoke pass. Execution evidence remains on HPC.

Local tests cover isolated CUDA installation commands, one-GPU scheduling,
training versus benchmark accounting, checkpoint/optimizer/dropout-RNG resume,
and 30-step fitting of the original architecture with synthetic weights/data.
The real GPU/CUDA runtime and source-data smoke behavior remain unverified.

Validation for this change: 67 ordinary unit/contract checks passed, plus two
focused CPU framework checks for original-architecture synthetic fitting and
exact optimizer/dropout-RNG recovery. Bash syntax, compilation and source
containment checks passed. CUDA execution and real-data smoke loss reduction
remain unverified until the GPU job runs.

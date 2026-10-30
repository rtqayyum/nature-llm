# HPC execution profile

The existing project is /home/tariq.qayyum/nature-llm/. Its work/ directory
contains the partially completed acquisition and remains the stable data root.
This project retains that location; the generic hpc/PROJECT_NAME example is
not used for this already-running project.

| Work | Queue | Tasks | CPUs per task | GPUs | Concurrency |
|---|---|---|---|---|---|
| Current acquisition and audit | longq | 1 | 4 | 0 | One job, four asset workers |
| PEMS08 checkpoint and split preflight | shortq when observed limit supports 30 min, otherwise longq | 1 | 1 | 0 | One read-only job |
| First released-checkpoint evaluation | longq | 1 | 8 | 0 | One CPU job with fixed batch order |
| Future parallel CPU experiments | longq by default | Match implementation | Up to 40 in one process when useful | 0 | Profile first |
| Future GPU training | gpuq | 1 | 20 | 1 | Start with four independent array jobs; six when permitted |

The user supplied queue names shortq, mediumq, longq and gpuq. The default
uncertain-runtime CPU request is 23:00:00. Submission reads longq MaxTime when
available and reduces the request if that maximum is shorter. The script does
not infer the cluster's maximum GPU time or assume that six GPU jobs run
simultaneously. Queue availability, GPU types, node egress and scheduler limits
still require actual HPC observations.

Forty CPUs are useful only with work that uses them, such as independent data
folds or parallel numerical tasks. The current workload uses four concurrent
HTTP downloads, then an array audit; it does not need forty MPI ranks or a GPU.
No new scientific experiments are implemented or claimed by this Batch change.

## Current submission

Refresh the standalone scripts/bootstrap_hpc.py from GitHub into this project
and run it with --project /home/tariq.qayyum/nature-llm --submit --handoff.
The --submit path is now the default. It installs the project's Batch and Check
entrypoints and calls sbatch, returning immediately with the real job ID.
The --handoff flag pauses only a validated unmanaged downloader from this
project, retaining its .part files. No Slurm jobs are cancelled. If a scheduler
managed download is already running, no unidentified process is stopped.

Batch loads anaconda3, creates project/.venv, uses project-local caches and
temporary files, and invokes exactly one worker through srun. Slurm captures
the Batch script and the submitted source snapshot path is exported per job.
Data/code updates do not change the source used by a queued job.

## Check and recover

Run bash /home/tariq.qayyum/nature-llm/Check to inspect the most recently
submitted project job, or append its explicit ID. This queries squeue, sacct
and sstat; prints accounting state/reason, CPU/memory observations, completed
asset counts, failed worker logs and the last stdout/stderr lines; and writes
results/slurm_jobs/JOB_ID/monitor.json.

Slurm stdout/stderr: logs/nutp-dl-audit_JOB_ID.out and .err.
Per-asset logs: logs/nutp-dl-audit_JOB_ID_workers/asset-NN.log.
Job report: results/slurm_jobs/JOB_ID/JOB_REPORT.md.
Detailed acquisition report: results/slurm_jobs/JOB_ID/report/SUMMARY.md.
Machine-readable reports: job_status.json, report/status.json,
report/asset_progress.json and report/acquisition_audit.json.
Partial and final payloads: work/reference/snapshots/ and work/models/.

Repeat the Python bootstrap with --submit to recover. Verified files are
reused and incomplete objects resume where the server supports ranges. An
already pending/running project download job is detected rather than duplicated.
The existing project environment and prior reports are preserved.

A completed download can report COMPLETED_WITH_REVIEW because the published
stage-one bike/taxi files are identical. That known scientific issue is not a
failed transfer of bytes. Missing/invalid required files, environment errors
and command failures remain nonzero job exits. Slurm TIMEOUT, OUT_OF_MEMORY,
NODE_FAIL and cancellation are checked through accounting even if an abrupt
kill prevents a final JSON report.

## PEMS08 next step

Acquisition job 605458 has now completed successfully in the user-supplied
HPC logs. Update the standalone bootstrap and add --stage pems08-audit
--submit to run the next stage. It reuses project/.venv and the downloaded
files, with no data/model transfers. The source snapshot and all outputs stay
inside the existing project. The job name is nutp-p08-preflight, with 1 CPU,
4 GiB RAM, no GPU and a requested 30-minute limit. Selection uses shortq only
when its observed maximum supports this duration, with longq as fallback.

Check reports five preflight checks rather than 37 download workers. Results
are results/slurm_jobs/JOB_ID/report/pems08_audit.json, SUMMARY.md,
origin_ledgers/*.csv and first_test_window.npz. Recovery repeats the same
stage command. Strict checkpoint loading and actual prediction are later
steps; a completed static audit is not a reproduced numerical result.

## First numerical checkpoint evaluation

Preflight job 605954 is complete in user-supplied HPC logs. After refreshing
submit_hpc.py, submit --stage pems08-infer --submit. This requests longq,
eight CPUs, 16 GiB RAM, zero GPUs and up to 23 hours (queue maximum capped).
The job is named nutp-p08-infer-cpu. It loads anaconda3, selects Python 3.8
through 3.11, and creates .venv-reference-cpu. PyTorch 2.0.0+cpu comes from
the official PyTorch CPU wheel index. Core author versions are pinned;
additional compatibility pins and the full installed freeze are recorded.
The shared Conda installation and existing audit .venv remain unchanged.

Only packages download. The original model, loader, metric implementation,
checkpoint, data and official GPT-2 files are used from verified local paths.
Missing/unexpected keys or incompatible tensor shapes stop execution; strict
loading is never relaxed. Full-array normalization, inverse scaling, masks,
16-window batch order and dropped tail are retained. CPU/device differences
from the publication and unresolved checkpoint training strategy remain
explicit limitations, even if paper-row rounding agrees.

Check reports reference-environment phase, strict-load/inference progress,
completed/resumed batches, measured and independently checked metrics, and
diagnostic paper-row rounding. Training experiments remain zero.
Predictions/shards: outputs/pems08_reference_cpu/RUN_HASH/.
Job details: results/slurm_jobs/JOB_ID/report/inference_result.json,
evaluation_config.json, strict_load.json, runtime.json and SUMMARY.md.
Environment: report/reference_environment_freeze.txt.
If interrupted, repeat the same stage command. Compatible completed shards
are verified and reused; changed data/code/environment identity gets its own
store. A corrupted completed shard is reported rather than silently reused.

## Eight public checkpoint evaluations

Current stage: public8-infer. Submit via bootstrap --stage public8-infer --submit.
Allocation: nutp-public8-infer, longq, one task, 16 CPUs, 32 GiB RAM, no GPU,
23 hours capped by observed MaxTime. srun launches a controller with four
independent dataset workers, four CPU threads each; maximum sixteen threads.
It checks .venv-reference-cpu once, then reuses existing verified assets.
The complete eight-domain workload has not been timed on the cluster.

Logs: logs/nutp-public8-infer_JOB_ID.out/.err and
logs/nutp-public8-infer_JOB_ID_workers/SLUG.out/.err.
Reports: results/slurm_jobs/JOB_ID/report/SUMMARY.md, public8_result.json,
and datasets/SLUG/{inference_result.json,SUMMARY.md,preflight_SUMMARY.md}.
Forecasts: outputs/SLUG_reference_cpu/RUN_HASH/.
Check prints all eight worker states, batch progress, metrics, failures and
CPU/memory accounting. Repeat the same submit command after a failure;
matching saved shards are verified and reused. Source/environment changes
produce a new run identity and preserve older outputs.

## Fixed release diagnostic job

Stage release-diagnose submits nutp-release-diag, one task, 12 CPUs,
24 GiB RAM, zero GPUs, 30 minutes. It uses shortq if observed MaxTime fits,
otherwise longq. srun launches a controller with three isolated child
processes, four threads each. The reference environment is checked once.
Existing pinned datasets, GPT-2 files and target checkpoints are reused.
Ten unique source-defined input/split cases have separate logs and reports.
The whole job fails if any required case or paired validation fails.

Reports: results/slurm_jobs/JOB_ID/report/{SUMMARY.md,diagnostic_plan.json,
diagnostic_result.json}, and cases/CASE/ for full metadata.
Logs: logs/nutp-release-diag_JOB_ID.out/.err and
logs/nutp-release-diag_JOB_ID_workers/CASE.out/.err.
Predictions: outputs/release_diagnostics/SLUG/VARIANT/RUN_HASH/.
Check displays per-case progress and errors. Repeat the same submission stage
for recovery; intact predictions resume under the same run identity.
This is a CPU checkpoint investigation, not training reproduction. No original
metric, layer, inverse transformation, checkpoint key or arbitrary split is
changed to force agreement. All diagnostics stay labeled and separately saved.

## Source workbook and archive inspection

Stage source-inspect: nutp-source-inspect, one CPU, 4 GiB RAM, no GPU,
30 minutes in shortq when supported, otherwise longq. One srun worker loads
Anaconda and creates a separate .venv-source-audit with pinned small parser
dependencies; reference inference packages are unchanged. The job verifies
the existing ZIP and copies roughly 1 GB of RAR containers into a project
cache for seekable metadata inspection. Nested payloads are not extracted.
The workbook is read with cached values; source formulas are recorded and
missing caches disclosed. Partial/complete container copies are verified.

Report: results/slurm_jobs/JOB_ID/report/SUMMARY.md and source_inspection.json.
Sheet metadata and exports: report/workbook/. RAR listings have separate
*_archive_inventory.json files. Item logs are in the job's _workers/ log folder.
Copied containers remain in work/source_data_inspection/SOURCE_ZIP_SHA/.
Check displays per-item status, copy bytes, sheets, keywords and archive names.
Repeat the same submit command to resume. Required failures return nonzero.

## Causality and gradient audit

Stage causality-audit requests nutp-causal-audit, four CPUs, 16 GiB RAM,
zero GPUs, 30 minutes in shortq when supported, otherwise longq. One srun
controller runs two independent dataset processes, two threads each, covering
eight public checkpoints. It reuses .venv-reference-cpu and existing assets.
Each worker strictly loads weights, preserves a fixed forecast's history,
perturbs only later companion inputs and records clipped-output differences.
It also performs a training-range backward pass without an optimizer or update.

Reports are results/slurm_jobs/JOB_ID/report/{SUMMARY.md,causality_summary.json}
and per-dataset metadata/predictions under report/datasets/SLUG/.
Logs: logs/nutp-causal-audit_JOB_ID.out/.err and its _workers/ folder.
Check prints progress, effects, gradient counts and failure reasons. Repeat
submission to retry; original checkpoints and earlier reports remain unchanged.
These are engineering probes, not training experiments or accuracy evaluations.

## GPU training smoke

Stage gpu-smoke requests nutp-s3-gpu-smoke, gpuq, one GPU, 20 CPUs,
16 GiB host RAM, two hours capped by observed MaxTime. It installs/checks
.venv-reference-gpu with torch 2.0.0+cu118 and pinned reference dependencies.
CUDA/Triton/extension caches are project-local; shared Conda is unchanged.
The first run downloads CUDA packages, but existing data/base files are reused.
PEMS08 and Beijing run sequentially, 30 updates each on one training batch.
No test evaluation, full epoch, joint training or clean-protocol claim occurs.

Reports: results/slurm_jobs/JOB_ID/report/{SUMMARY.md,gpu_smoke_summary.json,
gpu_runtime.json,nvidia-smi.txt}. Parent logs: logs/nutp-s3-gpu-smoke_JOB_ID.out
and .err. Dataset result/error logs are in the corresponding _workers/ folder.
Checkpoints are outputs/gpu_smoke/SLUG/RUN_HASH/step-NNNN.pt with a verified
LATEST.json pointer. Repeat submission to resume model, optimizer and RNG.
Check shows real update counts separately from benchmark experiment counts.
A failed CUDA check, nonfinite gradient or failed fixed overfit criterion
makes the job nonzero; successful smoke fits remain subject to science review.


## Three-seed single-domain training and notifications

Stage reference-train3 submits nutp-s3-3seed with six array tasks: PEMS08
seeds 11/22/33 then Beijing seeds 11/22/33. Maximum active tasks: four.
Each task requests gpuq, one GPU, five CPUs, 16 GiB host RAM and up to 10 days,
capped by observed partition MaxTime. Setup reuses .venv-reference-gpu and
verified local data/base weights. At least 3 GiB free per starting task is
checked; allow roughly 12 GiB free for all checkpoint stores together.
Runtime/early stopping and scheduler availability remain to be measured.

The locked protocol config is config/reference_training.json: original
architecture/freeze policy, FP32 Adam 1e-4, inner 16/outer 1, full-array
normalization, stage1/pretrain split route, maximum 1000 epochs, patience
100, validation selection beginning at epoch two as in main_run.py. Only
whole inner batches shuffle, using documented epoch-specific seeds. Lazy
batches match the released loader without materializing every window.
This is independent single-domain reference training; it is not an exact
published S3 claim or a causal benchmark. Stage2/finetune cohorts are distinct.

Live training_progress.json updates each ten batches and at phase changes.
Human reports remain available while running. Full model/Adam/dropout RNG
checkpoints save every epoch and after five minutes checked at 50-batch
boundaries. Two most recent states and their referenced validation-best
weights are retained. Each run locks only its own identity directory.
Resubmission resumes compatible saved cursors/order and verifies hashes.
Test forecasts evaluate the best validation checkpoint only after training
finishes; prediction shards resume and verify. Test results are not used
for checkpoint selection or hyperparameter changes.

Logs: logs/nutp-s3-3seed_ARRAYID_TASK.out/.err.
Task reports: results/slurm_jobs/ARRAYID_TASK/report/.
Checkpoints/loss histories/test shards:
outputs/reference_training/SLUG/seed-SEED/RUN_HASH/.
Check ARRAYID summarizes all tasks and saves aggregate SUMMARY.md and
TRAINING_SUMMARY.json in results/slurm_jobs/ARRAYID/report/. Partial means
and sample standard deviations identify how many seeds actually finished.
Check ARRAYID_TASK shows one task; squeue/sacct remain authoritative for
TIMEOUT/CANCELLED/OOM when a process cannot write its own final status.
Repeat the launcher submission after the old array ends to resume all
unfinished runs; verified completed runs reuse results without fitting.

Set TRAFFIC_MAIL_USER on the first launcher submission; it persists as a
private project-local work/notification_preferences.json for every future
launcher stage. Mail flags include BEGIN, END, FAIL, REQUEUE, INVALID_DEPEND,
TIME_LIMIT, TIME_LIMIT_80 and ARRAY_TASKS. Slurm has no queued/submitted mail
event: the launcher attempts a queued receipt using local sendmail/mailx/mail
and records availability/acceptance in submission.json. Receipt acceptance
is not delivery confirmation. Without a local mail service the queued
receipt is unavailable; cluster-admin Slurm mail configuration controls
scheduler-event delivery. No SMTP password or external account is required.

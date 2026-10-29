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

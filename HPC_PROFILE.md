# HPC execution profile

The existing project is /home/tariq.qayyum/nature-llm/. Its work/ directory
contains the partially completed acquisition and remains the stable data root.
This project retains that location; the generic hpc/PROJECT_NAME example is
not used for this already-running project.

| Work | Queue | Tasks | CPUs per task | GPUs | Concurrency |
|---|---|---|---|---|---|
| Current acquisition and audit | longq | 1 | 4 | 0 | One job, four asset workers |
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

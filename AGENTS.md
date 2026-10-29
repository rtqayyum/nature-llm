# Nature LLM research execution

GitHub is the source of truth. Commit completed changes to rtqayyum/nature-llm.
The HPC does not have a Git executable or a Git module; updates and downloads
use verified GitHub HTTPS snapshots through scripts/bootstrap_hpc.py.

The established project root is /home/tariq.qayyum/nature-llm/. Keep source,
environments, dependencies, temporary files, caches, logs, results, figures,
reports and checkpoints inside it. Preserve existing work/ downloads. Do not
move this project into /home/tariq.qayyum/hpc/ without an explicit migration.
Never use another project's directories or modify /cm/shared/apps/anaconda3.

Heavy work belongs in Slurm. Do not run lengthy downloads, audits, training or
experiment grids on login terminals. Use longq for uncertain CPU runtimes;
choose a shorter queue only when its policy and a measured runtime justify it.
The user's default longq wall-time request is 23 hours, capped by an observed
partition maximum when available. Load anaconda3 and create a project-local
environment. Never install dependencies into the shared Anaconda prefix.

The current download/audit Batch requests one task, four CPUs, 8 GiB RAM,
zero GPUs and four download workers. It runs a foreground worker through srun.
Completed hashes and partial downloads must survive retries. The original
public data scope, hashes and source revisions stay fixed during this change.

For future GPU training, the user reports gpuq can allocate one GPU and
20 CPUs per job, with four concurrent jobs as the initial array limit and
six possible when permitted. Up to 40 CPUs per CPU job are available in longq.
These cluster capacities are user-supplied, not independently verified.
Request a GPU only for work that needs CUDA. Parallelize independent seeds,
folds or configurations with job arrays; do not duplicate shared downloads.
Forty MPI tasks and one process with forty threads are different allocations.

Use meaningful job names and unique paths per job/configuration/seed. Record
the exact source version, environment, task progress and Slurm job ID. Provide
submission, status, utilization, failure-log, report and recovery commands.
Never cancel another project's job or signal an unvalidated process. The
explicit --handoff option may pause only a recorded unmanaged downloader
whose exact /proc command, project work path and session identity match.
It never calls scancel and refuses scheduler-managed processes.

Before publishing Batch changes: run bash -n, compile Python, run relevant
synthetic tests and check containment, environment isolation and resources.
Distinguish local synthetic checks, live public-source retrieval, actual HPC
execution and unverified cluster policies/network access in the handoff.
Successful data acquisition is not successful paper reproduction. Known
duplicate-domain findings block Public8 research claims even when Slurm marks
the download job complete. Genuine required-file failures exit nonzero.

User-supplied real HPC logs confirm acquisition job 605458 completed with
37/37 assets verified and zero failures. Do not repeat acquisition as the next
scientific step. User logs also confirm preflight job 605954 completed all
five checks in shortq with exit 0. See PROGRESS.md. The next stage is
pems08-infer: a CPU diagnostic evaluation in .venv-reference-cpu with
torch 2.0.0+cpu and transformers 4.35.0. It requires matching completed
preflight hashes, loads weights_only=True and strict=True, and preserves the
author's model/loader/metric source. Stop on any key or shape mismatch. Never
strip prefixes, discard buffers, change active layers, repair splits or change
the implicit softmax to force agreement. Batch size/order are immutable.
Save and verify prediction shards for recovery. Paper-row rounding agreement
is diagnostic, not proof of checkpoint strategy or training reproduction.
CPU evaluation differs from the paper's GPU runtime and must be labeled.
No training occurs in this stage. On a fresh login session, module load slurm
is the researcher-demonstrated setup before submission commands.

The current authorized next step is public8-infer, requested as all datasets
at once after the first numerical evaluation. It evaluates eight public
stage-two targets with four isolated CPU subprocesses, four threads each,
in one Slurm job. Keep execution evidence and result payloads on the HPC.
Do not request new GPU training or substitute restricted example datasets.

After public checkpoint evaluation, the current step is release-diagnose.
Use only the fixed released stage1/stage2 inputs and pretrain/finetune routes
for the three investigated datasets. Write the source-defined plan before
metrics. Retain every case, deduplicate only identical input/range contracts,
and compare common target IDs with equal raw truth. Never select a diagnostic
case as the baseline because it matches a paper number. Store forecasts apart
from faithful runs. Raw channel differences, missing inputs, strict-load or
paired-check failures must stop the relevant diagnostic. Training remains a
later step; execution evidence stays in the HPC reports, outside GitHub.

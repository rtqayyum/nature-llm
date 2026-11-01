# Sequential experiment execution

The 35-family plan is `research/MASTER_PROTOCOL_V1.md`. Exact science settings
remain `config/ectt_protocol_v1.json`. This release implements **admission**,
not all benchmark training kernels. It designs the full registry and makes the
first three stages runnable without presenting planned work as completed.

The user now authorizes staged implementation and execution, one job at a time.
Every GPU job requests **one GPU and five CPUs**. CPU-only admission jobs request
five CPUs and zero GPUs. No four-GPU job, concurrent training array, automatic
grid submission or job cancellation is introduced.

## Runnable stages

| Stage | Families | Allocation | Outcome |
|---|---|---|---|
| ectt-preflight | D01 D02 D03 | 5 CPUs, 0 GPUs, 30 minutes | Audit all eight existing stage-two arrays, raw duplicates, splits, features and value budgets |
| ectt-stat-check | S02 | 5 CPUs, 0 GPUs, 30 minutes | Fixed synthetic null/alternative streams for the sequential bound implementation |
| ectt-gpu-pilot | P01 P02 | 5 CPUs, 1 GPU, 2 hours | PEMS08 support-only numerical Qwen/LoRA gradients, invariance and recovery |

Observed partition limits cap every request. Short CPU stages use shortq only
when its observed maximum fits; otherwise longq. CUDA uses gpuq. These are time
limits, not measured promises. Later full training may request up to ten days,
capped by the observed gpuq maximum, after the pilot establishes its resources.

The pilot uses all 28 Qwen blocks, numerical four-patch tokens and rank-8 q/v
LoRA. It performs six engineering updates on two support origins. It does not
fit the full proposed objective, score a test set or establish benchmark gains.
Pinned official Qwen weights and separate dependencies are downloaded only on
the HPC during that batch job. Existing datasets and reference environments
are reused without modification.

The two infeasible 1-percent budgets (Beijing and Hangzhou) remain explicit.
We count their within-budget history warmup and never increase their labels.
Canonical channel-zero semantics follow the released registry description;
raw sensor provenance, calendar timestamps and service gaps remain unverified.

## Full experiment registry

The first job saves `experiment_registry.json` and `experiment_families.csv`.
These preserve all 35 experiment IDs, dependencies, required/conditional/optional
status and implementation readiness. They also specify eight targets, three
seeds, four budgets, policy controls, objective/backbone controls, external
comparators, outage masks, statistical analysis and decision thresholds.

Core two-objective planning has **42 unique source fits**, after sharing the
NYC bike/taxi source exclusion group, and **132 eligible target adaptations**.
The 60 zero/infeasible objective/seed/budget records retain persistence fallback.
These are planned units; no such source fit is counted as executed here.

S01 independent mathematical review remains pending. A derivation/checklist
is in `research/ECTT_STATISTICAL_REVIEW.md`. Synthetic calibration does not
clear independent review. The engineering pilot can inspect model computation
without claiming statistical certification. B01–B04, full M/A benchmark kernels,
external adapters and report-analysis kernels remain explicitly marked
`PLANNED_NOT_IMPLEMENTED`. The next-stage command stops after admission.
It never substitutes the older two-domain causal-train3 stage for this study.

## Download and first submission

Use the pinned bootstrap URL supplied with the handoff to save
`/home/tariq.qayyum/nature-llm/submit_hpc.py`. It fetches a verified committed
GitHub snapshot without Git. On the HPC, the submission command is:

```bash
module load slurm
python3 /home/tariq.qayyum/nature-llm/submit_hpc.py \
  --project /home/tariq.qayyum/nature-llm \
  --stage ectt-preflight --submit
```

This calls sbatch with the recorded stage allocation, project-contained Batch,
immutable code path and saved notification preferences. The exact sbatch
argument list is stored in the job's `submission.json`. Do not use bare
`sbatch Batch`, whose default is the historical acquisition stage.

After checking each successful stage, submit the next implemented admission:

```bash
python3 /home/tariq.qayyum/nature-llm/submit_hpc.py \
  --project /home/tariq.qayyum/nature-llm \
  --stage ectt-next --submit
```

One recorded active study job blocks another. A file lock serializes duplicate
submission attempts. Failed prerequisites stop dependent jobs. No background
login-node training loop is used. Original jobs/results are not cancelled.

Notification preferences remain in `work/notification_preferences.json`.
If absent, set TRAFFIC_MAIL_USER to the user's chosen address before submission.
Study submission refuses silently omitted notifications. Slurm begin/end/fail/
requeue/time-limit events and the existing local queue receipt are requested.
Acceptance by Slurm or local mail service does not verify inbox delivery.

## Monitoring and recovery

```bash
bash /home/tariq.qayyum/nature-llm/Check
squeue -u tariq.qayyum
J=$(cat /home/tariq.qayyum/nature-llm/work/LAST_JOB_ID)
sacct -j "$J" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,AllocTRES%30
sstat -j "${J}.0" --format=JobID,AveCPU,MaxRSS,MaxVMSize
tail -n 60 "/home/tariq.qayyum/nature-llm/logs/ectt-data-check_${J}.out"
tail -n 60 "/home/tariq.qayyum/nature-llm/logs/ectt-data-check_${J}.err"
cat "/home/tariq.qayyum/nature-llm/results/slurm_jobs/${J}/report/SUMMARY.md"
```

The tail commands above apply to the first stage; later job-name prefixes are
ectt-stat-check and ectt-qwen-pilot. `Check` picks the correct logs automatically.
No placeholder JOB_ID needs to be typed into a filename.

Reports are `results/slurm_jobs/JOBID/report/SUMMARY.md`, `ectt_progress.json`
and `ectt_result.json`. The report's `ectt_store.json` gives the precise store
under `outputs/ectt/admission/CONTRACT_HASH/STAGE/`. Data workers have separate
`worker_logs/DATASET.log`, per-domain audit JSON, CSV origin ledgers and complete
rectangular row/node budget ledgers. The pilot saves a checkpoint each update,
including trainable tensors, Adam state, CPU/CUDA RNG and cursor. Frozen model
weights remain separately pinned.

On failure, inspect Check and the named error. After resolving its cause,
repeat the **same explicit stage** submission, e.g. `--stage ectt-preflight`.
Do not change to ectt-next just to bypass a failure. Verified domain artifacts
and compatible checkpoints resume; corrupt artifacts fail and remain preserved.
A completed-stage replay performs no new optimizer updates. Different code or
contracts use a new identity and keep old outputs.

## Validation boundary

Local validation uses synthetic arrays, mocked Slurm and tiny randomly
initialized CPU Qwen models. It checks invariance, gradients, interrupted
optimizer recovery, data corruption, budgets, submission resources and failure
propagation. Official full-size weights and real datasets are not downloaded
locally. Full-size CUDA behavior, compute-node network access, cluster mail
delivery and actual job completion remain unverified until the HPC runs.

This environment cannot route to the university private SSH address. Code is
published through GitHub; a real Slurm ID is reported only after the user runs
the HPC submission command.

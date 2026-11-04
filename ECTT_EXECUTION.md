# Sequential experiment execution

The 35-family plan is `research/MASTER_PROTOCOL_V1.md`. Exact science settings
remain `config/ectt_protocol_v1.json`. This release implements **admission**,
not all benchmark training kernels. It designs the full registry and makes the
first three stages runnable without presenting planned work as completed.
The 4 November extension adds the first shared local baseline stage below.
The 5 November extension adds a development-only compact local pilot after
successful foundation job 620562; see the current next-stage section below.

Initial admission used one job at a time. The latest explicit user request
authorizes the complete remaining compact array, with up to four concurrent
tasks and a dependent CPU collector. Every GPU task requests **one GPU and
five CPUs**; CPU collection requests five CPUs and zero GPUs. No other model
grid or job cancellation is introduced. See the latest section below.

## Runnable stages

| Stage | Families | Allocation | Outcome |
|---|---|---|---|
| ectt-preflight | D01 D02 D03 | 5 CPUs, 0 GPUs, 30 minutes | Audit all eight existing stage-two arrays, raw duplicates, splits, features and value budgets |
| ectt-stat-check | S02 | 5 CPUs, 0 GPUs, 30 minutes | Fixed synthetic null/alternative streams for the sequential bound implementation |
| ectt-gpu-pilot | P01 P02 | 5 CPUs, 1 GPU, 2 hours | PEMS08 support-only numerical Qwen/LoRA gradients, invariance and recovery |
| shared-local-baselines | B01, ridge part of B02 | 5 CPUs, 0 GPUs, 1 hour | Internal verifier audit/simulations; persistence/ridge on all eight domains and four budgets |
| shared-compact-pilot | Development part of compact B02, P01/P02 | 5 CPUs, 1 GPU, 2 hours | PEMS08/Beijing, 5%, seed 11; support-only four-size selection/refit, exact compact optimizer recovery and saved-forecast metrics |
| shared-compact-baselines | Remaining compact B02 | Per task: 5 CPUs, 1 GPU, 2 hours; up to 4 concurrent | All 64 remaining eligible fits submitted together; 2 pilot cases reused |
| shared-compact-report | B02 artifact collection | 5 CPUs, 0 GPUs, 1 hour | After-any collection of all 96 records, including 30 deterministic fallbacks; fail on missing/corrupt evidence |

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
without claiming statistical certification. Full B02–B04, full M/A benchmark kernels,
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

### Recovery from GPU pilot job 612032

The user-reported job failed with exit 2 before model loading or optimizer
updates. Its private environment passed setup, but the pilot entrypoint tried
to import trainable_state and restore_trainable from ectt_model even though
both helpers are already defined in ectt_pilot. The import is corrected.

The repaired code has a new identity. It may reuse only the completed CPU
admissions under d70d0aa826dd4cd7d421f402fb730fed91d8bc9d79184bb9b39ea316c9c62eb2
(jobs 612029 and 612030), after verifying every saved artifact and a pinned
fingerprint of the unchanged admission dependencies, configurations and data
lock. Any dependency change, missing marker or corrupt artifact blocks reuse.
The import repair excluded only the pilot and workflow modules from this
compatibility check. Old pilot results/checkpoints cannot satisfy the new
pilot. Original stores/reports are never rewritten or relabeled.

The new report records its prerequisite store paths and identities in
ectt_prerequisites.json. The project CUDA environment and cached model files
retain their existing version/hash checks. Submit the same explicit stage:

```bash
module load slurm
python3 /home/tariq.qayyum/nature-llm/submit_hpc.py \
  --project /home/tariq.qayyum/nature-llm \
  --stage ectt-gpu-pilot --submit
```

The bootstrap downloads the latest verified commit. The retry gets a new
Slurm job ID, one GPU, five CPUs and the unchanged two-hour ceiling. Check the
new job using the printed ID or the project's Check command without an ID.
The failed 612032 report remains available for failure/resource accounting.

## Completed foundation after successful pilot 618360

User-supplied HPC evidence confirms the corrected Qwen pilot completed six
updates and all eight invariance checks with zero differences, plus strict
model reconstruction, in 37 seconds on one A100/five CPUs. It did not score
test accuracy or test an interrupted full-size optimizer resume.

The foundation stage was **shared-local-baselines**, now completed as job
620562. Its original submission instructions are retained here. Download the pinned
bootstrap first: older launchers do not know this stage name. Then submit:

```bash
module load slurm
python3 /home/tariq.qayyum/nature-llm/submit_hpc.py \
  --project /home/tariq.qayyum/nature-llm \
  --stage shared-local-baselines --submit
```

Allocation is five CPUs, zero GPUs, 8 GiB, one hour; shortq is used only when
its observed maximum fits, otherwise longq. Existing NumPy/data are reused;
there are no downloads or shared-environment changes. A recorded active study
job blocks another. Saved notification preferences remain mandatory.

The worker verifies successful 618360 evidence, pinned model/data code and
historical completion manifests, then reruns hardened synthetic verifier
checks and produces 32 deterministic dataset-budget cases. There are 22
eligible ridge cases under the real frozen contract. Seeds reference identical
deterministic results rather than independent replications. Zero/infeasible
budgets retain persistence. See research/LOCAL_BASELINE_RECIPE_V1.md.

Check prints baseline_progress.json with completed/32 cases. Results include
SUMMARY.md, baseline_result.json, baseline_results.csv, per-case physical-unit
forecasts, coefficients, independent metrics, access ledgers and runtime.
They live under outputs/ectt/shared_baselines/IDENTITY/ and the printed Slurm
report directory. Recovery repeats the same explicit stage; completed cases
are checksum-verified and reused. The bootstrap retrieves the latest commit.

S01 independent sign-off remains pending; the author audit cannot clear it.
B01 is implemented and B02 ridge is implemented. Compact B02 and B03/B04 remain
pending; the provisional persistence/ridge selector is not the final strong
fallback. The new status registry records these boundaries. The historical
ectt-next admission ordering does not submit this new stage automatically.

### Numerical repair after job 616045

Job 616045 passed environment setup, reused both completed CPU admissions,
downloaded/verified and loaded Qwen, then failed the pretraining node-permutation
check with maximum forecast difference 0.0988702178. No optimizer steps or test
evaluation occurred. Its original failure report and store remain intact.

Local reproduction found order-dependent FP32 spatial pooling changed BF16
tokens before Qwen. The repair accumulates the weighted node mean in FP64 and
casts back to FP32 before token fusion; Qwen stays BF16. The equation, weights,
architecture, data, label budget and 1e-3 absolute/relative admission tolerance
are unchanged. See research/ECTT_ENGINEERING_AMENDMENT_20261103.md.

CPU admission reuse now also excludes ectt_model, which those CPU stages never
import or execute. The pinned fingerprint of all remaining admission inputs
must still match the original release, and all old completion artifacts are
verified. A new GPU identity prevents reuse of old model results/checkpoints.
The cached environment and Qwen files retain their original version/hash checks.

Each completed invariance check is saved in pretraining_invariance.json or
posttraining_invariance.json and live progress, including the failed case.
The same explicit ectt-gpu-pilot command above submits the corrected pilot with
one GPU, five CPUs and two hours. Full-size CUDA admission remains unverified
until this new HPC execution; local tests are not a traffic accuracy result.

## Validation boundary

Local validation uses synthetic arrays, mocked Slurm and tiny randomly
initialized CPU Qwen models. It checks invariance, gradients, interrupted
optimizer recovery, data corruption, budgets, submission resources and failure
propagation. Official full-size weights and real datasets are not downloaded
locally. User-reported 618360 verifies the tested full-size CUDA smoke, and
620562 verifies the completed persistence/ridge foundation. Compact CUDA
execution, interrupted full-size Qwen optimizer resume and cluster mail
delivery remain unverified until their corresponding evidence is supplied.

This environment cannot route to the university private SSH address. Code is
published through GitHub; a real Slurm ID is reported only after the user runs
the HPC submission command.

## Current next stage: compact local development pilot

Job 620562 completed all 32 deterministic cases and 22 ridge fits in 53 seconds
with exit 0:0. Keep its artifacts; do not resubmit the completed CPU foundation.
The next explicit stage is **shared-compact-pilot**. Refresh submit_hpc.py from
the new pinned commit first, then run:

```bash
module load slurm
python3 /home/tariq.qayyum/nature-llm/submit_hpc.py \
  --project /home/tariq.qayyum/nature-llm \
  --stage shared-compact-pilot --submit
```

This requests one GPU, five CPUs, 32 GiB and two hours, capped by gpuq policy.
PEMS08 and Beijing use the primary 5% budget and seed 11. Four compact sizes
are selected on chronological support validation; the best compact is refit
on full support. Persistence/ridge forecasts and support scores are reused
from verified 620562 artifacts. See research/COMPACT_LOCAL_RECIPE_V1.md.

The job first checks compact exact optimizer/RNG/cursor recovery, then trained
invariance and all-origin physical-unit metrics. It performs 2,000 benchmark
updates plus 24 separate recovery updates on a fresh uninterrupted run. Check
displays compact_progress.json, completed/2 development cases and new updates.
Results include compact_result.json, compact_runtime.json, compact_store.json,
selection/validation records, checkpoints, common truths, saved forecasts,
access ledgers, independent metrics and SUMMARY.md. Logs use
logs/ectt-compact-pilot_JOBID.out and .err. Use the printed ID with Check.

Recovery repeats the same explicit stage. Completed fits/cases are verified
and reused; compatible partial checkpoints restore Adam, model, RNG and cursor.
No automatic chaining is enabled. Review GPU execution and runtime before the
all-domain compact grid. Other seeds/budgets, six confirmatory compact targets,
B03/B04, ECTT and independent S01 sign-off remain pending. This compact recovery
check cannot clear the separate full-size Qwen recovery requirement.

## Current request: submit all remaining compact cases together

User-reported pilot 620634 completed 0:0 in 77 seconds, all 2,024 updates and
compact recovery/invariance gates passed on one A100/five CPUs. The user now
requests all remaining jobs at once, with result analysis after completion.
This supersedes the earlier serial/development-only compact launch hold.

Use the new pinned launcher and submit one command:

```bash
module load slurm
python3 /home/tariq.qayyum/nature-llm/submit_hpc.py \
  --project /home/tariq.qayyum/nature-llm \
  --stage shared-compact-baselines --submit
```

It submits all 64 remaining GPU cases as an array, at most four running, and
an afterany CPU collection job. The complete 96-record matrix includes two
verified pilot cases and 30 zero/infeasible deterministic references without
extra GPU fitting. Scientific settings and the admitted compact implementation
are unchanged. Read research/COMPACT_GRID_EXECUTION_V1.md for the exact matrix,
artifact contracts, accounting and recovery boundaries.

The output prints ARRAY_JOB_ID and REPORT_JOB_ID. Return both real IDs;
Check ARRAY_ID summarizes the full array, and Check REPORT_ID shows the
collector. Repeating the command reuses known submission IDs or submits only
a missing collector. Once all old tasks have ended, add --retry-grid to create
a recovery array that reuses completed cases and restores partial checkpoints.
Failed/missing cases remain visible and make collection exit nonzero.

This submits the compact B02 family. B03/B04, ECTT, Qwen full-size interrupted
optimizer recovery and independent S01 sign-off remain pending. Actual grid
execution/results are established only by later HPC evidence.

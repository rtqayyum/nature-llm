# Compact local grid execution v1

5 November 2026. The user reports development pilot 620634 completed 0:0 in
77 seconds with 2,024 new optimizer steps, two development cases, compact
recovery/invariance gates passed, one A100/five CPUs and 405,189,632 bytes peak
GPU allocation. The two development test aggregates were visible. They do not
trigger recipe changes: compact_local_v1.json, the numerical model/training
helpers and all original dependencies remain byte-for-byte unchanged.

The user now explicitly requests submitting all remaining jobs together and
analyzing results after completion. This overrides the previous one-active-job
and development-only launch restriction for this compact B02 grid. It does
not claim unimplemented B03/B04/ECTT kernels exist or clear independent S01.

## Fixed matrix and execution

The eight domains, four budgets and seeds 11/22/33 produce 96 records. There are
66 eligible compact fits. Reuse the verified PEMS08 and Beijing 5%/seed-11
pilot artifacts, leaving 64 GPU tasks. Twenty-four zero-budget records and six
infeasible Beijing/Hangzhou 1% seed references retain persistence without
GPU fitting or extra labels. Deterministic references are not independent
replications. Do not select datasets, budgets or seeds after seeing outcomes.

`shared-compact-baselines` submits the 64 remaining tasks as one array with
`--array=0-63%4`, plus one CPU collector with `afterany:ARRAY_ID`. All tasks
enter the queue together; at most four run concurrently. Each requests one
GPU/five CPUs/32 GiB/two hours capped by gpuq. This ceiling preserves the pilot
profile; its measured 77-second allocation included two development cases,
but full-domain IO, scheduler waits and host load can differ. No total runtime
promise is made. The collector requests five CPUs/zero GPUs/8 GiB/one hour,
shortq only if its observed limit fits, otherwise longq.

Use the exact four architecture choices, 200 updates per inner/final fit,
support-only 70/30 selection, eight-origin effective batch, AdamW settings,
precision and recovery/invariance gates in COMPACT_LOCAL_RECIPE_V1.md. Only
dataset, declared budget and seed vary. New fresh GPU cases have 1,000 model
fit updates plus 12 separate recovery-probe updates each: 64,768 new updates
for all 64 uninterrupted first executions. Historical pilot updates are not
counted as new work. Every test origin and common raw truth remains fixed.

## Stores and verification

The grid has a separate identity under outputs/ectt/compact_grid/IDENTITY/.
plan.json enumerates all 96 records and the stable 64-task index mapping.
Each GPU case has its own lock, checkpoints, validation/selection, access
ledger, forecasts, independent metric reductions and completion hashes.
The unchanged admitted numerical helper runs inside cases/DOMAIN/BUDGET/SEED/
kernel/. A canonical outer result records actual development/confirmatory
membership and references the kernel's forecasts/access ledger. Original
pilot and persistence/ridge stores are verified and never rewritten.

Completed cases reuse their artifacts without optimizer updates. Compatible
partial fits restore model, Adam, CPU/CUDA/sampler RNG and cursor. Changed
computation, environment/device fit identity or corrupt artifacts stop reuse.
Each case logs actual source commit, environment, GPU/host memory and invocation
time; committed fit training time remains in its checkpoints. Final total cost
must include all scheduler allocations, retries and the historical pilot.

The after-any collector runs even when tasks fail. It requires successful final
task status, matching report/case identities, checksum-valid artifacts, passed
admission flags, support-only access ledgers and identical foundation truths/
origin IDs. It independently recomputes saved-forecast metrics. Reused pilot
and fallback records reference their original verified stores and prediction
columns rather than duplicate model fitting. Missing/failed/corrupt records
remain explicit and make collection exit nonzero. An incomplete array cannot
be called a completed B02 grid.

The collector writes compact_grid_result.json, compact_grid_results.csv and
SUMMARY.md under its Slurm report. The array report receives the result catalog
too. This collects evidence and metrics; scientific interpretation is deferred
until the user brings completed results. No success threshold, certificate or
method conclusion is produced automatically.

## Submission, monitoring and recovery

Refresh the pinned submit_hpc.py launcher, then submit once:

```bash
module load slurm
python3 /home/tariq.qayyum/nature-llm/submit_hpc.py \
  --project /home/tariq.qayyum/nature-llm \
  --stage shared-compact-baselines --submit
```

It prints ARRAY_JOB_ID and REPORT_JOB_ID, saves work/LAST_COMPACT_GRID.json,
and sets LAST_JOB_ID to the array. Check without an ID summarizes this array;
Check ARRAY_ID_INDEX shows an individual task; Check REPORT_ID shows collection.
Task logs are logs/ectt-compact-all_ARRAY_ID_INDEX.out/.err. Saved notifications,
including array-task events, remain mandatory. Local mail acceptance does not
verify inbox delivery. Other projects' jobs are untouched.

The launcher journals returned scheduler IDs before dependent submission.
Repeating the command reprints existing IDs without duplicate jobs. If only
collector submission failed, repeating it submits only that missing collector.
When the entire existing batch has ended, explicitly retry failed/incomplete
work using the same command plus `--retry-grid`. This creates a new array and
collector while checksum-verifying/reusing completed cases and restoring
compatible partial checkpoints. All earlier submissions remain recorded.
An active recorded study blocks an unrelated duplicate batch. No scancel,
unknown-process signaling or login-node fitting occurs.

Actual HPC submission cannot be made from the assistant's current private-
network route. IDs are established only from the user's real submission output.
The eventual full grid is pending until accounting and collection are supplied.
Qwen full-size interrupted recovery, B03/B04, ECTT and independent S01 remain
separate pending stages. Timestamp, service-gap and provenance limitations
continue to apply to this ordered-row benchmark.

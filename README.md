# Traffic forecasting reproduction

Reproduction-first evaluation of the LLM-UTP release. Source code is managed
through GitHub; all datasets, weights, environments, caches and experiment
outputs remain on the HPC under /home/tariq.qayyum/nature-llm/.
No Mac dataset upload or Git executable on the HPC is required.

A first public checkpoint evaluation has been completed by the researcher.
The next authorized step is evaluating all eight public targets together.
Training strategy, timestamp provenance and independent retraining remain
separate scientific checks. Execution results stay on the HPC.

## Next: independent GPU training-batch smoke

Refresh the launcher from a reviewed commit, then submit:

```bash
module load slurm
python3 /home/tariq.qayyum/nature-llm/submit_hpc.py \
  --project /home/tariq.qayyum/nature-llm \
  --stage gpu-smoke --submit
```

This requests nutp-s3-gpu-smoke in gpuq: one GPU, 20 CPUs, 16 GiB host RAM
and a two-hour ceiling capped by the observed queue maximum. The first run
installs the pinned torch 2.0.0 CUDA 11.8 build into .venv-reference-gpu.
Core author package versions match the separate CPU environment. Existing
datasets/base weights are reused; only CUDA framework dependencies download.
At least 8 GiB of free filesystem space is checked before creating the GPU
environment. CUDA, Triton and Torch extension caches remain project-local.

Following the pilot protocol, PEMS08 and Beijing run sequentially on one GPU.
Each receives a fresh seeded author encoder/head plus the pinned GPT-2 base.
No released target checkpoint is loaded. The original stage-one array,
pretraining training-range origins, full-array normalization, constructor
freeze policy, batch size, implicit softmax and Adam learning rate are retained.
This is a released-behavior engineering smoke, not a causal baseline or a
claim of exact original S3 training reproduction. CPU fallback is refused.

Thirty Adam updates fit one fixed training minibatch per dataset. Initial
and final MSE use evaluation mode on that same training batch. The locked
smoke criterion requires finite gradients and at least a 10% MSE reduction.
It is an engineering check, not validation/test improvement. No test forecast
or full epoch is evaluated. Full model/optimizer/RNG checkpoints are saved
at updates 10, 20 and 30 with hashes and experiment identity for recovery.
Original assets and previous forecast results stay untouched.

Check displays environment setup, actual GPU/runtime, per-dataset update
counts, batch losses, resumed steps and failures. It distinguishes real smoke
weight updates from completed benchmark experiments. Driver details are saved
in nvidia-smi.txt. Report: results/slurm_jobs/JOB_ID/report/SUMMARY.md and
gpu_smoke_summary.json. Checkpoints: outputs/gpu_smoke/SLUG/RUN_HASH/.
Individual result/error logs use the job's _workers/ folder. Repeat the same
submit command after failure; compatible checkpoints resume. CUDA/driver
compatibility and real-data overfit behavior remain to be verified on HPC.

This closes the initial reproduction investigation with its limitations
recorded. Later independent pilots and the proposed method need a separately
specified causal protocol. A smoke pass cannot remove the known batch,
scaling, preprocessing or provenance limitations of the released baseline.

## Training preparation: causality and gradient audit

Refresh the launcher, then submit --stage causality-audit --submit.
This uses nutp-causal-audit: four CPUs, 16 GiB RAM, no GPU, 30 minutes
in shortq when supported, otherwise longq. Two isolated workers use two
threads each and audit all eight public released checkpoints together.
Existing data, base files, weights and the CPU reference environment are reused.

For the first test forecast, all its own history and all companion observations
available at its issue time stay fixed. Only later companion observations are
replaced by two predeclared normalized values (zero and one). Both normalized
and clipped flow-output differences are recorded. Implicit-softmax call shapes
are traced without changing the mathematical operation. An unchanged repeated
pass checks numerical stability. These are synthetic counterfactuals on real
checkpoints, not new forecast accuracy measurements. A negative result on one
batch does not prove causality for all possible inputs.

A separate gradient probe uses the first complete training-route batch with
the release's stage-two input and normalization. It records each named
parameter's requires_grad flag, gradient presence and nonzero-gradient flag.
No optimizer is created, no update is performed, and parameter versions must
remain unchanged. This diagnoses the released constructor's effective policy;
it does not silently select a paper-consistent S3 training policy.

Reports: results/slurm_jobs/JOB_ID/report/SUMMARY.md and causality_summary.json.
Per-dataset causality_result.json, strict_load.json and probe_predictions.npz
are under report/datasets/SLUG/. Logs use nutp-causal-audit_JOB_ID and its
_workers/ folder. Check prints dataset states, future-input differences,
parameter counts and required failures. Repeat the same stage command after
failure; prior reports remain intact. No data/weight files are changed.

Available RAR-reader executable paths are recorded for a possible targeted
inspection of scalar-output tensors. Their capability and the tensors'
experiment identity remain unverified. No nested archive is expanded here.

The next independent training pilot must keep faithful released-code behavior
and a separately specified causal protocol distinct. Cross-window dependence,
precomputed channel provenance, scaling, split boundaries and trainable-parameter
policy must remain explicit. A diagnostic finding does not by itself establish
paper-wide leakage, explain all metric differences, or prove the new method.

## Current step: inspect released Source Data

Refresh the standalone launcher from a reviewed commit, then submit:

```bash
module load slurm
python3 /home/tariq.qayyum/nature-llm/submit_hpc.py \
  --project /home/tariq.qayyum/nature-llm \
  --stage source-inspect --submit
```

This submits nutp-source-inspect, one CPU, 4 GiB RAM, zero GPUs, 30 minutes
in shortq when supported, otherwise longq. The job uses the existing pinned
Source Data ZIP. It copies the workbook and two RAR containers into a separate
hash-keyed project cache, adding roughly 1 GB of storage. It checks ZIP CRCs,
records copied-file hashes, verifies completed copies and checks partial
prefixes before resuming. Original data and predictions stay unchanged.

The project-local .venv-source-audit contains pinned openpyxl/rarfile packages.
It is separate from the reference environment so inspection does not change
forecast-cache identities. Only the small inspection dependencies may download.
No Git or external RAR extraction program is needed for directory listing.

Every workbook sheet is exported as cached-value CSV and cell JSONL, recording
original formulas and missing cached values without recalculation. Sheet
names, previews and metric/dataset keyword rows are indexed. RAR member names,
compressed sizes, declared payload sizes and password flags are recorded.
RAR payloads are not decompressed, and attention tensors are not loaded.
File names and Supplementary Figure labels do not establish correspondence
with the main paper table; that needs inspection of actual experiment metadata.

Reports: results/slurm_jobs/JOB_ID/report/SUMMARY.md, source_inspection.json,
workbook/workbook_inventory.json, sheet CSV/JSONL files and separate RAR
inventories. Copied containers: work/source_data_inspection/SOURCE_ZIP_SHA/.
Logs: logs/nutp-source-inspect_JOB_ID.out/.err and its _workers/ item logs.
Use Check for sheet names, archive members, copy progress and failure reasons.
Recovery: repeat source-inspect submission; verified complete copies are reused
and valid partial copies continue. Required failures make Slurm exit nonzero.
The report remains subject to scientific review; no inference or training runs.

## Next: fixed released-input and split diagnostics

After the public checkpoint evaluations, investigate discrepancies using:

```bash
module load slurm
python3 /home/tariq.qayyum/nature-llm/submit_hpc.py \
  --project /home/tariq.qayyum/nature-llm \
  --stage release-diagnose --submit
```

Refresh the standalone launcher from the reviewed GitHub commit first.
This submits nutp-release-diag: 12 CPUs, 24 GiB RAM, zero GPUs, 30 minutes
in shortq when supported, otherwise longq. Three isolated workers use four
threads each. Existing verified assets and the reference environment are
reused. No training or data acquisition starts.

The job also verifies the existing Source Data ZIP and inventories its member
names/sizes without extracting files, to identify any original forecast data.
The plan is written and hashed before model evaluation. It contains exactly
the released stage1/stage2 supplied arrays crossed with the pretrain/finetune
split routes for PEMS04, Shanghai and Beijing. Identical input/range cases
are aliases, giving ten unique evaluations. The target checkpoint, original
model, effective batches, scoring and inverse scaling stay fixed. Input
changes retain each supplied array's original full-array normalization; the
recorded extrema show whether normalization also changed.

Every result is retained; no best paper-matching setting is selected as a
replacement baseline. Paired results align by target index and verify raw
truth equality. Split-route comparisons report common-origin scores because
both the evaluated cohort and inner-batch grouping can change. The original
faithful evaluations remain separate from these diagnostic variants.

Check shows completed cases, resumed batches, failures, metrics and paired
prediction differences. Reports are results/slurm_jobs/JOB_ID/report/
SUMMARY.md, diagnostic_plan.json and diagnostic_result.json, plus cases/CASE/.
Logs are logs/nutp-release-diag_JOB_ID.out/.err and the corresponding
_JOB_ID_workers/ directory. Forecasts are stored separately in
outputs/release_diagnostics/SLUG/VARIANT/RUN_HASH/.
Repeat the same stage command for recovery; verified compatible shards resume.
A required case or paired-comparison failure makes the job nonzero.
These controls can expose release sensitivity, but cannot prove which
preprocessing or checkpoint strategy produced the paper's table.

## Run all eight public checkpoint evaluations

Refresh scripts/bootstrap_hpc.py from a reviewed GitHub commit into
/home/tariq.qayyum/nature-llm/submit_hpc.py. Then run on the HPC:

```bash
module load slurm
python3 /home/tariq.qayyum/nature-llm/submit_hpc.py \
  --project /home/tariq.qayyum/nature-llm \
  --stage public8-infer --submit
```

The launcher verifies the newest committed code snapshot, installs managed
Batch/Check entrypoints, invokes sbatch and prints the actual job ID.
The descriptive job name is nutp-public8-infer: longq, one task, 16 CPUs,
32 GiB RAM, zero GPUs. The 23-hour ceiling is capped by observed MaxTime.
One srun allocation runs four independent dataset workers at a time, each
with four CPU threads. Larger road evaluations are scheduled first.
The complete batch runtime is unverified. Heavy work remains inside Slurm.

The job evaluates Beijing, Shanghai and Hangzhou Metro; PEMS03, PEMS04 and
PEMS08; NYC bike and taxi. These are eight separate checkpoint evaluations,
not joint training. All research assets are reused. The project-local
.venv-reference-cpu environment is checked once before workers start; only
missing dependencies may install, never into shared Anaconda. The existing
PEMS08 result is preserved and a short PEMS08 rerun is included.

Each worker checks immutable hashes, both stage arrays, split/scaling
conventions and checkpoint metadata, then loads weights_only=True and
strict=True. Original author model, loader and metric source remain unchanged.
Full-array normalization, inverse scaling, mask conventions, active blocks,
batch size/order and dropped tails are retained. A failed key/shape/target
or metric check stops that worker. Other workers continue and the whole job
exits nonzero if any required dataset fails.

## Check progress, failures and results

```bash
bash /home/tariq.qayyum/nature-llm/Check
```

Append a numeric job ID for an older project job. Check displays scheduler
state/reasons, exit codes, CPU/memory accounting, completed datasets out of
eight, each worker's batches/resumed batches, metrics and failed-worker logs.
It also prints the exact actual log filenames, avoiding a JOB_ID placeholder.

- Combined report: results/slurm_jobs/JOB_ID/report/SUMMARY.md.
- Machine-readable comparison: results/slurm_jobs/JOB_ID/report/public8_result.json.
- Per-dataset reports: results/slurm_jobs/JOB_ID/report/datasets/SLUG/.
- Parent logs: logs/nutp-public8-infer_JOB_ID.out and .err.
- Worker logs: logs/nutp-public8-infer_JOB_ID_workers/SLUG.out and .err.
- Persistent forecasts: outputs/SLUG_reference_cpu/RUN_HASH/.

For recovery, repeat the same public8-infer submission command. Completed
shards are hash/target/scaling verified and reused when the run identity
matches. A different environment or code contract creates a separate identity
and retains prior outputs. An already pending/running recorded project job
is not duplicated. No other project's files or jobs are modified.

## Scientific scope

Published S1 anchors come from Zhang et al., Nature Communications 17, 7466
(2026), Table 1. Comparison is diagnostic: no checkpoint is assigned a
training strategy from its filename or rounded metric agreement.
CPU execution differs from the paper's GPU runtime. Original full-array
scaling and unverified service gaps prevent a clean-protocol claim.
Public8 joint training remains blocked by identical stage-one NYC bike/taxi
payloads. Their distinct stage-two arrays can still be evaluated separately.
Full Nanning Metro, Beijing Bus and railway archives are unavailable;
restricted example arrays are excluded from this batch.
No new method, Qwen training or independent reference retraining starts here.

## Pinned assets and acquisition recovery

The upstream source is [LLM-UTP/LLM-UTP](https://github.com/LLM-UTP/LLM-UTP),
commit 50a4f5b0e76d09c952a01f07390bf57c77b22a26.
The paper is [Zhang et al.](https://doi.org/10.1038/s41467-026-73610-2),
with [archived release](https://doi.org/10.5281/zenodo.19503649).
config/sources.lock.json records exact source Git blob identities and LFS
payload sizes/hashes. The acquisition stage fetches actual LFS objects,
not pointer files, into work/reference/snapshots/COMMIT/. It does not
substitute standard benchmark mirrors for the author's prepared arrays.
Official [GPT-2 base files](https://huggingface.co/openai-community/gpt2)
are pinned at 607a30d783dfa663caf39e06633721c8d4cfcd7e under
work/models/gpt2/REVISION/. This base revision alone does not prove the
original authors' initialization history.

To repair missing research assets, explicitly submit --stage acquire --submit.
Use --audit-only to verify without dataset/model downloads. This is separate
from the public8-infer task. Exit 2 is a real required-task failure.
Pipeline exit 3 means execution finished with scientific review required;
Batch maps that state to Slurm exit 0 while retaining the original code and
REPRODUCTION_NOT_ESTABLISHED status. Paper-row mismatches are reported for
investigation, never silently tuned away or treated as a full reproduction pass.

## Validation

Run python3 -m unittest discover -s tests -v and bash -n Batch.
Unit fixtures use synthetic commands/data and do not submit cluster jobs.
Optional real-framework smoke tests use synthetic weights and the pinned
author model/loader/metrics to check numerical flow and complete-shard
recovery. They do not measure released-checkpoint forecasting accuracy.
Real cluster execution is identified separately in PROGRESS.md.
See [HPC_PROFILE.md](HPC_PROFILE.md) and [AGENTS.md](AGENTS.md) for containment,
resource and research constraints.

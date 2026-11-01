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

The next source-evidence step is source-inspect. Inspect the existing workbook
and nested RAR directories in Slurm with a separate .venv-source-audit.
Keep full workbook cells, formulas/caches, directory metadata and per-item
failures on HPC. Do not assume Supplementary Figure archive predictions
match main-table cohorts. Never unpickle attention tensors or silently expand
nested payloads in this stage. Preserve original archives and forecast stores;
new container copies use verified, resumable project-local cache paths.

Training preparation now includes causality-audit on all eight checkpoints.
Keep the first target's history and every observation available at its issue
time fixed; change only later observations in companion windows. Trace
implicit softmax shapes without changing the calculation. Record normalized
and clipped-output changes. Gradient diagnostics must use training-route
origins, never optimizer updates; verify parameter versions unchanged.
Do not present synthetic counterfactuals as measured traffic accuracy or a
clean-protocol pass. Keep any future faithful and causal retraining separate.
Actual HPC evidence remains outside GitHub.

Current next stage: gpu-smoke, the first actual weight-update engineering
check. This is authorized training preparation on PEMS08 and Beijing under
the pilot protocol, superseding the earlier audit-only stage restrictions.
Use one GPU/20 CPUs in gpuq, project-local .venv-reference-gpu and caches.
Do not silently replace the released softmax/scaling/freeze policy. Start
from the pinned base plus seeded fresh encoder/head, not target checkpoints.
Fit a training minibatch only, save optimizer/RNG checkpoints and report
smoke updates separately from benchmark experiments. A separate causal
protocol remains necessary before new-method performance comparisons.


Current authorization: skip the one-seed timing pilot and run three seeds
(11,22,33) for independent single-domain PEMS08/Beijing reference training.
Use reference-train3, one dataset/seed per array task, at most four active,
one GPU/five CPUs/16 GiB per task, gpuq with a 10-day request capped by its
observed maximum. Preserve released stage1/pretrain inputs and architecture;
no clean/causal or exact published S3 claim. Keep test evaluation after
validation checkpoint selection. Save live progress, losses, model/Adam/RNG
recovery, prediction shards and per-run/aggregate reports. Serialize shared
environment setup only, never hold its lock through concurrent training.
All future launcher submissions read persistent work/notification_preferences.json.
The user explicitly authorizes job-event emails to the configured address.
Set it initially through TRAFFIC_MAIL_USER; keep the address/preferences and
actual mail receipt status on HPC, outside version control. Request Slurm
BEGIN/END/FAIL/REQUEUE/TIME_LIMIT and per-array-task messages. A submission
receipt uses local sendmail/mailx/mail; report missing mail support honestly.
Never claim delivery merely because Slurm/local mail accepted a request.


Current authorization: correct baseline evaluation before proceeding to the
proposed method. baseline-rescore is a CPU-only saved-forecast correction,
with explicit --source-array and no optimizer/model inference. Verify source
contracts, data and selected-model hashes, old forecast checksums, exact raw
truth and target coverage, then replay historical max-only scores and apply
the recorded training min/max inverse followed by unchanged zero clipping.
Preserve every source run/report/checkpoint. New derived forecasts and
reports use separate project-local paths and resume by verified hashes.
Zero-minimum examples must retain identical scores; negative minima require
the full affine inverse. Future reference_train results use the correct
inverse and export canonical physical-unit forecasts with an explicit version.
Leave inverse_author unchanged for faithful released-checkpoint evaluation.
A scaling correction does not remove full-array fitting or cross-window
future dependence. Never call this a clean baseline or certified S3 result.
Reconcile the baseline protocol before launching new-method comparisons.


Current authorization: execute the separate index-causal reference baseline
before the proposed method. Use causal-train3 and config/causal_reference.json,
with PEMS08/Beijing x seeds 11/22/33, one GPU/five CPUs/16 GiB per task,
max four active and up to ten days capped by gpuq maximum. See
CAUSAL_BASELINE.md for the locked protocol. Source row order is assumed;
calendar/service-gap metadata remains unverified. Do not claim exact S3.
Use raw channel zero only, require stage-one/two raw equality, fit min/max
on scored training support only, rebuild prefix means inside each 12-row
window and apply a two-call feature-axis fusion softmax patch. Leave original
source and released-behavior outputs intact. Pretraining and selected-model
causality gates must pass; failure is nonzero, never tune gates on test scores.
Use separate outputs/causal_reference stores, corrected physical-unit shards,
method-independent data contract hashes, live losses and strict recovery.
No proposed-method/Qwen training or downloads are authorized by this stage.

Current authorization supersedes the preceding causal-baseline launch:
reproducibility first, then the proposed solution. causal-train3 is on hold.
Use reproduction-review to consolidate explicitly registered historical HPC
evidence into a comparison/protocol matrix. This is one CPU, 2 GiB, zero GPU,
30 minutes in shortq if observed queue limits permit, otherwise longq.
No research assets, dependencies, new training, inference or rescoring are
needed. Missing/corrupt required evidence must fail; unavailable optional
reports must be labeled, never silently counted as successful checks.
Keep all source artifacts intact and all actual result evidence on HPC.
Support the historical checkpoint schema only by its reviewed source hash.
Do not equate paper-row rounding matches or better scores with reproduction,
or silently change trainability/data/batch behavior to force agreement.
See REPRODUCIBILITY.md. No proposed-method work or causal launch is authorized
until the reproducibility assessment and experiment scope are reviewed.

Latest authorization: after successful user-reported reproduction-review job
610996, the user requests related-work review, a defensible mechanism and
falsifiable hypotheses, and a frozen matched evaluation protocol. This
authorizes research design and versioned documents/configuration, not launching
training or calling a new method validated. research/ECTT_V2_DESIGN.md and
config/ectt_protocol_v1.json define the candidate verification-aware ECTT study.
Retain the old causal-stage submission hold. New benchmark implementation
must use a separate stage/identity: all eight canonical stage-two raw channels,
shared source/target ledgers and explicit limits on calendar/causality claims.
Do not call existing sequential statistics or Qwen substitution new inventions.

Latest project-plan consolidation: read research/MASTER_PROTOCOL_V1.md before
new experiment implementation. It records 35 experiment families, stage gates,
internal research thresholds and the Nature Communications target. The next
unit is shared-data preflight, then statistical/model admission, a one-GPU
engineering pilot and fair shared-data baselines before ECTT comparisons.
The Word document and master plan do not authorize submitting the full grid.
Keep config/ectt_protocol_v1.json as the exact numeric contract; any scientific
conflict or change needs a documented version amendment before affected runs.

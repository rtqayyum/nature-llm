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

Latest user authorization supersedes the design-only restriction: implement
the experiment schedule and run stages one by one. Every new GPU task uses
one GPU/five CPUs, with one active study job. CPU-only admission uses five
CPUs without a GPU. Read ECTT_EXECUTION.md. This release implements admission
and engineering stages, with all 35 experiment families registered; it must
not claim unimplemented full benchmark kernels or independent S01 review are
complete. Use the new shared-stage identities; do not relabel causal-train3.
Notifications are required for study submissions. GitHub publication is
authorized. Direct HPC SSH currently has no route; provide exact download,
submission, check/report and recovery commands without inventing job IDs.

User-reported HPC evidence now confirms ectt-preflight job 612029 completed
0:0 in nine seconds: 8/8 domains validated, zero failures, no canonical raw
duplicates. Study identity d70d0aa826dd4cd7d421f402fb730fed91d8bc9d79184bb9b39ea316c9c62eb2
must remain usable. Next is ectt-stat-check (five CPUs, no GPU), then the
one-GPU/five-CPU engineering pilot. Do not repeat data acquisition or call
numerical admission a complete provenance, causality or benchmark validation.

User-reported HPC job 612030 (ectt-stat-check) is now completed 0:0, with all
three synthetic scenarios passed. It does not clear independent S01 review.
The next implemented stage is ectt-gpu-pilot, one GPU/five CPUs in gpuq,
two hours subject to queue maximum. It is a six-update PEMS08 support-only
engineering pilot, not full baseline/method training or a traffic accuracy
claim. Preserve the validated d70d0aa... study identity during status updates.

User logs now confirm pilot job 612032 failed before model loading/training
with an ImportError for checkpoint helpers already local to ectt_pilot.
Environment setup passed; optimizer updates were zero. The repaired release
uses a new identity and narrowly reuses only completed d70d0aa... CPU admissions
after verifying their manifests and pinned unchanged dependency fingerprint.
Record reuse in ectt_prerequisites.json; never rewrite old stores or count an
old pilot as passed. Regression checks must call the real pilot entrypoint,
in addition to helper-level recovery tests. Resubmit ectt-gpu-pilot with one
GPU/five CPUs; real CUDA execution remains pending until new HPC evidence.

User-reported retry 616045 loaded pinned Qwen successfully and reused both CPU
admissions, then failed node-permutation admission before updates. A local
regression reproduced FP32 pooling order changing BF16 tokens. Accumulate the
node weighted mean in FP64, output FP32 and keep Qwen BF16 and both 1e-3 gate
tolerances. Preserve all old failures. Model code is unused by CPU admissions;
exclude it from their compatibility fingerprint along with pilot/workflow,
while still checking all remaining inputs and old completion artifacts. Never
reuse an old GPU result for this changed computation. Record per-case admission
progress even on failure. See research/ECTT_ENGINEERING_AMENDMENT_20261103.md.

User logs confirm GPU pilot 618360 completed 0:0 in 37 seconds: six updates,
eight invariant prediction checks with zero differences, strict reconstruction,
one A100/5 CPUs and 6.21 GiB peak allocation. No ECTT objective or test accuracy
was evaluated; full-size interrupted optimizer recovery remains untested.

Latest authorization: proceed with statistical review and matched baselines.
The next runnable unit is shared-local-baselines, five CPUs, no GPU, 8 GiB,
one hour capped/placed by observed queue limits. It verifies 618360 and pinned
model/scientific dependencies, then runs the internal verifier audit and B01
persistence/B02 ridge on all eight datasets and four budgets. See
research/LOCAL_BASELINE_RECIPE_V1.md. Preserve infeasible cases, shared targets,
support-only selection and all completed cases. Seeds reference deterministic
fits, not independent replications. Compact B02, B03/B04 and method kernels
remain pending. S01 independent sign-off cannot be cleared by the author audit.
The hardened verifier changes numeric code but not locked statistical settings;
rerun its fixed synthetic tests in the new CPU job, preserving old evidence.

User reports shared-local-baselines job 620562 completed 0:0 in 53 seconds:
32 deterministic cases, 22 ridge fits, independent saved-forecast reductions
and internal audit passed; independent S01 sign-off remains pending. Preserve
its identity 5aac118c3de58f0ba5aeb3b7bb219133359490fdd66cd2a8c642a4c80f181f34
and all case artifacts. No rerun of this completed foundation is needed.

Latest user authorization: implement the compact local baseline next. The
next runnable unit is shared-compact-pilot: PEMS08/Beijing, 5%, seed 11,
one GPU/five CPUs/32 GiB/two hours capped by gpuq. Read
research/COMPACT_LOCAL_RECIPE_V1.md. Verify and reuse 620562, select only on
support, test exact compact optimizer recovery and invariance, save common
truths/physical forecasts and independent metrics. Confirmatory compact runs,
other seeds/budgets, B03/B04 and ECTT remain pending. This pilot does not clear
the separate full-size Qwen interrupted recovery or independent S01 review.
Do not launch an all-domain GPU grid or automatic chaining from this release.

Latest explicit user authorization supersedes compact development-only and
one-active-job restrictions: after successful user-reported pilot 620634,
submit all remaining compact B02 jobs together and analyze once complete.
Use shared-compact-baselines: 64 GPU array tasks, at most four concurrent,
one GPU/five CPUs per task, plus an afterany five-CPU/no-GPU collector. The
96-record fixed matrix includes two reused pilot cases and 30 deterministic
fallback references. Read research/COMPACT_GRID_EXECUTION_V1.md. Keep the
admitted compact recipe/model/training helpers unchanged. Preserve old stores,
failed/infeasible cases, source identities and notification preferences.
Do not relabel unimplemented B03/B04/ECTT or independent S01 as completed.
The assistant cannot submit through the private HPC route; provide the single
array+collector command and ask for both real scheduler IDs. No need to pause
between each compact seed/domain task. Never cancel other project jobs.

User logs confirm array 620653 completed all 64 tasks 0:0 with 64,768 new
updates, and collector 620656 verified 96/96 records with zero failures.
The local baseline grid is complete; its full CSV has been inspected. Preserve
all recipes, selected families and output stores. No test-driven retuning.

Latest next-step authorization: perform existing development-log review and
the remaining full-size Qwen interrupted recovery check in one go. Use one
shared-transfer-check job, one GPU/five CPUs/32 GiB/two hours capped by gpuq.
Read research/PRETRANSFER_CHECKS_V1.md. Review only existing PEMS08/Beijing
training/validation artifacts, without refits or new test evaluation. Run
cached full-size Qwen in three fresh processes (six uninterrupted updates,
three updates then controlled process exit, fresh restore/continuation to six).
Require model/Adam/CPU-CUDA RNG/loss/cursor/support prediction equality and
unchanged invariance gates. Keep fixed-support recovery scope explicit;
future episodic/adaptation trainers require their own recovery tests. B03/B04,
ECTT and independent S01 remain pending. Publish the tested stage and provide
one submission command, then request its real Slurm ID. Do not invent IDs.

The user reports pre-transfer job 621799 passed the cached full-size Qwen
fixed-support process-restart check. Latest authorization queues the next
implemented jobs together with dependencies. Use shared-transfer-baselines
for the B03/B04 unconditional comparisons. See research/SHARED_TRANSFER_V1.md.
Queue data, recovery probes, source fits, target adaptation, and collection
in one submission. Preserve completed results and cap the GPU arrays at four
active tasks. Use saved notification preferences. Verify and publish the exact
commit and request the returned scheduler IDs. S01 independent review remains
pending, so verification decisions and ECTT mechanism runs remain separate.

The first transfer chain 621844–621848 passed data admission and stopped
at both CUDA probes. B03 failed cuDNN GRU backward in eval mode. B04 failed
single-origin invariance before updates. Apply the transfer-only fixes in
research/SHARED_TRANSFER_V1.md and retain the original run. The revised
chain must pass both GPU probes before starting source fits.

The user supplied successful corrected transfer receipts: both probes passed
in 621879 and collection 621882 verified 42 source fits and 192 target records.
The 132 learned cases all lost to the matched local reference on MAE, RMSE and
WMAPE. Next use shared-transfer-diagnose, a CPU-only read of the frozen run with
a separate report and S01 input inventory. Recheck support scalers, paired
test errors and saved adaptation traces. Confirmatory diagnostics are
descriptive. Do not tune on them, access verification labels, run the verifier
or launch the ECTT grid while the independent review remains pending.

Diagnostic job 622973 completed 0:0 and reconciled all 192 records and 132
frozen candidate pairs. At 5%, development transfer errors were widespread.
The user requests shared-transfer-attribution next. Use one GPU, five CPUs,
32 GiB RAM and up to four hours capped by gpuq. Compare source-only forecasts
and archived adapted forecasts for PEMS08 and Beijing at the frozen primary
5% budget, both model families and all three seeds. Keep the same target
support scaler in both arms. Replay the adapted outputs from verified
checkpoints and compare them at the existing normalized 1e-3 tolerances.
Save both pre-inverse outputs and paired metrics in a separate resumable
store. Use existing checkpoints for inference, with zero optimizer updates.
S01 independent review and ECTT training remain pending.

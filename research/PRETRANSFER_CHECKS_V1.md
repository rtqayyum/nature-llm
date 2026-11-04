# Development review and Qwen process-restart check

5 November 2026. User-supplied HPC evidence confirms array 620653 completed
all 64 GPU tasks with exit 0:0 and 64,768 new updates. Collector 620656 completed
0:0 in 47 seconds and verified all 96 records with no failed/missing case.
These include two historical pilot cases and 30 deterministic fallbacks.
The full local baseline CSV has been inspected. No baseline recipe, selected
family, architecture, budget, source data or threshold is changed in this work.

The user's next-step request authorizes one combined job, shared-transfer-check,
to perform the two outstanding preparation checks together. It requests one
GPU/five CPUs/32 GiB/two hours capped by observed gpuq policy. The CPU artifact
review is a short first phase of this GPU-required allocation. Notifications
and serial duplicate guards remain active. This is one submission and one
Slurm ID; main B03/B04 source training is a later implementation stage.

## Existing development artifacts

Review PEMS08 and Beijing only, all four budgets and seeds 11/22/33. There
are 24 development records: 15 trained cases and nine deterministic fallback
references. Verify completed cases and fit hashes without altering them.
For each trained case, inspect four inner training curves and the final
support refit: 75 curves, 200 updates each. Export all 15,000 existing recorded
sampled-minibatch losses in development_training_curves.csv, and report their
first/last 20-update means, minimum and recorded fit time/parameters.

Check the exact support split, inner/full scalers, unique-value access count,
absence of verification/test labels in fitting/selection, declared configuration
order, best support-validation configuration, family tie order and support
error scale. Report the existing compact/ridge/persistence support-validation
scores and chosen family. No new fitting, model inference or test evaluation
is performed by this review. Completion hashing reads original artifacts for
integrity; stored test metrics are not used to choose changes.

Resampled training-batch MSE is not a fixed-cohort learning curve. A first/last
mean and a single support-validation score cannot establish underfitting,
overfitting, representation collapse or a coding defect. The report describes
the observed diagnostics. Any later scientific change must be justified on
eligible development data and documented before affected comparisons; all
original baseline results remain available. No result-driven replacement of
the already evaluated confirmatory grid occurs here.

## Cached full-size Qwen recovery

Use the original admitted model factory, all 28 Qwen2.5-1.5B blocks, BF16 frozen
backbone, FP32 trainable interface/LoRA, FP64 pooling and the unchanged original
pilot optimizer helper. AdamW uses source learning rate 0.0001, weight decay
0.01 and gradient norm clipping 1. PEMS08 seed 11 and the first two origins
inside the 5% support block match the historical support-only smoke inputs.
No new target, verification or test cohort is used.

Verify all cached PINNED_FILES.json config/safetensors hashes before and after
the check, without refreshing the marker or downloading model/data assets.
Use the existing pinned project CUDA environment, one visible BF16-capable
GPU and five CPUs. Environment repair, if required, remains project-private
inside Slurm. The model cache and old checkpoints are never modified.

The coordinator launches three fresh Python processes sequentially:

1. Train an uninterrupted six-update path, saving every update.
2. Recreate the same seeded model, train three updates, checkpoint and exit
   with controlled code 75. The Python process ends; this is an intentional
   process interruption, not an HPC failure or a cancellation of another job.
3. Start a fresh process, reload the same pinned model and restore the saved
   trainable tensors, Adam state, CPU/CUDA RNG and update cursor, then continue
   from update three to update six.

Require bitwise equality of final trainable parameters, complete Adam state,
CPU/CUDA RNG, all six losses, cursor, contract and normalized support forecasts.
Run all four established invariance checks before/after each segment with the
unchanged 1e-3 absolute/relative tolerances. Preserve hashes/receipts, original
interruption cursor, process IDs, device/runtime/input fingerprints and peak
GPU allocation. A changed runtime/input identity cannot resume. A completed
segment reconstructed only for missing admission output retains its original
receipt and must reproduce the same support forecasts.

A fresh execution makes 12 engineering updates total (6+3+3), not 12 benchmark
experiments. Partial checkpoints and complete segments resume/reuse. Reports
separate total engineering work from new updates in this invocation, including
committed child progress on failure. Previous scheduler allocations must still
be included in final lifetime cost accounting.

This tests the full-size fixed-support source-optimizer path on the allocated
runtime. It does not test Slurm requeue/node failure, a future source episode
sampler, a new residual-head optimizer or their scheduler/episode cursors.
Those benchmark trainers need their own recovery checks when implemented.
It performs no ECTT objective, benchmark accuracy evaluation or S01 sign-off.

## Outputs and handoff

Stores are outputs/ectt/pretransfer_checks/IDENTITY/, with separate completed
development_review/ and qwen_recovery/ manifests. The Slurm report includes
SUMMARY.md, development_review.json, development_training_curves.csv,
qwen_recovery_result.json, transfer_check_result.json, live
transfer_check_progress.json and qwen_segment_progress.json. Child output
and failures remain in logs/ectt-transfer-check_JOBID.out/.err and the report's
failure_traceback.txt. No actual HPC evidence payload is copied into GitHub.

Refresh the pinned bootstrap and submit --stage shared-transfer-check --submit.
Check the printed job ID. Repeat that same stage after resolving a failure;
verified development reports and compatible Qwen segments/checkpoints reuse.
Actual full-size recovery is pending until the real HPC result is supplied.
Success prepares the next matched LLM-UTP/ordinary-Qwen implementation; it
does not mark B03/B04, the proposed method or independent review complete.
